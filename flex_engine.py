"""Rollout engine for the GRPO assignment (provided code; you do not need to change it).

A small vLLM-style inference engine in PyTorch, used by starter.py and zero_shot_eval.py

The paged-attention code (section 2) and the scheduler / CUDA-graph decoding (section 4) are adapted from
flex-nano-vllm (https://github.com/changjonathanc/flex-nano-vllm, MIT, Copyright (c) 2025 Jonathan Chang), whose
paged attention builds on attention-gym (https://github.com/pytorch-labs/attention-gym, BSD-3-Clause,
Copyright (c) 2023 Driss Guessous). See THIRD_PARTY_NOTICES.md.
"""
import math
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention, noop_mask
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

_flex_attention = torch.compile(flex_attention, fullgraph=True)
_create_block_mask = torch.compile(create_block_mask)


# ======================================================================================================================
# 1. Sampling settings
# ======================================================================================================================
@dataclass
class SamplingParams:
    """Sampling settings (same field names as vllm.SamplingParams)."""
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0  # only 1.0 is supported
    min_tokens: int = 0
    max_tokens: int = 16
    logprobs: int | None = None  # not None: return the sampled token's log-prob for every generated token
    stop: list = field(default_factory=list)
    include_stop_str_in_output: bool = True  # only True is supported


# ======================================================================================================================
# 2. Paged KV cache
# ======================================================================================================================
class PageTable:
    """Hands out fixed-size pages of a shared KV buffer to sequences.

    Each running sequence owns a *slot* (a row of the tables below). The KV buffer of every layer has shape
    [1, H_kv, n_pages * page_size, D], i.e. all pages laid end to end.
      page_table[slot, logical_page]          -> physical page (or -1)
      physical_to_logical[slot, physical_page] -> logical page (or -1)
    Slot 0 and page 0 are reserved: padding rows (of a packed prefill or a CUDA-graph batch) use slot 0, whose pages
    all map to page 0, so they never overwrite a real sequence's KV.
    """

    def __init__(self, n_pages: int, page_size: int, max_slots: int, device):
        self.n_pages, self.page_size, self.device = n_pages, page_size, device
        self.page_table = -torch.ones((max_slots, n_pages), dtype=torch.int64, device=device)
        self.page_table[0, :] = 0
        self.physical_to_logical = -torch.ones((max_slots, n_pages), dtype=torch.int64, device=device)
        self.pages = [[] for _ in range(max_slots)]  # CPU copy of each slot's physical pages
        self.capacity = [0] * max_slots  # tokens each slot can hold with its pages
        self.free_pages = list(reversed(range(1, n_pages)))
        self.free_slots = list(reversed(range(1, max_slots)))

    # --- bookkeeping (CPU; the GPU tables are updated in place, so captured CUDA graphs stay valid) ---
    def can_admit(self, n_tokens: int) -> bool:
        return bool(self.free_slots) and len(self.free_pages) * self.page_size >= n_tokens

    def can_grow(self, slot: int, n_tokens: int) -> bool:
        return n_tokens <= self.capacity[slot] or math.ceil((n_tokens - self.capacity[slot]) / self.page_size) <= len(self.free_pages)

    def admit(self) -> int:
        slot = self.free_slots.pop()
        self.capacity[slot] = 0
        self.page_table[slot, :] = -1
        self.physical_to_logical[slot, :] = -1
        return slot

    def reserve(self, slot: int, n_tokens: int) -> None:
        """Give `slot` enough pages to hold n_tokens."""
        if n_tokens <= self.capacity[slot]:
            return
        n_new = math.ceil((n_tokens - self.capacity[slot]) / self.page_size)
        assert n_new <= len(self.free_pages), "out of KV pages (check can_grow first)"
        start = self.capacity[slot] // self.page_size
        new_pages = self.free_pages[-n_new:]
        self.free_pages = self.free_pages[:-n_new]
        new = torch.tensor(new_pages, device=self.device)
        self.page_table[slot, start:start + n_new] = new
        self.physical_to_logical[slot, new] = torch.arange(start, start + n_new, device=self.device)
        self.pages[slot] += new_pages
        self.capacity[slot] += n_new * self.page_size

    def release(self, slot: int) -> None:
        """Free a finished (or preempted) sequence's slot and pages. GPU rows are reset when the slot is reused."""
        self.free_slots.append(slot)
        self.free_pages.extend(reversed(self.pages[slot]))
        self.pages[slot] = []

    # --- KV writes ---
    def write_decode(self, slots, pos, k, v, k_cache, v_cache):
        """Decode: write k, v [B, H, S, D] for slots [B] at positions pos [B, S]; return the whole caches."""
        B, H, S, D = k.shape
        page = torch.gather(self.page_table[slots], 1, (pos // self.page_size).to(torch.int64))
        addr = (page * self.page_size + pos % self.page_size).view(-1)  # [B*S] physical positions
        k_cache[:, :, addr, :] = k.permute(1, 0, 2, 3).reshape(1, H, B * S, D)
        v_cache[:, :, addr, :] = v.permute(1, 0, 2, 3).reshape(1, H, B * S, v.shape[-1])
        return k_cache, v_cache

    def write_prefill(self, slot_ids, pos, k, v, k_cache, v_cache):
        """Packed prefill: write k, v [1, H, L, D] (tokens of several prompts, slot_ids/pos [1, L]); return k, v
        unchanged, because a prompt attends only to its own fresh K/V."""
        addr = self.page_table[slot_ids, pos // self.page_size] * self.page_size + pos % self.page_size
        k_cache[:, :, addr.view(-1), :] = k
        v_cache[:, :, addr.view(-1), :] = v
        return k, v

    # --- block masks ---
    def causal_mask(self, n_slots: int, max_len: int) -> BlockMask:
        """Logical causal mask for every slot; decode masks are sliced from it."""
        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        return _create_block_mask(causal, B=n_slots, H=None, Q_LEN=max_len, KV_LEN=max_len,
                                  BLOCK_SIZE=self.page_size, device=self.device)

    def prefill_mask(self, slot_ids) -> BlockMask:
        """Packed prefill: causal within each prompt (document), nothing across prompts."""
        docs = slot_ids.view(-1)
        def document_causal(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (docs[q_idx] == docs[kv_idx])
        L = docs.shape[0]
        return _create_block_mask(document_causal, B=1, H=None, Q_LEN=L, KV_LEN=L, BLOCK_SIZE=self.page_size)

    def to_physical(self, mask: BlockMask, slots) -> BlockMask:
        """Turn a logical decode mask (KV blocks = a sequence's logical pages) into a mask over the shared buffer:
        KV block indices go through the page table, and mask_mod maps physical positions back to logical ones."""
        B, H, ROWS, MAX_COLS = mask.kv_indices.shape
        table = self.page_table[slots]

        def remap(num_blocks, indices):
            if num_blocks is None:
                return None, None
            phys = torch.zeros((B, H, ROWS, self.n_pages), dtype=torch.int32, device=indices.device)
            phys[:, :, :, :MAX_COLS] = torch.gather(table, 1, indices.view(B, -1).to(torch.int64)).view(indices.shape).to(torch.int32)
            return num_blocks.clone(), phys

        kv_num, kv_idx = remap(mask.kv_num_blocks, mask.kv_indices)
        full_num, full_idx = remap(mask.full_kv_num_blocks, mask.full_kv_indices)
        logical_mask_mod = mask.mask_mod or noop_mask

        def physical_mask_mod(b, h, q_idx, physical_kv_idx):
            slot = slots[b]
            logical_page = self.physical_to_logical[slot, physical_kv_idx // self.page_size]
            logical_kv_idx = (logical_page * self.page_size + physical_kv_idx % self.page_size).clamp(min=0)
            return torch.where(logical_page >= 0, logical_mask_mod(b, h, q_idx, logical_kv_idx), False)

        return BlockMask.from_kv_blocks(kv_num, kv_idx, full_num, full_idx, mask.BLOCK_SIZE, physical_mask_mod,
                                        seq_lengths=(mask.seq_lengths[0], self.n_pages * self.page_size))


# ======================================================================================================================
# 3. Qwen3 on the policy's own modules (only attention is replaced)
# ======================================================================================================================
class _PagedQwen3Attention(nn.Module):
    def __init__(self, hf_attn, page_table: PageTable, n_kv_heads: int, dtype, device):
        super().__init__()
        self.hf, self.page_table = hf_attn, page_table
        n = page_table.n_pages * page_table.page_size
        self.k_cache = torch.zeros((1, n_kv_heads, n, hf_attn.head_dim), dtype=dtype, device=device)
        self.v_cache = torch.zeros_like(self.k_cache)

    def forward(self, h, position_embeddings, mask, pos, slots, kernel_options):
        a, shape = self.hf, (*h.shape[:-1], -1, self.hf.head_dim)
        q = a.q_norm(a.q_proj(h).view(shape)).transpose(1, 2)
        k = a.k_norm(a.k_proj(h).view(shape)).transpose(1, 2)
        v = a.v_proj(h).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        write = self.page_table.write_decode if slots.ndim == 1 else self.page_table.write_prefill
        k, v = write(slots, pos, k, v, self.k_cache, self.v_cache)
        out = _flex_attention(q, k, v, scale=a.scaling, block_mask=mask, enable_gqa=True, kernel_options=kernel_options)
        return a.o_proj(out.transpose(1, 2).reshape(*h.shape[:-1], -1).contiguous())


class _PagedQwen3:
    """Forward pass of a HF Qwen3ForCausalLM with paged attention; every weight is the policy's own tensor."""

    def __init__(self, policy, page_table: PageTable):
        cfg = policy.config
        assert cfg.model_type == "qwen3", f"only Qwen3 is supported, got {cfg.model_type}"
        assert not getattr(cfg, "use_sliding_window", False)
        self.hf, self.lm_head, self.vocab_size = policy.model, policy.lm_head, cfg.vocab_size
        self.attn = [_PagedQwen3Attention(layer.self_attn, page_table, cfg.num_key_value_heads, policy.dtype, policy.device)
                     for layer in policy.model.layers]

    def __call__(self, input_ids, pos, mask, slots, kernel_options):
        """input_ids/pos: [B, S] (decode) or [1, L] (packed prefill); returns final hidden states."""
        h = self.hf.embed_tokens(input_ids)
        position_embeddings = self.hf.rotary_emb(h, pos)
        for layer, attn in zip(self.hf.layers, self.attn):
            h = h + attn(layer.input_layernorm(h), position_embeddings, mask, pos, slots, kernel_options)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.hf.norm(h)


# ======================================================================================================================
# 4. Engine: scheduling, prefill / decode, sampling, stop rules, CUDA graphs
# ======================================================================================================================
class _Sequence:
    def __init__(self, prompt: str, input_ids: torch.Tensor):
        self.prompt, self.input_ids = prompt, input_ids
        self.out_ids, self.out_logprobs = [], []
        self.slot, self.finish_reason = None, None

    @property
    def total_length(self) -> int:
        return len(self.input_ids) + len(self.out_ids)

    @property
    def token_ids(self) -> torch.Tensor:
        """Prompt + generated tokens (a preempted sequence is recomputed from all of them when re-admitted)."""
        if not self.out_ids:
            return self.input_ids
        return torch.cat([self.input_ids, torch.tensor(self.out_ids, dtype=self.input_ids.dtype)])


class _Engine:
    def __init__(self, policy, tokenizer, *, seed, max_num_seqs, max_model_len, kv_cache_tokens, page_size,
                 prefill_token_limit, kernel_options, prefill_kernel_options, capture_cudagraph):
        assert max_model_len % page_size == 0, "max_model_len must be a multiple of page_size"
        self.tokenizer, self.device = tokenizer, policy.device
        self.max_len, self.prefill_token_limit = max_model_len, prefill_token_limit
        self.kernel_options = kernel_options  # decode
        self.prefill_kernel_options = prefill_kernel_options | {"FORCE_USE_FLEX_ATTENTION": True}  # main (non-decoding) kernel
        self.max_slots = max_num_seqs + 1  # slot 0 is reserved
        self.page_table = PageTable(math.ceil(kv_cache_tokens / page_size) + 1, page_size, self.max_slots, self.device)
        self.model = _PagedQwen3(policy, self.page_table)
        self.causal = self.page_table.causal_mask(self.max_slots, max_model_len)
        self.input_pos = torch.zeros(self.max_slots, dtype=torch.int32, device=self.device)  # per-slot decode position
        eos = policy.generation_config.eos_token_id if policy.generation_config is not None else None
        self.eos_ids = set([eos] if isinstance(eos, int) else list(eos or [tokenizer.eos_token_id]))
        self.eos_tensor = torch.tensor(sorted(self.eos_ids), device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.capture_cudagraph, self.graphs = capture_cudagraph, None
        self.stop, self.stop_ids, self.stop_tail = [], set(), 0

    # --- forward passes ---
    def _prefill(self, seqs):
        """One packed forward pass over several sequences' tokens; returns next-token logits [n_seqs, V]."""
        ids = torch.cat([s.token_ids for s in seqs])
        pos = torch.cat([torch.arange(s.total_length) for s in seqs])
        slot_ids = torch.cat([torch.full((s.total_length,), s.slot) for s in seqs])
        last = torch.tensor([s.total_length for s in seqs]).cumsum(0) - 1
        pad = -ids.shape[0] % self.page_table.page_size  # padding tokens use the reserved slot 0
        ids, pos, slot_ids = (torch.nn.functional.pad(x, (0, pad)).view(1, -1).to(self.device) for x in (ids, pos, slot_ids))
        mask = self.page_table.prefill_mask(slot_ids)
        hidden = self.model(ids, pos, mask, slot_ids, self.prefill_kernel_options)
        return self.model.lm_head(hidden[0, last.to(self.device), :])

    def _decode_eager(self, slots, input_ids):
        """One decode step for slots [B]; positions come from self.input_pos. Returns logits [B, 1, V]."""
        B = slots.shape[0]
        pos = self.input_pos[slots]
        row = pos // self.causal.BLOCK_SIZE[0]
        c = self.causal
        full = (c.full_kv_num_blocks[slots, :, row].view(B, 1, 1), c.full_kv_indices[slots, :, row].view(B, 1, 1, -1)) \
            if c.full_kv_num_blocks is not None else (None, None)

        def causal_at(b, h, q_idx, kv_idx):
            return q_idx + pos[b] >= kv_idx
        mask = BlockMask.from_kv_blocks(c.kv_num_blocks[slots, :, row].view(B, 1, 1), c.kv_indices[slots, :, row].view(B, 1, 1, -1),
                                        *full, BLOCK_SIZE=c.BLOCK_SIZE, mask_mod=causal_at, seq_lengths=(1, c.seq_lengths[1]))
        mask = self.page_table.to_physical(mask, slots)
        hidden = self.model(input_ids.view(B, 1), pos.view(B, 1), mask, slots, self.kernel_options)
        return self.model.lm_head(hidden)

    def _capture_graphs(self):
        bs_max = self.max_slots
        self.graph_bs = [1, 2, 4, 8] + list(range(16, bs_max + 1, 16))
        ids = torch.zeros(bs_max, dtype=torch.int64, device=self.device)
        slots = torch.arange(bs_max, dtype=torch.int64, device=self.device)
        out = torch.zeros((bs_max, 1, self.model.vocab_size), device=self.device)
        self.graphs, pool = {}, None
        for bs in reversed(self.graph_bs):
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            out[:bs] = self._decode_eager(slots[:bs], ids[:bs])  # warm-up
            with torch.cuda.graph(graph, pool):
                out[:bs] = self._decode_eager(slots[:bs], ids[:bs])
            pool = pool or graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
        self.graph_vars = {"input_ids": ids, "slots": slots, "out": out}

    def _decode(self, slots, input_ids, pos):
        self.input_pos.zero_()
        self.input_pos[slots] = pos
        if self.graphs is None:
            return self._decode_eager(slots, input_ids)
        bs = input_ids.shape[0]
        g = self.graph_vars
        g["input_ids"].zero_(); g["slots"].zero_()  # unused rows -> reserved slot 0
        g["input_ids"][:bs], g["slots"][:bs] = input_ids, slots
        self.graphs[next(x for x in self.graph_bs if x >= bs)].replay()
        return g["out"][:bs]

    # --- sampling and stop rules ---
    def _sample(self, logits, seqs):
        logits = logits.float()
        if self.min_tokens > 0:  # like vLLM's min_tokens: no EOS before min_tokens generated tokens
            rows = [i for i, s in enumerate(seqs) if len(s.out_ids) < self.min_tokens]
            if rows:
                logits[torch.tensor(rows, device=logits.device).unsqueeze(1), self.eos_tensor.unsqueeze(0)] = float("-inf")
        if self.temperature == 0.0:
            tok = logits.argmax(dim=-1, keepdim=True)
            logp = torch.log_softmax(logits, dim=-1)
        else:
            logp = torch.log_softmax(logits / self.temperature, dim=-1)
            tok = torch.multinomial(logp.exp(), num_samples=1, generator=self.generator)
        for s, t, lp in zip(seqs, tok.view(-1).tolist(), logp.gather(-1, tok).view(-1).tolist()):
            s.out_ids.append(t)
            s.out_logprobs.append(lp)

    def _still_running(self, seqs):
        running = []
        for s in seqs:
            n, last, reason = len(s.out_ids), s.out_ids[-1], None
            if n >= self.min_tokens:  # vLLM skips all stop checks before min_tokens
                if last in self.eos_ids:
                    reason = "stop"
                elif last in self.stop_ids and any(x in self.tokenizer.decode(s.out_ids[-self.stop_tail:]) for x in self.stop):
                    reason = "stop"
            if reason is None and (n >= self.max_new_tokens or s.total_length >= self.max_len):
                reason = "length"
            if reason is None:
                running.append(s)
            else:
                s.finish_reason = reason
                self.page_table.release(s.slot)
        return running

    # --- scheduler ---
    def _step(self):
        """Prefill newly admitted prompts if any fit; otherwise one decode step for every running sequence
        (preempting the newest sequences when the cache is full)."""
        batch, n_tokens = [], 0
        while (self.waiting and self.page_table.can_admit(self.waiting[0].total_length)
               and (not batch or n_tokens + self.waiting[0].total_length < self.prefill_token_limit)):
            s = self.waiting.popleft()
            n_tokens += s.total_length
            s.slot = self.page_table.admit()
            self.page_table.reserve(s.slot, s.total_length)
            batch.append(s)
        if batch:
            self._sample(self._prefill(batch), batch)
            self.running.extend(self._still_running(batch))
            return
        while self.running:
            s = self.running.popleft()
            if self.page_table.can_grow(s.slot, s.total_length):
                self.page_table.reserve(s.slot, s.total_length)
                batch.append(s)
            else:  # out of pages: put s back, preempt the newest sequence
                self.running.appendleft(s)
                newest = self.running.pop()
                self.page_table.release(newest.slot)
                self.waiting.appendleft(newest)
        if not batch:
            if self.waiting and not self.running:
                raise RuntimeError(f"a {self.waiting[0].total_length}-token sequence does not fit in the KV cache")
            return
        slots = torch.tensor([s.slot for s in batch], dtype=torch.int64, device=self.device)
        ids = torch.tensor([s.out_ids[-1] for s in batch], dtype=torch.int64, device=self.device)
        pos = torch.tensor([s.total_length - 1 for s in batch], dtype=torch.int32, device=self.device)
        self._sample(self._decode(slots, ids, pos)[:, -1, :], batch)
        self.running = deque(self._still_running(batch))

    @torch.inference_mode()
    def generate(self, seqs, *, temperature, min_tokens, max_new_tokens, stop):
        self.temperature, self.min_tokens, self.max_new_tokens = float(temperature), int(min_tokens), int(max_new_tokens)
        if list(stop) != self.stop:  # a stop string can only complete on a token that contains its last character
            self.stop = list(stop)
            last_chars = {x[-1] for x in self.stop}
            texts = self.tokenizer.batch_decode([[i] for i in range(len(self.tokenizer))])
            self.stop_ids = {i for i, t in enumerate(texts) if any(c in t for c in last_chars)}
            self.stop_tail = max((len(x) for x in self.stop), default=0) + 2
        if self.capture_cudagraph and self.graphs is None:
            self._capture_graphs()
        self.waiting, self.running = deque(seqs), deque()
        while self.waiting or self.running:
            self._step()


# ======================================================================================================================
# 5. FlexLLM: the vLLM-like interface used by starter.py and zero_shot_eval.py
# ======================================================================================================================
# Kernel options (measured on A100 40/80GB, H100, H200, RTX PRO 6000 with bench_flex_kernels.py):
#  decode:  BLOCK_N=64 (BLOCK_M=16 is the minimum tile for 1 query x 2 GQA heads) is the fastest everywhere: 1.47-1.56x
#           faster than 32x32 on A100/H100/H200 (1.08x on RTX PRO 6000). PyTorch's own decode defaults are 28-40% slower
#           and need 107 KB of shared memory, more than the RTX PRO 6000 has (101 KB). BLOCK_N=128 does not fit anywhere;
#           SPLIT_KV, num_warps, num_stages and max-autotune gave no gain or failed.
#  prefill: PyTorch's defaults ({}) are 3-7% faster than 16x64.
DEFAULT_KERNEL_OPTIONS = {"BLOCK_M": 16, "BLOCK_N": 64}  # decode
DEFAULT_PREFILL_KERNEL_OPTIONS = {}  # PyTorch's defaults for the main kernel


class FlexLLM:
    """generate(prompts, SamplingParams) -> one result per prompt with .prompt, .prompt_token_ids and .outputs[i] with
    .text, .token_ids, .logprobs, .finish_reason ("stop" / "length"), like vLLM. Runs on the policy's own weights.

    Settings (recommended values per GPU: train_settings / eval_settings below):
      kv_cache_tokens: KV-cache capacity in tokens (~0.11 MB per token). If it cannot hold every running sequence,
                       the newest are paused and recomputed later (slower, still correct).
      max_num_seqs:    sequences decoded together.
      kernel_options:  FlexAttention options for decode steps (default {"BLOCK_M": 16, "BLOCK_N": 64}).
      prefill_kernel_options: options for prefill (default {}: PyTorch's defaults, the fastest there).
    """

    def __init__(self, policy, tokenizer, *, seed: int, max_num_seqs: int = 256, max_model_len: int = 2048,
                 kv_cache_tokens: int = 200_000, page_size: int = 128, prefill_token_limit: int = 16384,
                 capture_cudagraph: bool = True, kernel_options: dict | None = None,
                 prefill_kernel_options: dict | None = None):
        self.tokenizer = tokenizer
        self.llm_engine = SimpleNamespace(tokenizer=tokenizer)  # vLLM-style access to the tokenizer
        self.engine = _Engine(policy, tokenizer, seed=seed, max_num_seqs=max_num_seqs, max_model_len=max_model_len,
                              kv_cache_tokens=kv_cache_tokens, page_size=page_size, prefill_token_limit=prefill_token_limit,
                              kernel_options=dict(DEFAULT_KERNEL_OPTIONS if kernel_options is None else kernel_options),
                              prefill_kernel_options=dict(DEFAULT_PREFILL_KERNEL_OPTIONS if prefill_kernel_options is None
                                                          else prefill_kernel_options),
                              capture_cudagraph=capture_cudagraph)

    def generate(self, prompts: list[str], sampling_params: SamplingParams, use_tqdm: bool = False):
        sp = sampling_params
        assert getattr(sp, "top_p", 1.0) == 1.0, "only top_p=1.0 is supported"
        n = getattr(sp, "n", 1) or 1
        seqs = []
        for p in prompts:
            ids = self.tokenizer(p, return_tensors="pt")["input_ids"].squeeze(0)
            seqs += [_Sequence(p, ids) for _ in range(n)]
        self.engine.generate(seqs, temperature=sp.temperature, min_tokens=getattr(sp, "min_tokens", 0) or 0,
                             max_new_tokens=sp.max_tokens, stop=list(sp.stop or []))
        results = []
        for i, prompt in enumerate(prompts):
            outs = []
            for s in seqs[i * n:(i + 1) * n]:
                text = self.tokenizer.decode(s.out_ids, skip_special_tokens=True)
                if s.finish_reason == "stop":  # truncate after the first stop string, as vLLM does
                    cut = [text.find(x) + len(x) for x in self.engine.stop if x in text]
                    if cut:
                        text = text[:min(cut)]
                logprobs = [{t: SimpleNamespace(logprob=lp)} for t, lp in zip(s.out_ids, s.out_logprobs)] \
                    if getattr(sp, "logprobs", None) is not None else None
                outs.append(SimpleNamespace(text=text, token_ids=list(s.out_ids), logprobs=logprobs, finish_reason=s.finish_reason))
            results.append(SimpleNamespace(prompt=prompt, prompt_token_ids=seqs[i * n].input_ids.tolist(), outputs=outs))
        return results


# ======================================================================================================================
# 6. Recommended settings per GPU (used by starter.py / zero_shot_eval.py with --preset auto)
# ======================================================================================================================
# Measured in 12-step training runs (peak memory by nvidia-smi; seconds per step = generation + training):
#   A100 40GB:    L256 34.6 GiB, 5.0 + 5.3 s | L512 32.7 GiB, 9.9 + 9.8 s (microbatch 2)
#   A100 80GB:    L256 52.6 GiB, 4.6 + 4.9 s | L512 73.1 GiB, 8.9 + 8.1 s
#   H100 80GB:    L256 53.8 GiB, 3.6 + 3.2 s | L512 73.4 GiB, 7.1 + 5.1 s
#   H200:         L256 53.8 GiB, 2.8 + 2.3 s | L512 73.5 GiB, 5.5 + 3.4 s
#   RTX PRO 6000: L256 52.9 GiB, 3.6 + 3.9 s | L512 (200k cache) 51.5 GiB, 7.3 + 5.2 s
# kv_cache_tokens: a 128-rollout batch needs ~72k tokens at L256 and ~100k at L512; more mainly speeds up evaluation.
# max_num_seqs: 512 makes the full-test-set eval ~15% faster than 256 when the cache fits.
# block_m x block_n (decode): 16x64 is 1.47-1.56x faster than 32x32 on A100/H100/H200 and 1.08x on RTX PRO 6000;
#   it is also the only tested setting that is fast everywhere (see DEFAULT_KERNEL_OPTIONS).
# grad_acc_steps: microbatches per 128-rollout step (32 = microbatch 4). Larger microbatches are only 3-6% faster;
#   64 (microbatch 2) frees ~8 GiB, which is what lets L512 fit on a 40 GB card.
KV_BYTES_PER_TOKEN = 28 * 8 * 128 * 2 * 2  # Qwen3-1.7B: layers x KV heads x head dim x (K, V) x bf16
TRAIN_PRESETS = {  # memory class -> {max_tokens: settings}
    "40GB": {256: dict(kv_cache_tokens=100_000, max_num_seqs=256, grad_acc_steps=32),
             512: dict(kv_cache_tokens=100_000, max_num_seqs=256, grad_acc_steps=64)},
    "80GB": {256: dict(kv_cache_tokens=270_000, max_num_seqs=512, grad_acc_steps=32),
             512: dict(kv_cache_tokens=400_000, max_num_seqs=512, grad_acc_steps=32)},
    "141GB": {256: dict(kv_cache_tokens=270_000, max_num_seqs=512, grad_acc_steps=32),
              512: dict(kv_cache_tokens=400_000, max_num_seqs=512, grad_acc_steps=32)},
}
TILES = {8: (16, 64), 9: (16, 64), 12: (16, 64)}  # compute capability (major) -> (block_m, block_n)


def _gpu():
    p = torch.cuda.get_device_properties(0)
    tile = TILES.get(p.major, (32, 32))
    return p.name, p.total_memory / 2**30, {"block_m": tile[0], "block_n": tile[1]}


def train_settings(max_tokens: int) -> dict:
    """Preset for a training run at this generation budget on the current GPU."""
    name, gib, tiles = _gpu()
    if gib < 36:
        raise RuntimeError(f"{gib:.0f} GiB GPU: training this assignment needs at least a 40 GB GPU")
    cls = "141GB" if gib >= 120 else "80GB" if gib >= 60 else "40GB"
    budgets = sorted(TRAIN_PRESETS[cls])
    key = next((b for b in budgets if b >= max_tokens), budgets[-1])
    s = dict(TRAIN_PRESETS[cls][key])
    if max_tokens > key:  # beyond the measured budgets (e.g. L1024): halve the microbatch, cap concurrency
        s["grad_acc_steps"] = min(128, s["grad_acc_steps"] * 2)
        s["max_num_seqs"] = min(s["max_num_seqs"], max(128, s["kv_cache_tokens"] // (max_tokens + 256)))
        print(f"[flex_engine] L{max_tokens} was not benchmarked; using the L{key} preset with grad_acc_steps "
              f"{s['grad_acc_steps']} and max_num_seqs {s['max_num_seqs']}")
    return {**s, **tiles, "gpu": name, "memory_class": cls}


def eval_settings(max_tokens: int, max_num_seqs: int = 512, page_size: int = 128) -> dict:
    """Preset for evaluation only (no optimizer state): the largest useful KV cache that fits."""
    name, gib, tiles = _gpu()
    pages = -(-(200 + max_tokens) // page_size)  # prompt (~150-200 tokens) + response, in pages
    budget = int((gib - 8) * 2**30 / KV_BYTES_PER_TOKEN)  # leave ~8 GiB for weights and activations
    if max_num_seqs * pages * page_size > budget:
        max_num_seqs = max(64, budget // (pages * page_size))
    return {"kv_cache_tokens": min(budget, max_num_seqs * pages * page_size + 4 * page_size), "max_num_seqs": max_num_seqs,
            **tiles, "gpu": name}

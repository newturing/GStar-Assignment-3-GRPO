"""Zero-shot (or trained-checkpoint) evaluation on the Countdown test split with the flex engine.

Uses the same prompt (TEMPLATE) and reward function as starter.py, so implement Tasks 1-2 first.
    python zero_shot_eval.py --max_tokens 256              # also 512 and 1024
    python zero_shot_eval.py -m output/checkpoints/<run>    # a trained model
    python zero_shot_eval.py --max_tokens 512 --n_samples 4  # mean@4: 4 samples per problem, ~2x less sampling noise
"""
import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from flex_engine import FlexLLM, SamplingParams, eval_settings
from starter import build_prompt, reward_fn


def main():
    p = argparse.ArgumentParser(description="Countdown evaluation with the flex engine")
    p.add_argument("-m", "--model", default="Qwen/Qwen3-1.7B", help="model id or checkpoint directory")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--n_samples", type=int, default=1, help="samples per problem; >1 reports mean@k")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default=None, help="write <save_dir>/<model>_L<len>_k<k>.json with per-problem results")
    args = p.parse_args()
    K, L = args.n_samples, args.max_tokens

    settings = eval_settings(L)
    print(f"Evaluating {args.model} with max_tokens={L}, n_samples={K} | engine settings: {settings}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    llm = FlexLLM(model, tokenizer, seed=args.seed, kv_cache_tokens=settings["kv_cache_tokens"],
                  max_num_seqs=settings["max_num_seqs"], kernel_options={"BLOCK_M": settings["block_m"], "BLOCK_N": settings["block_n"]})
    sampling_params = SamplingParams(n=K, temperature=1.0, top_p=1.0, max_tokens=L, stop=["</answer>"],
                                     include_stop_str_in_output=True)

    data = load_dataset("justinphan3110/Countdown-Tasks-3to4", split="test")
    prompts = [build_prompt(tokenizer, ex["nums"], ex["target"], L) for ex in data]
    print(f"Generating {len(prompts)} x {K} responses...")
    outputs = llm.generate(prompts, sampling_params)

    N = len(outputs)
    per_problem, n_trunc, n_tok = [], 0, 0
    for out, ex in zip(outputs, data):
        gt = {"target": ex["target"], "numbers": ex["nums"]}
        per_problem.append(sum(reward_fn(o.text, gt) == 1.0 for o in out.outputs))
        n_trunc += sum(o.finish_reason == "length" for o in out.outputs)
        n_tok += sum(len(o.token_ids) for o in out.outputs)
    ci = [c / K for c in per_problem]
    nums = [len(ex["nums"]) for ex in data]
    summary = {
        "model": args.model, "max_tokens": L, "n_samples": K, "seed": args.seed, "n_problems": N,
        "accuracy": 100 * sum(ci) / N,  # mean@K
        # sampling standard error of the accuracy on this fixed test set (needs K > 1)
        "sampling_se": 100 * (sum(x * (1 - x) for x in ci) / (K - 1)) ** 0.5 / N if K > 1 else None,
        "pass_at_k": 100 * sum(c > 0 for c in per_problem) / N,
        "acc_3num": 100 * sum(c for c, n in zip(ci, nums) if n == 3) / nums.count(3),
        "acc_4num": 100 * sum(c for c, n in zip(ci, nums) if n == 4) / nums.count(4),
        "truncated_frac": n_trunc / (N * K), "avg_tokens": n_tok / (N * K),
    }
    se = f" (± {summary['sampling_se']:.2f})" if K > 1 else ""
    print(f"\nAccuracy{'' if K == 1 else f' (mean@{K})'}: {summary['accuracy']:.2f}%{se} | pass@{K} {summary['pass_at_k']:.2f}% | "
          f"3-number {summary['acc_3num']:.1f}% / 4-number {summary['acc_4num']:.1f}% | truncated {summary['truncated_frac']:.3f} | "
          f"avg tokens {summary['avg_tokens']:.0f}")
    print("EVAL_JSON " + json.dumps(summary))
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        name = os.path.basename(os.path.normpath(args.model))
        with open(os.path.join(args.save_dir, f"{name}_L{L}_k{K}.json"), "w") as f:
            json.dump({**summary, "per_problem_correct": per_problem}, f)


if __name__ == "__main__":
    main()

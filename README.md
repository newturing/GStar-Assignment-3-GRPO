# GRPO for Countdown

In this assignment you will implement **GRPO** and **Dr.GRPO** and train `Qwen/Qwen3-1.7B` to solve Countdown problems: reach a target number using the given numbers and `+ - * /`.

The handout is `gstar_assignment3.pdf`.

## Setup

```bash
bash setup_env.sh              # creates .venv: Python 3.11, torch 2.7.1 (CUDA 12.8), transformers 4.51.3, datasets
source .venv/bin/activate
```

## Run

```bash
# 1. Implement Tasks 1-2, then check them against the handout's asserts and get the zero-shot baselines:
python zero_shot_eval.py --max_tokens 256        # also 512 and 1024  (expect ~0.2%, ~15%, ~38%)

# 2. Implement Tasks 3-5, then train (80 steps, eval every 10):
python starter.py --loss_type grpo    --max_tokens 256
python starter.py --loss_type dr_grpo --max_tokens 256
python starter.py --loss_type grpo    --max_tokens 512
python starter.py --loss_type dr_grpo --max_tokens 512

# 3. Re-score a trained model (4 samples per problem halves the sampling noise):
python zero_shot_eval.py -m output/checkpoints/<run>_<timestamp> --max_tokens 512 --n_samples 4
```

- **Useful flags:** `--n_grpo_steps`, `--eval_every`, `--seed`, `--lr`, `--run_tag`, `--no-save_model`.
- **Outputs:**
  - TensorBoard logs and `train_metrics.jsonl` / `eval_metrics.jsonl` / `config.json` go to `output/tb/<loss>_L<len>[_tag]/<timestamp>/`.
  - Models go to `output/checkpoints/` (3.3 GB each).
  - `../GStar-Assignment-3-GRPO-Solution/plot_results.py` makes the report plots.

## Acknowledgments

We are grateful to [TinyZero](https://github.com/Jiayi-Pan/TinyZero) for the Countdown task and to the Stanford CS336 staff for Assignment 5, which served as a basis for this project. The rollout engine builds on Jonathan Chang's [flex-nano-vllm](https://github.com/changjonathanc/flex-nano-vllm) (MIT License; see `THIRD_PARTY_NOTICES.md`).

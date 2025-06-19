#!/bin/bash
set -euo pipefail

[ ! -e "output/test-run" ] || rm -r "output/test-run"

uv run python -m scripts.run_gpt2 \
    --output_dir "output" \
    --train_dataset "tirlimster/arc-small" \
    --run_name "test-run" \
    --add_row_sep \
    --segregate row \
    --train_batch_size 2 \
    --test_batch_size 1 \
    --hidden_size 4 \
    --num_hidden_layers 1 \
    --num_attention_heads 1 \
    --max_position_embeddings 24000 \
    --learning_rate 0.001 \
    --num_steps 500 \
    --scheduler_type constant \
    --log_every 50 \
    --eval_every 100 \
    --save_every 250 \
    --memory_tokens_strategy left \
    --num_memory_tokens 0 \
    --seed 98 \
    --disable_wandb \
    --use_tqdm \
    --use_bf16

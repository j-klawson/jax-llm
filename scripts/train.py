#!/usr/bin/env python3
"""CLI entry point for training the MiniGPT model."""
import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MiniGPT on a text dataset")
    parser.add_argument("--data-path", required=True, help="Path to text file with <|endoftext|> separators")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu", "tpu"], help="JAX backend device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--end-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--max-stories", type=int, default=None, help="Limit number of stories to load")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=2)

    args = parser.parse_args()

    # Set JAX backend before importing JAX
    os.environ["JAX_PLATFORMS"] = args.device

    from jax_llm.config import TrainConfig
    from jax_llm.train import train

    config = TrainConfig(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        peak_lr=args.peak_lr,
        end_lr=args.end_lr,
        weight_decay=args.weight_decay,
        warmup_fraction=args.warmup_fraction,
        max_stories=args.max_stories,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        log_every=args.log_every,
        device=args.device,
    )

    train(config)


if __name__ == "__main__":
    main()

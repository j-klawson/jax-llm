#!/usr/bin/env python3
"""CLI entry point for text generation with a trained MiniGPT model."""
import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a trained MiniGPT model")
    parser.add_argument("--checkpoint", required=True, help="Path to Orbax checkpoint directory")
    parser.add_argument("--prompt", default="Once upon a time", help="Text prompt to start generation")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum new tokens to generate")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    platform = "cuda" if args.device == "gpu" else args.device
    os.environ["JAX_PLATFORMS"] = platform

    import jax
    import flax.nnx as nnx
    import orbax.checkpoint
    from jax.sharding import SingleDeviceSharding
    from pathlib import Path

    from jax_llm.model import MiniGPT
    from jax_llm.generate import generate_story

    # Create model with default config
    model = MiniGPT()

    # Restore checkpoint
    device = jax.devices(platform)[0]
    sharding = SingleDeviceSharding(device)
    restore_args = jax.tree_util.tree_map(
        lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding),
        nnx.state(model),
    )

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored_state = checkpointer.restore(
        str(checkpoint_path),
        item=nnx.state(model),
        restore_args=restore_args,
    )
    nnx.update(model, restored_state)

    # Generate
    output = generate_story(
        model,
        args.prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        seed=args.seed,
    )
    print(output)


if __name__ == "__main__":
    main()

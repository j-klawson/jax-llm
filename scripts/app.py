#!/usr/bin/env python3
"""Gradio web UI for MiniGPT story generation."""
import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Gradio UI for MiniGPT")
    parser.add_argument("--checkpoint", default="checkpoints/model.orbax")
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    platform = "cuda" if args.device == "gpu" else args.device
    os.environ["JAX_PLATFORMS"] = platform

    import random
    from pathlib import Path

    import gradio as gr
    import jax
    import flax.nnx as nnx
    import orbax.checkpoint
    from jax.sharding import SingleDeviceSharding

    from jax_llm.model import MiniGPT
    from jax_llm.generate import generate_story

    # Load model once at startup
    print("Loading model...")
    model = MiniGPT()
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
    print("Model loaded!")

    def generate(prompt: str, temperature: float, max_tokens: int, repetition_penalty: float) -> str:
        seed = random.randint(0, 2**31)
        return generate_story(
            model, prompt,
            temperature=temperature,
            max_new_tokens=max_tokens,
            seed=seed,
            repetition_penalty=repetition_penalty,
        )

    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(label="Prompt", value="Once upon a time", lines=2),
            gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Temperature"),
            gr.Slider(10, 500, value=200, step=10, label="Max tokens"),
            gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="Repetition penalty"),
        ],
        outputs=gr.Textbox(label="Generated story", lines=10),
        title="MiniGPT Story Generator",
        description="A small GPT trained on TinyStories. Enter a prompt and generate a short story.",
    )

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

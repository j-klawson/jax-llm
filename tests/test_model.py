"""Smoke tests for MiniGPT model."""
import jax.numpy as jnp
import flax.nnx as nnx

from jax_llm.config import ModelConfig
from jax_llm.model import MiniGPT


def test_forward_pass_shape() -> None:
    config = ModelConfig(maxlen=16, vocab_size=100, embed_dim=32, num_heads=2, feed_forward_dim=64, num_transformer_blocks=2)
    model = MiniGPT(config=config, rngs=nnx.Rngs(42))

    batch_size = 4
    seq_len = 16
    tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    logits = model(tokens)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_default_instantiation() -> None:
    model = MiniGPT()
    tokens = jnp.ones((1, 8), dtype=jnp.int32)
    logits = model(tokens)
    assert logits.shape == (1, 8, 50_257)

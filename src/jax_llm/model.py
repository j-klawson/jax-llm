import jax.numpy as jnp
import flax.nnx as nnx

from jax_llm.config import ModelConfig


class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs) -> None:
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)


class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, *, rngs: nnx.Rngs) -> None:
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        attn_out = self.attention(x, mask=mask)
        x = x + attn_out
        return x


class MiniGPT(nnx.Module):
    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        maxlen: int = 128,
        vocab_size: int = 50_257,
        embed_dim: int = 192,
        num_heads: int = 6,
        feed_forward_dim: int = 512,
        num_transformer_blocks: int = 6,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        if config is not None:
            maxlen = config.maxlen
            vocab_size = config.vocab_size
            embed_dim = config.embed_dim
            num_heads = config.num_heads
            feed_forward_dim = config.feed_forward_dim
            num_transformer_blocks = config.num_transformer_blocks

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.maxlen = maxlen
        self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, rngs=rngs)
        self.transformer_blocks = nnx.List([
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
            for _ in range(num_transformer_blocks)
        ])
        self.output_layer = nnx.Linear(embed_dim, vocab_size, use_bias=False, rngs=rngs)

    def causal_attention_mask(self, seq_len: int) -> jnp.ndarray:
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)
        x = self.embedding(token_ids)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
        logits = self.output_layer(x)
        return logits

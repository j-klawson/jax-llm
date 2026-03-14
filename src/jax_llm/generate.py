import jax
import jax.numpy as jnp
import tiktoken

from jax_llm.model import MiniGPT


def generate_text(
    model: MiniGPT,
    prompt_tokens: list[int],
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    seed: int = 0,
    repetition_penalty: float = 1.2,
) -> list[int]:
    """Autoregressive token-by-token generation with temperature sampling."""
    tokens = list(prompt_tokens)
    key = jax.random.PRNGKey(seed)
    eot_token = 50256  # <|endoftext|> token id for GPT-2

    for _ in range(max_new_tokens):
        # Truncate to maxlen if needed
        input_tokens = tokens[-model.maxlen :]
        seq_len = len(input_tokens)
        # Pad to fixed maxlen so JAX compiles the model only once
        padded = input_tokens + [0] * (model.maxlen - seq_len)
        input_array = jnp.array([padded], dtype=jnp.int32)
        logits = model(input_array)
        # Get logits for the actual last token position (not the padding)
        next_token_logits = logits[0, seq_len - 1, :]

        # Penalize tokens that have already appeared
        if repetition_penalty != 1.0:
            seen = jnp.array(list(set(tokens)), dtype=jnp.int32)
            penalty = jnp.ones_like(next_token_logits)
            penalty = penalty.at[seen].set(repetition_penalty)
            # Divide positive logits, multiply negative logits (shrinks toward 0)
            next_token_logits = jnp.where(
                next_token_logits > 0,
                next_token_logits / penalty,
                next_token_logits * penalty,
            )

        if temperature > 0:
            scaled_logits = next_token_logits / temperature
            key, subkey = jax.random.split(key)
            next_token = jax.random.categorical(subkey, scaled_logits)
        else:
            next_token = jnp.argmax(next_token_logits)

        next_token = int(next_token)
        tokens.append(next_token)

        if next_token == eot_token:
            break

    return tokens


def generate_story(
    model: MiniGPT,
    prompt: str,
    temperature: float = 0.8,
    max_new_tokens: int = 50,
    seed: int = 0,
    repetition_penalty: float = 1.2,
) -> str:
    """Generate text from a string prompt, returning decoded text."""
    tokenizer = tiktoken.get_encoding("gpt2")
    prompt_tokens = tokenizer.encode(prompt)
    output_tokens = generate_text(
        model,
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
        repetition_penalty=repetition_penalty,
    )
    return tokenizer.decode(output_tokens)

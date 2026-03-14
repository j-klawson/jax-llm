# jax-llm

A GPT-style LLM built with JAX/Flax NNX, based on the DeepLearning.AI course.

## Project structure

- `src/jax_llm/` — installable package (model, data, training, generation, config)
- `scripts/` — CLI entry points (train.py, generate.py)
- `tests/` — pytest smoke tests
- `notebooks/` — original course notebooks (L2-L5) for reference

## Development

- Install: `pip install -e ".[dev]"`
- Test: `pytest tests/`
- Train: `python scripts/train.py --data-path <file> [options]`
- Generate: `python scripts/generate.py --checkpoint <path> [options]`

## Conventions

- Use type hints on all functions
- Config dataclasses in `config.py` — don't scatter magic numbers
- `--device cpu|gpu|tpu` sets `JAX_PLATFORMS` env var before JAX import
- Dataset format: plaintext with `<|endoftext|>` as document separator

# jax-llm

A small GPT-style language model built with JAX based on the course [Build and Train an LLM with JAX](https://learn.    deeplearning.ai/courses/build-and-train-an-llm-with-jax/information). 

## Setup

```bash
pip install -e ".[dev]"
```

## Training

```bash
python scripts/train.py --data-path data/TinyStories-1000.txt --max-stories 100 --num-epochs 3
```

Use `--device gpu` or `--device tpu` to train on accelerators.

## Generation

```bash
python scripts/generate.py --checkpoint checkpoints/model.orbax --prompt "Once upon a time"
```

## Tests

```bash
pytest tests/
```

## Attribution

The model architecture and training code are based on the DeepLearning.AI course [Build and Train an LLM with JAX](https://learn.deeplearning.ai/courses/build-and-train-an-llm-with-jax/information).

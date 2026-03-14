from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import orbax.checkpoint
import tiktoken

from jax_llm.config import ModelConfig, TrainConfig
from jax_llm.data import load_stories_from_file, create_dataloader
from jax_llm.generate import generate_story
from jax_llm.model import MiniGPT


def loss_fn(
    model: nnx.Module, batch: tuple[jnp.ndarray, jnp.ndarray]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    inputs, targets = batch
    logits = model(inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss, logits


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch: tuple[jnp.ndarray, jnp.ndarray],
) -> None:
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch[1])
    optimizer.update(model, grads)


def train(config: TrainConfig) -> nnx.Module:
    """Full training loop."""
    model_config = ModelConfig()
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load data
    stories = load_stories_from_file(config.data_path, max_stories=config.max_stories)
    dataloader, batches_per_epoch = create_dataloader(
        stories=stories,
        tokenizer=tokenizer,
        maxlen=model_config.maxlen,
        batch_size=config.batch_size,
        shuffle=True,
        num_epochs=1,
        seed=config.seed,
        worker_count=0,
    )

    # Create model and optimizer
    model = MiniGPT(config=model_config, rngs=nnx.Rngs(config.seed))

    total_steps = batches_per_epoch * config.num_epochs
    warmup_steps = max(1, int(total_steps * config.warmup_fraction))
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=config.end_lr,
    )

    optimizer = nnx.Optimizer(
        model, optax.adamw(learning_rate=lr_schedule, weight_decay=config.weight_decay),
        wrt=nnx.Param,
    )
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    prep_target_batch = jax.vmap(
        lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0])))
    )

    metrics_history: list[float] = []
    global_step = 0

    for epoch in range(config.num_epochs):
        # Recreate dataloader each epoch so Grain yields a fresh pass
        if epoch > 0:
            dataloader, _ = create_dataloader(
                stories=stories,
                tokenizer=tokenizer,
                maxlen=model_config.maxlen,
                batch_size=config.batch_size,
                shuffle=True,
                num_epochs=1,
                seed=config.seed + epoch,
                worker_count=0,
            )

        step = 0
        for batch in dataloader:
            input_batch = jnp.array(jnp.array(batch).T).astype(jnp.int32)
            target_batch = prep_target_batch(jnp.array(jnp.array(batch).T)).astype(jnp.int32)

            train_step(model, optimizer, metrics, (input_batch, target_batch))
            global_step += 1

            if (step + 1) % config.log_every == 0:
                computed = metrics.compute()
                loss_val = float(computed["loss"])
                metrics_history.append(loss_val)
                metrics.reset()
                current_lr = lr_schedule(global_step)
                print(
                    f"Epoch {epoch + 1}/{config.num_epochs}, "
                    f"Step {step + 1}/{batches_per_epoch}, "
                    f"Loss: {loss_val:.4f}, "
                    f"LR: {current_lr:.2e}"
                )
            step += 1

    # Save checkpoint
    checkpoint_path = Path(config.checkpoint_dir) / "model.orbax"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer.save(str(checkpoint_path.resolve()), nnx.state(model), force=True)
    print(f"Model saved to {checkpoint_path}")

    # Generate a sample
    print("\nSample generation:")
    sample = generate_story(model, "Once upon a time", temperature=0.8, max_new_tokens=50)
    print(sample)

    return model

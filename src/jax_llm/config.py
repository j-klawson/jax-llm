from dataclasses import dataclass


@dataclass
class ModelConfig:
    maxlen: int = 128
    vocab_size: int = 50_257
    embed_dim: int = 192
    num_heads: int = 6
    feed_forward_dim: int = 512
    num_transformer_blocks: int = 6


@dataclass
class TrainConfig:
    data_path: str = ""
    batch_size: int = 32
    num_epochs: int = 3
    peak_lr: float = 3e-4
    end_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    max_stories: int | None = None
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    log_every: int = 2
    device: str = "cpu"

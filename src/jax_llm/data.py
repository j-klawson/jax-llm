import csv
from pathlib import Path

import grain.python as grain
import tiktoken


def load_stories_from_file(path: str | Path, max_stories: int | None = None) -> list[str]:
    """Read stories from a text file (split on <|endoftext|>) or a CSV with a 'text' column."""
    path = Path(path)
    print(f"Loading stories from {path}...")

    if path.suffix.lower() == ".csv":
        stories = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["text"].strip()
                if text:
                    stories.append(text + "<|endoftext|>")
                if max_stories is not None and len(stories) >= max_stories:
                    break
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = f.read()
        stories = [s.strip() for s in data.split("<|endoftext|>") if s.strip()]
        stories = [s + "<|endoftext|>" for s in stories]
        if max_stories is not None:
            stories = stories[:max_stories]

    print(f"Loaded {len(stories)} stories")
    return stories


class StoryDataset:
    def __init__(self, stories: list[str], maxlen: int, tokenizer: tiktoken.Encoding) -> None:
        self.stories = stories
        self.maxlen = maxlen
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.stories)

    def __getitem__(self, idx: int) -> list[int]:
        story = self.stories[idx]
        tokens = self.tokenizer.encode(story, allowed_special={"<|endoftext|>"})
        if len(tokens) > self.maxlen:
            tokens = tokens[: self.maxlen]
        tokens.extend([0] * (self.maxlen - len(tokens)))
        return tokens


def create_dataloader(
    stories: list[str],
    tokenizer: tiktoken.Encoding,
    maxlen: int,
    batch_size: int,
    shuffle: bool = False,
    num_epochs: int = 1,
    seed: int = 42,
    worker_count: int = 0,
) -> tuple[grain.DataLoader, int]:
    """Create a Grain DataLoader from a list of story strings."""
    dataset = StoryDataset(stories, maxlen, tokenizer)
    estimated_batches = len(dataset) // batch_size

    sampler = grain.IndexSampler(
        num_records=len(dataset),
        shuffle=shuffle,
        seed=seed,
        shard_options=grain.NoSharding(),
        num_epochs=num_epochs,
    )
    dataloader = grain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=True)],
        worker_count=worker_count,
    )
    return dataloader, estimated_batches


def load_and_preprocess_data(
    file_path: str | Path,
    batch_size: int = 32,
    maxlen: int = 128,
    max_stories: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> tuple[grain.DataLoader, int]:
    """Convenience wrapper: load stories from file and create a dataloader."""
    print(f"Loading data from {file_path} (max {max_stories} stories)")
    stories = load_stories_from_file(file_path, max_stories=max_stories)
    tokenizer = tiktoken.get_encoding("gpt2")

    dataloader, estimated_batches = create_dataloader(
        stories=stories,
        tokenizer=tokenizer,
        maxlen=maxlen,
        batch_size=batch_size,
        shuffle=shuffle,
        num_epochs=1,
        seed=seed,
        worker_count=0,
    )
    print(f"Estimated batches per epoch: {estimated_batches}")
    print(f"Created DataLoader with batch_size={batch_size}, maxlen={maxlen}")
    return dataloader, estimated_batches

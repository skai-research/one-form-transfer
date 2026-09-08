"""Tokenize a text column of one or more Hugging Face datasets into sharded .bin files.

Datasets are interleaved in proportion to their size (sampling without replacement) and
shuffled through a buffer. Every document is prefixed with the <|eos|> token. The train
split goes to data_train_*.bin and the validation split to data_val_*.bin.
"""
import argparse
import json
import multiprocessing as mp
import os
import random

import numpy as np
from datasets import load_dataset, load_dataset_builder, load_from_disk
from tokenizers import Tokenizer

MAGIC, VERSION, HEADER_INTS = 20240520, 2, 256


def write_shard(path, tokens):
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0], header[1], header[2], header[3] = MAGIC, VERSION, len(tokens), 32
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint32).tobytes())


def open_split(name, split):
    """(size, iterator) of a split of a Hub dataset (streamed) or a local save_to_disk directory."""
    if os.path.isdir(name):
        ds = load_from_disk(name)[split]
        return len(ds), iter(ds)
    return load_dataset_builder(name).info.splits[split].num_examples, iter(load_dataset(name, split=split, streaming=True))


def interleave(datasets, split, seed, buffer_size):
    """Yield documents from all datasets, choosing each next source with probability
    proportional to its remaining size, then shuffling through a fixed-size buffer."""
    rng = random.Random(seed)
    remaining, streams = map(list, zip(*(open_split(d, split) for d in datasets)))
    buffer = []
    while any(remaining):
        i = rng.choices(range(len(streams)), weights=remaining)[0]
        remaining[i] -= 1
        buffer.append(next(streams[i]))
        if len(buffer) >= buffer_size:
            yield buffer.pop(rng.randrange(len(buffer)))
    rng.shuffle(buffer)
    yield from buffer


_tokenizer = None


def encode(args_and_text):
    global _tokenizer
    tokenizer_path, eos, text = args_and_text
    if _tokenizer is None:
        _tokenizer = Tokenizer.from_file(tokenizer_path)
    return np.array([eos] + _tokenizer.encode(text).ids, dtype=np.uint32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True, help="Hub dataset ids or local save_to_disk directories")
    p.add_argument("--column", required=True, help="text column: text, ipa_stripped or romanized")
    p.add_argument("--tokenizer", required=True, help="tokenizer.json")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--shard-size", type=int, required=True, help="tokens per shard")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--shuffle-buffer", type=int, default=1_000_000)
    p.add_argument("--max-docs", type=int, default=0, help="for smoke tests; 0 = all documents")
    args = p.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos = tokenizer.token_to_id("<|eos|>")
    assert eos is not None, "tokenizer has no <|eos|> token"
    os.makedirs(args.output, exist_ok=True)
    totals = {}
    with mp.Pool(max(1, os.cpu_count() - 2)) as pool:
        for split, name in [(args.val_split, "val"), (args.train_split, "train")]:
            docs = ((args.tokenizer, eos, row[args.column] or "")
                    for row in interleave(args.datasets, split, args.seed, args.shuffle_buffer))
            if args.max_docs:
                import itertools
                docs = itertools.islice(docs, args.max_docs)
            shard, count, index = np.empty(args.shard_size, dtype=np.uint32), 0, 0
            totals[name] = 0
            for tokens in pool.imap(encode, docs, chunksize=16):
                totals[name] += len(tokens)
                while len(tokens):
                    take = min(len(tokens), args.shard_size - count)
                    shard[count:count + take] = tokens[:take]
                    count += take
                    tokens = tokens[take:]
                    if count == args.shard_size:
                        write_shard(f"{args.output}/data_{name}_{index:06d}.bin", shard)
                        count, index = 0, index + 1
            if count:
                write_shard(f"{args.output}/data_{name}_{index:06d}.bin", shard[:count])
            print(f"{name}: {totals[name]:,} tokens in {index + 1} shard(s)")
    meta = dict(tokenizer=os.path.abspath(args.tokenizer), vocab_size=tokenizer.get_vocab_size(),
                eos=eos, column=args.column, datasets=args.datasets, train_tokens=totals["train"],
                val_tokens=totals["val"])
    json.dump(meta, open(f"{args.output}/meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()

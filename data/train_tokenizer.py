"""Train a byte-level BPE tokenizer on one text column of one or more Hugging Face datasets."""
import argparse
import os

from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
MIN_FREQUENCY = 2


def open_split(name, split):
    """A split of a Hub dataset (streamed) or of a local save_to_disk directory."""
    return load_from_disk(name)[split] if os.path.isdir(name) else load_dataset(name, split=split, streaming=True)


def documents(datasets, column, split, max_docs):
    seen = 0
    for name in datasets:
        for row in open_split(name, split):
            yield row[column] or ""
            seen += 1
            if max_docs and seen >= max_docs:
                return


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", required=True, help="Hub dataset ids or local save_to_disk directories")
    p.add_argument("--column", required=True, help="text column: text, ipa_stripped or romanized")
    p.add_argument("--split", default="train")
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--output", required=True, help="path of the tokenizer.json to write")
    p.add_argument("--max-docs", type=int, default=0, help="for smoke tests; 0 = all documents")
    args = p.parse_args()

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>", byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=MIN_FREQUENCY,
                                  special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(documents(args.datasets, args.column, args.split, args.max_docs), trainer)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tokenizer.save(args.output)
    print(f"vocab size {tokenizer.get_vocab_size()} -> {args.output}")


if __name__ == "__main__":
    main()

# On corpora of tens of billions of tokens, the most frequent byte pairs occur more than
# 2^31 times. The BPE trainer in huggingface/tokenizers (<= 0.23) accumulates pair counts in
# 32-bit integers, so these pairs overflow and are dropped from the merge list. Build the
# library from source with 64-bit counts before training on the full corpus:
#   tokenizers/src/models/bpe/trainer.rs
#   - fn count_pairs(...) -> (HashMap<Pair, i32>, HashMap<Pair, HashSet<usize>>)
#   + fn count_pairs(...) -> (HashMap<Pair, i64>, HashMap<Pair, HashSet<usize>>)
#   - *pair_counts.entry(cur_pair).or_insert(0) += counts[i] as i32;
#   + *pair_counts.entry(cur_pair).or_insert(0) += counts[i] as i64;
#   - let change = change * counts[iw] as i32;
#   + let change = change as i64 * counts[iw] as i64;
# then `pip install -e bindings/python` (requires Rust and maturin).

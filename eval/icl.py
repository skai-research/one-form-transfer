"""Zero- and few-shot evaluation on XStoryCloze and XCOPA.

Each candidate ending is scored by the log-likelihood of its tokens given the context
(optionally preceded by k demonstrations) and the prediction is the argmax. Demonstrations
are drawn per test example from the demonstration split.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pretrain"))
from model import load_checkpoint  # noqa: E402

TASKS = {  # dataset, evaluation split, demonstration split
    "xstorycloze": ("mugezhang/xstorycloze_eval_multirepr", "eval", "train"),
    "xcopa": ("mugezhang/xcopa_eval_multirepr", "test", "validation"),
}
COLUMN_SUFFIX = {"text": "", "ipa": "_ipa_stripped", "romanized": "_romanized"}
XCOPA_CONNECTOR = {"cause": {"text": "because", "romanized": "because", "ipa": "bɪkʌz"},
                   "effect": {"text": "so", "romanized": "so", "ipa": "soʊ"}}
DEMO_SEPARATOR = "\n\n"


def load_examples(task, dataset, language, split, representation):
    ds = load_from_disk(dataset)[split] if os.path.isdir(dataset) else load_dataset(dataset, language, split=split)
    sfx = COLUMN_SUFFIX[representation]
    examples = []
    for row in ds:
        if task == "xstorycloze":
            context = " ".join(row[f"input_sentence_{i}{sfx}"].strip() for i in range(1, 5))
            choices = [row[f"sentence_quiz{i}{sfx}"].strip() for i in (1, 2)]
            label = int(row["answer_right_ending"]) - 1
        else:
            context = f"{row['premise' + sfx].strip()} {XCOPA_CONNECTOR[row['question']][representation]}"
            choices = [row[f"choice{i}{sfx}"].strip() for i in (1, 2)]
            label = int(row["label"])
        examples.append((context, choices, label))
    return examples


def build_prompt(demos, context):
    return "".join(f"{c} {choices[label]}{DEMO_SEPARATOR}" for c, choices, label in demos) + context


@torch.no_grad()
def score(model, tokenizer, window, context, choice, device):
    """Sum and count of log-probabilities of the choice tokens given the context."""
    context_ids = tokenizer.encode(context).ids
    ids = tokenizer.encode(context + " " + choice).ids
    assert ids[:len(context_ids)] == context_ids, "tokenization of the context changed when the choice was appended"
    input_ids = torch.tensor([ids], device=device)
    logits = model.logits(input_ids, torch.ones_like(input_ids), window)[0]
    logp = F.log_softmax(logits[len(context_ids) - 1:-1], dim=-1)
    targets = input_ids[0, len(context_ids):]
    return logp.gather(1, targets[:, None]).sum().item(), len(targets)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--language", required=True)
    p.add_argument("--representation", choices=COLUMN_SUFFIX, required=True, help="representation of the inputs")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True, help="tokenizer.json of the checkpoint")
    p.add_argument("--k", type=int, nargs="+", required=True, help="numbers of demonstrations")
    p.add_argument("--reduce", choices=["sum", "mean"], required=True, help="log-likelihood reduction over candidate tokens")
    p.add_argument("--subsample", type=int, required=True, help="number of test examples (0 = all)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", required=True, help="JSON file with accuracies and per-example scores")
    p.add_argument("--dataset", default=None, help="override the Hub dataset with a local transcribe.py output")
    p.add_argument("--max-seq-len", type=int, default=4096)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model, config = load_checkpoint(args.checkpoint, device, args.max_seq_len, eos=tokenizer.token_to_id("<|eos|>"))
    hub, eval_split, demo_split = TASKS[args.task]
    dataset = args.dataset or hub
    test = load_examples(args.task, dataset, args.language, eval_split, args.representation)
    demos = load_examples(args.task, dataset, args.language, demo_split, args.representation) if max(args.k) else []

    results = dict(vars(args), window=config["window"], accuracy={}, examples={})
    for k in args.k:
        rng = np.random.default_rng(args.seed)
        if args.subsample and args.subsample < len(test):
            test_k = [test[i] for i in rng.choice(len(test), size=args.subsample, replace=False)]
        else:
            test_k = test
        records = []
        for context, choices, label in test_k:
            shots = [demos[j] for j in rng.choice(len(demos), size=k, replace=False)] if k else []
            prompt = build_prompt(shots, context)
            while shots and len(tokenizer.encode(prompt).ids) + max(len(tokenizer.encode(" " + c).ids) for c in choices) > args.max_seq_len:
                shots = shots[1:]
                prompt = build_prompt(shots, context)
            scores = []
            for choice in choices:
                total, n = score(model, tokenizer, config["window"], prompt, choice, device)
                scores.append(total if args.reduce == "sum" else total / n)
            records.append(dict(label=label, prediction=int(np.argmax(scores)), scores=scores))
        accuracy = float(np.mean([r["label"] == r["prediction"] for r in records]))
        results["accuracy"][k], results["examples"][k] = accuracy, records
        print(f"{args.task}/{args.language}/{args.representation} k={k} accuracy={accuracy:.4f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    json.dump(results, open(args.output, "w"), indent=1)


if __name__ == "__main__":
    main()

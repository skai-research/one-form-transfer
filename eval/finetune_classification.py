"""Fine-tune a pretrained GPT for classification: natural language
inference (XNLI / IndicXNLI / CDSC-E) and intent classification (MASSIVE).

A classification head reads the hidden state of the last token. Pairs of sentences are
joined with the end-of-text token. The checkpoint with the lowest mean validation loss
across languages is kept and evaluated on the test split (macro-F1 per language).
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from sklearn.metrics import f1_score
from tokenizers import Tokenizer
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pretrain"))
from model import load_checkpoint  # noqa: E402

COLUMN_SUFFIX = {"text": "", "ipa": "_ipa_stripped", "romanized": "_romanized"}
XNLI = {"en": ("mugezhang/xnli-en-ipa_ipa_romanized", None), "es": ("mugezhang/xnli-es-ipa_ipa_romanized", None),
        "ru": ("mugezhang/russian-xnli-ipa-rosetta_ipa_romanized", None), "hi": ("mugezhang/hindi-xnli-ipa_ipa_romanized", None),
        "ur": ("mugezhang/urdu-xnli-ipa_ipa_romanized", None), "ta": ("mugezhang/indicxnli_ipa_romanized", "ta"),
        "ml": ("mugezhang/indicxnli_ipa_romanized", "ml"), "pl": ("mugezhang/cdsc-e-ipa_ipa_romanized", None)}
MASSIVE = {lang: ("mugezhang/massive_ipa_romanized", locale) for lang, locale in
           [("en", "en-US"), ("es", "es-ES"), ("ru", "ru-RU"), ("pl", "pl-PL"), ("hi", "hi-IN"), ("ur", "ur-PK"),
            ("ta", "ta-IN"), ("ml", "ml-IN")]}
TASKS = {"xnli": dict(sources=XNLI, columns=("premise", "hypothesis"), label="label", num_classes=3),
         "massive": dict(sources=MASSIVE, columns=("utt",), label="intent", num_classes=60)}
DROPOUT, WEIGHT_DECAY, GRAD_CLIP = 0.1, 0.01, 1.0


def resolve(column_names, column, representation):
    """Column holding `column` in the requested representation (transcribe.py naming)."""
    for name in (column + COLUMN_SUFFIX[representation], COLUMN_SUFFIX[representation].strip("_") or column):
        if name in column_names:
            return name
    raise KeyError(f"no {representation} version of column {column!r} in {column_names}")


class Classifier(nn.Module):
    def __init__(self, backbone, dim, num_classes, window):
        super().__init__()
        self.backbone, self.window = backbone, window
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(dim, dim // 2), nn.ReLU(), nn.Dropout(DROPOUT),
                                  nn.Linear(dim // 2, num_classes))

    def forward(self, input_ids, attention_mask):
        h = self.backbone.hidden(input_ids, attention_mask, self.window)
        last = attention_mask.sum(1) - 1
        return self.head(h[torch.arange(h.size(0), device=h.device), last].float())


def batches(examples, batch_size, pad_id, device):
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, (seq, _) in enumerate(chunk):
            ids[j, :len(seq)], mask[j, :len(seq)] = torch.tensor(seq), 1
        yield ids.to(device), mask.to(device), torch.tensor([label for _, label in chunk], device=device)


@torch.no_grad()
def evaluate(model, examples, batch_size, pad_id, device):
    model.eval()
    loss, predictions, labels = 0.0, [], []
    for ids, mask, y in batches(examples, batch_size, pad_id, device):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, mask)
        loss += F.cross_entropy(logits.float(), y, reduction="sum").item()
        predictions += logits.argmax(-1).tolist()
        labels += y.tolist()
    model.train()
    return loss / len(examples), f1_score(labels, predictions, average="macro", zero_division=0), predictions


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--representation", choices=COLUMN_SUFFIX, required=True)
    p.add_argument("--languages", nargs="+", required=True, help="languages trained on jointly and evaluated")
    p.add_argument("--dataset", default=None, help="directory with one transcribe.py output per language, instead of the Hub sources")
    p.add_argument("--columns", nargs="+", default=None, help="text columns of --dataset (default: the task's)")
    p.add_argument("--checkpoint", required=True, help="pretrained GPT checkpoint")
    p.add_argument("--tokenizer", required=True, help="tokenizer.json of the checkpoint")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--warmup-ratio", type=float, required=True, help="fraction of steps with linear warmup; linear decay after")
    p.add_argument("--max-length", type=int, required=True, help="inputs are truncated to this many tokens")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--micro-batch-size", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=0, help="optimizer steps between validations (0 = each epoch)")
    p.add_argument("--train-samples", type=int, default=0, help="training examples per language (0 = all)")
    args = p.parse_args()
    task = TASKS[args.task]
    device = "cuda"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    encode, sep, pad_id = (lambda s: tokenizer.encode(s).ids), tokenizer.token_to_id("<|eos|>"), tokenizer.token_to_id("<|pad|>")
    backbone, config = load_checkpoint(args.checkpoint, eos=sep)
    model = Classifier(backbone.float(), config["model_dim"], task["num_classes"], config["window"]).to(device).train()

    def load(language):
        if args.dataset:
            return load_from_disk(os.path.join(args.dataset, language))
        repo, config = task["sources"][language]
        return load_dataset(repo, config)

    def encode_split(ds, columns):
        names = [resolve(ds.column_names, c, args.representation) for c in columns]
        out = []
        for row in ds:
            ids = []
            for name in names:
                ids += (ids and [sep] or []) + encode(row[name])
            out.append((ids[:args.max_length], int(row[task["label"]])))
        return out

    data = {}
    for language in args.languages:
        ds = load(language)
        columns = args.columns or (("sentence_A", "sentence_B") if language == "pl" and args.task == "xnli" and not args.dataset else task["columns"])
        if args.train_samples:
            ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(min(args.train_samples, len(ds["train"]))))
        data[language] = {split: encode_split(ds[split], columns) for split in ("train", "validation", "test")}
    train = [ex for language in args.languages for ex in data[language]["train"]]

    assert args.batch_size % args.micro_batch_size == 0, "batch size must be a multiple of the micro-batch size"
    accum = args.batch_size // args.micro_batch_size
    steps_per_epoch = len(train) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup = int(args.warmup_ratio * total_steps)
    decay, no_decay = [q for q in model.parameters() if q.ndim >= 2], [q for q in model.parameters() if q.ndim < 2]
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=WEIGHT_DECAY), dict(params=no_decay, weight_decay=0.0)], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: s / max(1, warmup) if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    log = open(os.path.join(args.output, "log.txt"), "a")
    log.write(json.dumps(vars(args)) + "\n")

    def validate(step):
        nonlocal best
        results = {language: evaluate(model, data[language]["validation"], args.micro_batch_size, pad_id, device)[:2]
                   for language in args.languages}
        mean_loss = sum(r[0] for r in results.values()) / len(results)
        line = f"step {step} val_loss {mean_loss:.4f} " + " ".join(f"{l}:f1={r[1]:.4f}" for l, r in results.items())
        print(line, flush=True)
        log.write(line + "\n")
        if mean_loss < best:
            best = mean_loss
            torch.save(dict(model=model.state_dict(), args=vars(args), step=step, val_loss=mean_loss),
                       os.path.join(args.output, "best.pt"))

    best, step = float("inf"), 0
    for epoch in range(args.epochs):
        random.shuffle(train)
        running, micro = 0.0, 0
        for ids, mask, y in batches(train[:steps_per_epoch * args.batch_size], args.micro_batch_size, pad_id, device):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(ids, mask).float(), y) / accum
            loss.backward()
            running, micro = running + loss.item(), micro + 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    log.write(f"step {step}/{total_steps} train_loss {running / 50:.4f}\n")
                    log.flush()
                    running = 0.0
                if args.eval_every and step % args.eval_every == 0:
                    validate(step)
        if not args.eval_every or step % args.eval_every:
            validate(step)

    model.load_state_dict(torch.load(os.path.join(args.output, "best.pt"), map_location=device)["model"])
    results = {}
    for language in args.languages:
        loss, f1, predictions = evaluate(model, data[language]["test"], args.micro_batch_size, pad_id, device)
        results[language] = dict(test_loss=loss, macro_f1=f1, predictions=predictions)
        print(f"test {language}: macro_f1 {f1:.4f}", flush=True)
    results["average_macro_f1"] = sum(r["macro_f1"] for r in results.values()) / len(args.languages)
    print(f"test average macro_f1 {results['average_macro_f1']:.4f}")
    json.dump(dict(args=vars(args), best_val_loss=best, results=results), open(os.path.join(args.output, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

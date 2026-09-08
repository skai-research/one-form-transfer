"""XL-Sum summarization: supervised fine-tuning of a pretrained GPT and generation with ROUGE
evaluation.

  train     the loss is computed on the summary tokens only; the article is truncated so that
            the whole prompt fits in --max-length; languages are upsampled to equal size; the
            checkpoint with the lowest mean validation loss across languages is kept
  generate  beam search from the fine-tuned checkpoint and ROUGE-1/2/L with the XL-Sum
            tokenization (lowercased words, punctuation split off)
"""
import argparse
import json
import os
import random
import sys
import unicodedata

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from rouge_score import rouge_scorer, tokenizers
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pretrain"))
from model import load_checkpoint  # noqa: E402

TEMPLATE = ("[LANG={lang}] Summarize the following article.\nArticle:\n", "\nSummary:\n")
COLUMNS = {"text": ("text", "summary"), "ipa": ("text_ipa_stripped", "summary_ipa_stripped"),
           "romanized": ("text_romanized", "summary_romanized")}
WEIGHT_DECAY, GRAD_CLIP = 0.01, 1.0
REPETITION_PENALTY, NO_REPEAT_NGRAM = 1.2, 3


# ---- model wrapper -----------------------------------------------------------------------
class LanguageModel:
    """Model plus tokenizer, loaded from a pretrained or a fine-tuned checkpoint."""

    def __init__(self, args, checkpoint=None):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint else None
        self.tok = Tokenizer.from_file(state["args"]["tokenizer"] if state else args.tokenizer)
        self.eos, self.pad = self.tok.token_to_id("<|eos|>"), self.tok.token_to_id("<|pad|>")
        self.model, self.config = load_checkpoint(checkpoint or args.checkpoint, "cuda", args.max_length, eos=self.eos)
        self.encode = lambda s: self.tok.encode(s).ids

    def decode(self, ids):
        return self.tok.decode(ids)

    def loss(self, ids, mask, labels):
        """Cross-entropy over the summary tokens only (logits are computed just for those positions)."""
        targets = labels[:, 1:]
        keep = targets != -100
        logits = self.model.head(self.model.hidden(ids, mask, self.config["window"])[:, :-1][keep])
        return F.cross_entropy(logits.float(), targets[keep])

    def parameters(self):
        return self.model.parameters()


# ---- data --------------------------------------------------------------------------------
def load_split(dataset, language, split):
    return load_from_disk(os.path.join(dataset, language))[split] if os.path.isdir(dataset) else load_dataset(dataset, language, split=split)


def prompt_ids(lm, language, article, budget):
    prefix, suffix = lm.encode(TEMPLATE[0].format(lang=language)), lm.encode(TEMPLATE[1])
    return prefix + lm.encode(article)[:max(0, budget - len(prefix) - len(suffix))] + suffix


def encode_example(lm, language, article, summary, max_length, max_target):
    target = lm.encode(summary)[:max_target] + [lm.eos]
    prompt = prompt_ids(lm, language, article, max_length - len(target))
    return prompt + target, [-100] * len(prompt) + target


def batches(examples, batch_size, pad, device):
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        labels = torch.full((len(chunk), width), -100, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, (seq, lab) in enumerate(chunk):
            ids[j, :len(seq)], labels[j, :len(seq)], mask[j, :len(seq)] = torch.tensor(seq), torch.tensor(lab), 1
        yield ids.to(device), mask.to(device), labels.to(device)


# ---- training ----------------------------------------------------------------------------
@torch.no_grad()
def validation_loss(lm, examples, batch_size):
    lm.model.eval()
    total, count = 0.0, 0
    for ids, mask, labels in batches(examples, batch_size, lm.pad, "cuda"):
        n = (labels[:, 1:] != -100).sum().item()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total, count = total + lm.loss(ids, mask, labels).item() * n, count + n
    lm.model.train()
    return total / count


def train(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.output, exist_ok=True)
    lm = LanguageModel(args)
    lm.model.float().train()
    data = {}
    for language in args.languages:
        for split in ("train", "validation"):
            ds = load_split(args.dataset, language, split)
            limit = args.val_samples if split == "validation" else args.train_samples
            if limit:
                ds = ds.shuffle(seed=args.seed).select(range(min(limit, len(ds))))
            src, tgt = COLUMNS[args.representation]
            data[language, split] = [encode_example(lm, language, row[src], row[tgt], args.max_length, args.max_target_tokens)
                                     for row in ds]
    largest = max(len(data[l, "train"]) for l in args.languages)
    train_examples = [ex for l in args.languages for ex in data[l, "train"] + rng.choices(data[l, "train"], k=largest - len(data[l, "train"]))]

    assert args.batch_size % args.micro_batch_size == 0, "batch size must be a multiple of the micro-batch size"
    accum = args.batch_size // args.micro_batch_size
    steps_per_epoch = len(train_examples) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup = int(args.warmup_ratio * total_steps)
    decay, no_decay = [q for q in lm.parameters() if q.ndim >= 2], [q for q in lm.parameters() if q.ndim < 2]
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=WEIGHT_DECAY), dict(params=no_decay, weight_decay=0.0)], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: s / max(1, warmup) if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    log = open(os.path.join(args.output, "log.txt"), "a")
    log.write(json.dumps(vars(args)) + "\n")

    def validate(step):
        nonlocal best
        losses = {l: validation_loss(lm, data[l, "validation"], args.micro_batch_size) for l in args.languages}
        mean = sum(losses.values()) / len(losses)
        line = f"step {step} val_loss {mean:.4f} " + " ".join(f"{l}:{v:.4f}" for l, v in losses.items())
        print(line, flush=True)
        log.write(line + "\n")
        if mean < best:
            best = mean
            torch.save(dict(model=lm.model.state_dict(), config=lm.config, args=vars(args), step=step, val_loss=mean),
                       os.path.join(args.output, "best.pt"))

    best, step = float("inf"), 0
    for epoch in range(args.epochs):
        rng.shuffle(train_examples)
        running, micro = 0.0, 0
        for ids, mask, labels in batches(train_examples[:steps_per_epoch * args.batch_size], args.micro_batch_size, lm.pad, "cuda"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lm.loss(ids, mask, labels) / accum
            loss.backward()
            running, micro = running + loss.item(), micro + 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(lm.parameters(), GRAD_CLIP)
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
    print(f"best validation loss {best:.4f}")


# ---- generation --------------------------------------------------------------------------
def penalize(logits, generated):
    """Repetition penalty on already generated tokens and a ban on repeating n-grams."""
    if generated.size(1):
        seen = logits.gather(1, generated)
        logits.scatter_(1, generated, torch.where(seen < 0, seen * REPETITION_PENALTY, seen / REPETITION_PENALTY))
    n = NO_REPEAT_NGRAM
    if generated.size(1) >= n:
        grams = generated.unfold(1, n, 1)                                  # (rows, count, n)
        match = (grams[:, :, :-1] == generated[:, None, -(n - 1):]).all(-1)  # earlier (n-1)-grams equal to the current suffix
        banned = torch.where(match, grams[:, :, -1], logits.size(1))        # index V = dummy column
        logits = torch.cat([logits, logits.new_zeros(logits.size(0), 1)], 1).scatter_(1, banned, float("-inf"))[:, :-1]
    return logits


@torch.no_grad()
def beam_search(lm, prompts, num_beams, max_new_tokens):
    """Batched beam search with left-padded prompts; returns one token list per prompt."""
    E, K, device = len(prompts), num_beams, "cuda"
    width = max(len(p) for p in prompts)
    ids = torch.full((E, width), lm.pad, dtype=torch.long, device=device)
    mask = torch.zeros((E, width), dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        ids[i, width - len(p):], mask[i, width - len(p):] = torch.tensor(p, device=device), 1
    ids, mask = ids.repeat_interleave(K, 0), mask.repeat_interleave(K, 0)
    cache = [{} for _ in lm.model.blocks]
    logits = lm.model.head(lm.model.hidden(ids, mask, lm.config["window"], cache)[:, -1])
    scores = torch.full((E, K), float("-inf"), device=device)
    scores[:, 0] = 0
    generated = torch.zeros((E * K, 0), dtype=torch.long, device=device)
    finished = [[] for _ in range(E)]
    for step in range(max_new_tokens):
        logp = F.log_softmax(penalize(logits, generated), -1)
        V = logp.size(1)
        top_scores, top_idx = (scores.view(-1, 1) + logp).view(E, K * V).topk(2 * K, dim=1)
        rows, next_scores, tokens = [], [], []
        for e in range(E):
            kept = 0
            for s, idx in zip(top_scores[e].tolist(), top_idx[e].tolist()):
                row, token = e * K + idx // V, idx % V
                if s == float("-inf") or kept == K:
                    break
                if token == lm.eos:
                    if len(finished[e]) < K:
                        finished[e].append((s / (step + 1), generated[row].tolist()))
                else:
                    rows.append(row), next_scores.append(s), tokens.append(token)
                    kept += 1
            while kept < K:  # dead beams
                rows.append(e * K), next_scores.append(float("-inf")), tokens.append(lm.pad)
                kept += 1
        rows, tokens = torch.tensor(rows, device=device), torch.tensor(tokens, device=device)
        generated = torch.cat([generated[rows], tokens[:, None]], 1)
        scores = torch.tensor(next_scores, device=device).view(E, K)
        if all(len(f) >= K for f in finished):
            break
        for layer in cache:
            if layer:
                layer["k"], layer["v"] = layer["k"][rows], layer["v"][rows]
        mask = torch.cat([mask, torch.ones(E * K, 1, dtype=torch.long, device=device)], 1)
        logits = lm.model.head(lm.model.hidden(tokens[:, None], mask, lm.config["window"], cache)[:, -1])
    outputs = []
    for e in range(E):
        live = [(scores[e, b].item() / generated.size(1), generated[e * K + b].tolist()) for b in range(K)]
        outputs.append(max(finished[e] + live)[1])
    return outputs


class XlsumTokenizer(tokenizers.Tokenizer):
    """Tokenization used by the XL-Sum ROUGE evaluation: lowercase, split punctuation."""

    def tokenize(self, text):
        tokens, current = [], []
        for ch in text:
            if unicodedata.category(ch).startswith("P") or ch.isspace():
                tokens.append("".join(current).lower())
                current = []
            else:
                current.append(ch)
        tokens.append("".join(current).lower())
        return [t for t in tokens if t]


def generate(args):
    lm = LanguageModel(args, checkpoint=args.checkpoint)
    lm.model.eval()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=XlsumTokenizer())
    os.makedirs(args.output, exist_ok=True)
    predictions = open(os.path.join(args.output, f"{args.split}_predictions.jsonl"), "w")
    results = {}
    for language in args.languages:
        ds = load_split(args.dataset, language, args.split)
        if args.max_samples:
            ds = ds.select(range(min(args.max_samples, len(ds))))
        src, tgt = COLUMNS[args.representation]
        prompts = [prompt_ids(lm, language, row[src], args.max_length - args.max_new_tokens) for row in ds]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outputs = [None] * len(prompts)
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            for j, tokens in zip(idx, beam_search(lm, [prompts[j] for j in idx], args.num_beams, args.max_new_tokens)):
                outputs[j] = lm.decode(tokens[:tokens.index(lm.eos)] if lm.eos in tokens else tokens)
        scores = [scorer.score(row[tgt], out) for row, out in zip(ds, outputs)]
        results[language] = {m: sum(s[m].fmeasure for s in scores) / len(scores) for m in ("rouge1", "rouge2", "rougeL")}
        results[language]["num_examples"] = len(scores)
        for row, out in zip(ds, outputs):
            predictions.write(json.dumps(dict(language=language, reference=row[tgt], prediction=out), ensure_ascii=False) + "\n")
        print(f"{language}: " + " ".join(f"{m} {v:.4f}" for m, v in results[language].items() if m != "num_examples"), flush=True)
    results["average"] = {m: sum(results[l][m] for l in args.languages) / len(args.languages) for m in ("rouge1", "rouge2", "rougeL")}
    print("average: " + " ".join(f"{m} {v:.4f}" for m, v in results["average"].items()))
    json.dump(dict(args=vars(args), results=results), open(os.path.join(args.output, f"{args.split}_results.json"), "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("train", "generate"):
        s = sub.add_parser(name)
        s.add_argument("--dataset", required=True, help="Hub dataset with one config per language, or a directory with one transcribe.py output per language")
        s.add_argument("--languages", nargs="+", required=True)
        s.add_argument("--representation", choices=COLUMNS, required=True)
        s.add_argument("--max-length", type=int, required=True, help="context length in tokens")
        s.add_argument("--output", required=True)
    t, g = sub.choices["train"], sub.choices["generate"]
    t.add_argument("--checkpoint", required=True, help="pretrained GPT checkpoint")
    t.add_argument("--tokenizer", required=True, help="tokenizer.json of the checkpoint")
    t.add_argument("--max-target-tokens", type=int, required=True, help="summaries are truncated to this many tokens for training")
    t.add_argument("--lr", type=float, required=True)
    t.add_argument("--batch-size", type=int, required=True)
    t.add_argument("--epochs", type=int, required=True)
    t.add_argument("--warmup-ratio", type=float, required=True, help="fraction of steps with linear warmup; linear decay after")
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--micro-batch-size", type=int, default=4)
    t.add_argument("--eval-every", type=int, default=0, help="optimizer steps between validations (0 = each epoch)")
    t.add_argument("--train-samples", type=int, default=0, help="training examples per language (0 = all)")
    t.add_argument("--val-samples", type=int, default=0, help="validation examples per language (0 = all)")
    g.add_argument("--checkpoint", required=True, help="best.pt written by train")
    g.add_argument("--split", default="test")
    g.add_argument("--num-beams", type=int, required=True)
    g.add_argument("--max-new-tokens", type=int, required=True)
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--max-samples", type=int, default=0, help="examples per language (0 = all)")
    args = p.parse_args()
    train(args) if args.mode == "train" else generate(args)


if __name__ == "__main__":
    main()

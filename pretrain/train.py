"""Pretrain a GPT on .bin token shards with AdamW (embeddings, head, scalars) and Muon (hidden
matrices), a constant-then-linear-cooldown learning rate and a growing sliding window.

Launch with torchrun, one process per GPU. The number of tokens per optimizer step is fixed
by --tokens-per-step and split over GPUs and gradient-accumulation micro-steps.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import BLOCK_SIZE, GPT, MAX_WINDOW, SIZES, next_multiple  # noqa: E402
from muon import Muon  # noqa: E402

ADAM_LR = dict(head=1 / 320, embed=0.3, scalars=0.015)
ADAM_EPS = 1e-10
MUON_WEIGHT_DECAY = 0.01
MOMENTUM_WARMUP_STEPS = 300
MOMENTUM_WARMUP_START = 0.85
LOG_EVERY = 50


def read_header(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 2 and header[3] == 32, f"unexpected .bin header in {path}"
    return int(header[2])


def load_shard(path):
    tokens = torch.from_numpy(np.fromfile(path, dtype=np.uint32, offset=1024, count=read_header(path)))
    return tokens.pin_memory()


class StreamLoader:
    """Serves consecutive windows of the concatenated shards; each rank reads its own slice."""

    def __init__(self, pattern, tokens_per_batch, rank, world_size):
        self.files = sorted(glob.glob(pattern))
        assert self.files, f"no files match {pattern}"
        self.batch, self.local = tokens_per_batch, tokens_per_batch // world_size
        self.rank, self.file_index, self.pos, self.tokens = rank, 0, 0, None

    def state_dict(self):
        return dict(file_index=self.file_index, pos=self.pos)

    def load_state_dict(self, state):
        self.file_index, self.pos, self.tokens = state["file_index"], state["pos"], None

    def next(self):
        if self.tokens is None:
            self.tokens = load_shard(self.files[self.file_index % len(self.files)])
        if self.pos + self.batch + 1 > len(self.tokens):
            self.file_index, self.pos = self.file_index + 1, 0
            self.tokens = load_shard(self.files[self.file_index % len(self.files)])
        buf = self.tokens[self.pos + self.rank * self.local:][:self.local + 1]
        self.pos += self.batch
        return (buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True),
                buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True))


def window_tokens(step, num_iterations):
    x = step / num_iterations
    return next_multiple(MAX_WINDOW * (4 * x ** 3 - 6 * x ** 2 + 3 * x), BLOCK_SIZE)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", choices=SIZES, required=True)
    p.add_argument("--train-files", required=True, help="glob, e.g. bins/text/data_train_*.bin")
    p.add_argument("--val-files", required=True)
    p.add_argument("--tokenizer", required=True, help="tokenizer.json (vocabulary size and <|eos|> id)")
    length = p.add_mutually_exclusive_group(required=True)
    length.add_argument("--epochs", type=float, help="passes over the training tokens")
    length.add_argument("--num-iterations", type=int)
    p.add_argument("--tokens-per-step", type=int, required=True)
    p.add_argument("--seq-len", type=int, required=True, help="tokens per GPU per micro-step")
    p.add_argument("--muon-lr", type=float, required=True)
    p.add_argument("--muon-momentum", type=float, required=True)
    p.add_argument("--adam-betas", type=float, nargs=2, required=True)
    p.add_argument("--cooldown-frac", type=float, required=True, help="final fraction of steps with linear decay")
    p.add_argument("--val-every", type=int, required=True)
    p.add_argument("--val-tokens", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint-every", type=int, default=0, help="write last.pt (resumable) every N steps")
    p.add_argument("--keep-every", type=int, default=0, help="also keep a resumable step_*.pt every N steps")
    p.add_argument("--resume", default=None, help="last.pt or step_*.pt to continue from")
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    master = rank == 0
    os.makedirs(args.output, exist_ok=True)

    def log(msg):
        if master:
            print(msg, flush=True)
            with open(os.path.join(args.output, "log.txt"), "a") as f:
                print(msg, file=f)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size, eos = tokenizer.get_vocab_size(), tokenizer.token_to_id("<|eos|>")
    meta_path = os.path.join(os.path.dirname(args.train_files), "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        assert meta["vocab_size"] == vocab_size and meta["eos"] == eos, "tokenizer does not match the bins"
    train_tokens = sum(read_header(f) for f in glob.glob(args.train_files))
    num_iterations = args.num_iterations or math.ceil(train_tokens * args.epochs / args.tokens_per_step)
    assert args.tokens_per_step % (world_size * args.seq_len) == 0, "tokens-per-step must be a multiple of GPUs x seq-len"
    accum = args.tokens_per_step // (world_size * args.seq_len)
    val_seq_len = 4 * args.seq_len
    assert args.val_tokens % (world_size * val_seq_len) == 0, "val-tokens must be a multiple of GPUs x 4 x seq-len"
    config = dict(size=args.size, vocab_size=vocab_size, eos=eos, num_layers=SIZES[args.size][0],
                  num_heads=SIZES[args.size][1], model_dim=SIZES[args.size][2])
    log(json.dumps(dict(vars(args), num_iterations=num_iterations, train_tokens=train_tokens,
                        grad_accum=accum, world_size=world_size, **config)))

    model = GPT(vocab_size, *SIZES[args.size], max_seq_len=val_seq_len, eos=eos).cuda()
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)
    hidden = sorted((q for q in model.blocks.parameters() if q.ndim >= 2), key=lambda q: q.size(), reverse=True)
    adam = torch.optim.AdamW([dict(params=[model.lm_head_w], lr=ADAM_LR["head"]),
                              dict(params=[*model.embed.parameters(), *model.value_embeds.parameters()], lr=ADAM_LR["embed"]),
                              dict(params=[model.scalars], lr=ADAM_LR["scalars"])],
                             betas=tuple(args.adam_betas), eps=ADAM_EPS, weight_decay=0.0, fused=True)
    muon = Muon(hidden, lr=args.muon_lr, momentum=args.muon_momentum, weight_decay=MUON_WEIGHT_DECAY,
                rank=rank, world_size=world_size)
    optimizers = [adam, muon]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    loader = StreamLoader(args.train_files, world_size * args.seq_len, rank, world_size)

    step, best_val = 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        for opt, s in zip(optimizers, state["optimizers"]):
            opt.load_state_dict(s)
        loader.load_state_dict(state["loader"])
        step, best_val = state["step"], state["best_val"]
        log(f"resumed from {args.resume} at step {step}")
    compiled = model if args.no_compile else torch.compile(model, dynamic=False)

    def lr_scale(step):
        x = step / num_iterations
        return 1.0 if x < 1 - args.cooldown_frac else (1 - x) / args.cooldown_frac

    def window_blocks(step):
        return torch.tensor(window_tokens(step, num_iterations) // BLOCK_SIZE, dtype=torch.int32, device="cuda")

    def snapshot(step, val_loss):
        return dict(model=model.state_dict(), config=dict(config, step=step, val_loss=val_loss,
                                                          window=window_tokens(step, num_iterations)))

    def save_state(step, name):
        if master:
            torch.save(dict(snapshot(step, best_val), optimizers=[o.state_dict() for o in optimizers],
                            loader=loader.state_dict(), step=step, best_val=best_val),
                       os.path.join(args.output, name))

    train_loss, t0 = torch.zeros((), device="cuda"), time.perf_counter()
    while True:
        if step % args.val_every == 0 or step == num_iterations:
            compiled.eval()
            val_loader = StreamLoader(args.val_files, world_size * val_seq_len, rank, world_size)
            val_loss = torch.zeros((), device="cuda")
            with torch.no_grad():
                for _ in range(args.val_tokens // (world_size * val_seq_len)):
                    val_loss += compiled(*val_loader.next(), window_blocks(step))
            val_loss /= args.val_tokens // (world_size * val_seq_len)
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            if val_loss.item() < best_val:
                best_val = val_loss.item()
                if master:
                    torch.save(snapshot(step, best_val), os.path.join(args.output, "best.pt"))
            log(f"step:{step}/{num_iterations} val_loss:{val_loss.item():.4f} best:{best_val:.4f}")
            compiled.train()
        if args.keep_every and step % args.keep_every == 0 and step > 0:
            save_state(step, f"step_{step:07d}.pt")
        if args.checkpoint_every and step % args.checkpoint_every == 0 and step > 0:
            save_state(step, "last.pt")
        if step == num_iterations:
            break

        for _ in range(accum):
            loss = compiled(*loader.next(), window_blocks(step)) / accum
            loss.backward()
            train_loss += loss.detach()
        futures = [dist.all_reduce(q.grad, op=dist.ReduceOp.AVG, async_op=True).get_future() for q in model.parameters()]
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lr_scale(step)
        frac = min(step / MOMENTUM_WARMUP_STEPS, 1)
        for group in muon.param_groups:
            group["momentum"] = (1 - frac) * MOMENTUM_WARMUP_START + frac * args.muon_momentum
        torch.futures.collect_all(futures).wait()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        step += 1
        if step % LOG_EVERY == 0:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
            log(f"step:{step}/{num_iterations} train_loss:{train_loss.item() / LOG_EVERY:.4f} "
                f"lr:{lr_scale(step):.3f} window:{window_tokens(step, num_iterations)} "
                f"{1000 * (time.perf_counter() - t0) / LOG_EVERY:.0f}ms/step")
            train_loss.zero_()
            t0 = time.perf_counter()

    save_state(step, "last.pt")
    log(f"done: best val_loss {best_val:.4f}; peak memory {torch.cuda.max_memory_allocated() >> 20} MiB")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

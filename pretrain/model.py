"""GPT with value embeddings, U-net skip weights and sliding-window causal attention
(modded-nanogpt). Training runs on one long token stream with FlexAttention block masks;
evaluation and fine-tuning use padded batches with dense masks."""
import json
import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

flex_attention = torch.compile(flex_attention, dynamic=False)  # the fused kernel needs compilation

SIZES = {"small": (12, 6, 768), "medium": (16, 8, 1024), "large": (20, 10, 1280)}  # layers, heads, dim
HEAD_DIM = 128
BLOCK_SIZE = 128                      # block-sparse attention mask granularity
MAX_WINDOW = 3456                     # sliding window reached at the end of training
ATTN_SCALE = 0.12
WINDOW_PATTERN = "LSSSLSSSSSSLSSSL"   # long / short window per layer, tiled over the layers
SKIP_MAP = {9: 6, 10: 4, 11: 2}       # layer -> earlier layer whose output is added (U-net)
EVAL_CHUNK = 16384                    # tokens per logits chunk when computing loss without grad


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


def softcap(logits: Tensor) -> Tensor:
    return 15 * logits * torch.rsqrt(logits.square() + 225)


def next_multiple(v: float, n: int) -> int:
    return max(n, n * ((int(v) + n - 1) // n))


def init_linear(w: Tensor) -> Tensor:
    bound = (3 ** 0.5) * 0.5 * w.size(-1) ** -0.5
    return w.uniform_(-bound, bound)


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        theta = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:  # x: (B, T, H, D)
        T = x.size(1)
        assert self.cos.size(0) >= offset + T, "sequence longer than max_seq_len"
        cos, sin = self.cos[None, offset:offset + T, None, :], self.sin[None, offset:offset + T, None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkvo_w = nn.Parameter(init_linear(torch.empty(4, num_heads * HEAD_DIM, dim)).bfloat16())
        self.qkvo_w.detach()[3].zero_()
        self.rotary = Rotary(HEAD_DIM, max_seq_len)

    def forward(self, x: Tensor, ve: Tensor | None, mask, lambdas: Tensor, cache: dict | None = None, offset: int = 0) -> Tensor:
        B, T = x.size(0), x.size(1)
        q, k, v = F.linear(x, self.qkvo_w[:3].flatten(end_dim=1)).view(B, T, 3 * self.num_heads, HEAD_DIM).chunk(3, dim=-2)
        q, k, v = self.rotary(norm(q), offset), self.rotary(norm(k), offset), norm(v)
        v = lambdas[0] * v + lambdas[1] * ve.view_as(v) if ve is not None else lambdas[0] * v
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if cache is not None:  # incremental decoding: keys/values of earlier positions
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        if isinstance(mask, BlockMask):
            y = flex_attention(q, k, v, block_mask=mask, scale=ATTN_SCALE)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=ATTN_SCALE)
        return F.linear(y.transpose(1, 2).contiguous().view(B, T, -1), self.qkvo_w[3])


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc_w = nn.Parameter(init_linear(torch.empty(4 * dim, dim)).bfloat16())
        self.proj_w = nn.Parameter(torch.zeros(dim, 4 * dim).bfloat16())
        self.fc_w.wd_mul = self.proj_w.wd_mul = 2.0

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.relu(F.linear(x, self.fc_w)).square(), self.proj_w)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(self, x, ve, x0, mask, lambdas, sa_lambdas, cache=None, offset=0):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(x, ve, mask, sa_lambdas, cache, offset)
        return x + self.mlp(norm(x))


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int, eos: int):
        super().__init__()
        assert num_layers % 2 == 0 and num_layers >= 12
        self.vocab_size, self.eos = vocab_size, eos
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        self.lm_head_w = nn.Parameter(torch.zeros(next_multiple(vocab_size, 128), model_dim))
        self.scalars = nn.Parameter(torch.cat([torch.ones(num_layers),                       # skip weights
                                               torch.tensor([1.0, 0.0]).repeat(num_layers),  # block lambdas
                                               torch.tensor([0.5, 0.5]).repeat(num_layers)]))  # value-embedding lambdas
        self.window_pattern = (WINDOW_PATTERN * ((num_layers + 15) // 16))[:num_layers]
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                m.bfloat16()

    def _trunk(self, input_ids: Tensor, long_mask, short_mask, cache=None, offset=0) -> Tensor:
        n = len(self.blocks)
        ve = [e(input_ids) for e in self.value_embeds]
        ve = ve + [None] * (n - 6) + ve
        skip_weights = self.scalars[:n]
        lambdas = self.scalars[n:3 * n].view(-1, 2)
        sa_lambdas = self.scalars[3 * n:5 * n].view(-1, 2)
        x = x0 = norm(self.embed(input_ids))
        skips = []
        for i, block in enumerate(self.blocks):
            if i in SKIP_MAP:
                x = x + skip_weights[SKIP_MAP[i]] * skips[SKIP_MAP[i]]
            mask = long_mask if self.window_pattern[i] == "L" else short_mask
            x = block(x, ve[i], x0, mask, lambdas[i], sa_lambdas[i], cache[i] if cache else None, offset)
            skips.append(x)
        return norm(x)

    # ---- training / validation on one token stream --------------------------------------
    def create_blockmasks(self, input_seq: Tensor, window_blocks: Tensor):
        """Causal, same-document, sliding-window block masks (long window and half window)."""
        assert self.eos is not None, "document separator id unknown: pass eos to load_checkpoint"
        docs = (input_seq == self.eos).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (docs[q_idx] == docs[kv_idx])

        def dense_to_ordered(dense: Tensor):
            num_blocks = dense.sum(dim=-1, dtype=torch.int32)
            indices = dense.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        assert len(input_seq) % BLOCK_SIZE == 0
        block_idx = torch.arange(len(input_seq) // BLOCK_SIZE, dtype=torch.int32, device=input_seq.device)
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        any_ = (block_idx[:, None] >= block_idx) & (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        all_ = (block_idx[:, None] > block_idx) & (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        partial_num, partial_idx = dense_to_ordered(any_ & ~all_)
        full_num, full_idx = dense_to_ordered(all_)

        def build(window: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_num, torch.clamp_min(window - full_num, 1)), partial_idx,
                torch.clamp_max(full_num, window - 1), full_idx, BLOCK_SIZE=BLOCK_SIZE, mask_mod=document_causal)

        return build(window_blocks), build(window_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, window_blocks: Tensor) -> Tensor:
        """Mean next-token cross-entropy on a 1-D token stream."""
        long_mask, short_mask = self.create_blockmasks(input_seq, window_blocks)
        x = self._trunk(input_seq[None], long_mask, short_mask).flatten(end_dim=1)
        if torch.is_grad_enabled():
            logits = F.linear(x, self.lm_head_w.bfloat16()).float()
            return F.cross_entropy(softcap(logits), target_seq)
        loss = 0.0
        for xs, ts in zip(x.split(EVAL_CHUNK), target_seq.split(EVAL_CHUNK)):
            logits = F.linear(xs, self.lm_head_w.bfloat16()).float()
            loss = loss + F.cross_entropy(softcap(logits), ts, reduction="sum")
        return loss / len(target_seq)

    # ---- batched inference / fine-tuning ---------------------------------------------------
    def hidden(self, input_ids: Tensor, attention_mask: Tensor, window: int, cache: list | None = None) -> Tensor:
        """Final hidden states of input_ids for padded batches. attention_mask (1 = real token)
        covers the cached positions followed by input_ids; cache (one dict per layer) is
        filled on the first call and extended on later calls for incremental decoding."""
        offset = cache[0]["k"].size(2) if cache and cache[0] else 0
        T = attention_mask.size(1)
        assert T == offset + input_ids.size(1)
        q_pos = torch.arange(offset, T, device=input_ids.device)[:, None]
        k_pos = torch.arange(T, device=input_ids.device)[None, :]
        keys = attention_mask.bool()[:, None, None, :] | (q_pos == k_pos)[None, None]

        def dense(w):
            return ((q_pos >= k_pos) & (q_pos - k_pos < w))[None, None] & keys

        short = window // BLOCK_SIZE // 2 * BLOCK_SIZE
        return self._trunk(input_ids, dense(window), dense(short), cache, offset)

    def head(self, x: Tensor) -> Tensor:
        """Softcapped logits over the padded vocabulary; softmax gives the model's distribution."""
        return softcap(F.linear(x, self.lm_head_w.bfloat16()).float())

    def logits(self, input_ids: Tensor, attention_mask: Tensor, window: int) -> Tensor:
        return self.head(self.hidden(input_ids, attention_mask, window))


def load_checkpoint(path: str, device="cpu", max_seq_len: int = 4096, eos: int | None = None):
    """Load a checkpoint written by train.py or released on the Hub. The architecture is read
    from the weights; the sliding window comes from the stored config (or config.json alongside
    for Hub checkpoints), defaulting to the end-of-training window. `eos` overrides the stored
    document-separator id (needed only for loss computation on token streams)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    config = dict(ckpt.get("config") or {})
    hub_config = os.path.join(os.path.dirname(os.path.abspath(path)), "config.json")
    if "window" not in config and os.path.exists(hub_config):
        hub = json.load(open(hub_config))
        config.update(eos=hub["eot_token"], window=hub["long_window"])
    config.setdefault("window", MAX_WINDOW)
    if eos is not None:
        config["eos"] = eos
    config.update(num_layers=1 + max(int(k.split(".")[1]) for k in state if k.startswith("blocks.")),
                  num_heads=state["blocks.0.attn.qkvo_w"].size(1) // HEAD_DIM)
    config["vocab_size"], config["model_dim"] = state["embed.weight"].shape
    model = GPT(config["vocab_size"], config["num_layers"], config["num_heads"], config["model_dim"],
                max_seq_len, config.get("eos"))
    model.load_state_dict(state)
    return model.to(device).eval(), config

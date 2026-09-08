"""Muon optimizer (Jordan et al.), distributed as in modded-nanogpt: each rank updates a strided
subset of the parameters and the results are all-gathered."""
import torch
import torch.distributed as dist
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    """Quintic Newton-Schulz iteration approximating the orthogonalization of G, run in bf16."""
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for a, b, c in [(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037),
                    (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012)]:
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def update(acc_bf16_view_u16: Tensor, mantissa: Tensor, momentum_buffer: Tensor, grad: Tensor,
           momentum: Tensor, eff_lr: Tensor, eff_weight_decay: Tensor):
    """Momentum + orthogonalized update applied to a bf16 parameter whose low 16 mantissa bits
    are kept separately, giving fp32 accumulation without an fp32 master copy."""
    grad = grad.float()
    momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
    v = zeropower_via_newtonschulz5(momentum * momentum_buffer + (1 - momentum) * grad)
    acc = (acc_bf16_view_u16.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
    acc.view(torch.float32).mul_(1 - eff_weight_decay)
    acc.view(torch.float32).add_(other=v, alpha=-eff_lr)
    acc_bf16_view_u16.copy_((acc >> 16).to(torch.uint16))
    mantissa.copy_(acc.to(torch.uint16))


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, weight_decay, rank=0, world_size=1):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))
        self.rank, self.world_size = rank, world_size
        assert all(p.dtype == torch.bfloat16 for g in self.param_groups for p in g["params"])

    @torch.no_grad()
    def step(self):
        futures = []
        for group in self.param_groups:
            params = group["params"]
            padded = params + [torch.empty_like(params[-1])] * self.world_size
            momentum = torch.as_tensor(group["momentum"], dtype=torch.float64)
            for base in range(0, len(params), self.world_size):
                i = base + self.rank
                if i < len(params):
                    p, state = params[i], self.state[params[i]]
                    if not state:
                        state["mantissa"] = torch.zeros_like(p, dtype=torch.uint16)
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                    eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5
                    eff_wd = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                    update(p.view(torch.uint16), state["mantissa"], state["momentum_buffer"], p.grad,
                           momentum, torch.as_tensor(eff_lr, dtype=torch.float64),
                           torch.as_tensor(eff_wd, dtype=torch.float64))
                if self.world_size > 1:
                    futures.append(dist.all_gather(padded[base:base + self.world_size], padded[i],
                                                   async_op=True).get_future())
        if futures:
            torch.futures.collect_all(futures).wait()

    def load_state_dict(self, state_dict):
        # The base implementation casts state to the parameter dtype (bf16), which would truncate
        # the fp32 momentum buffers; restore them as saved instead.
        for group, saved in zip(self.param_groups, state_dict["param_groups"]):
            group.update({k: v for k, v in saved.items() if k != "params"})
            for p, idx in zip(group["params"], saved["params"]):
                if idx in state_dict["state"]:
                    self.state[p] = {k: v.to(p.device) for k, v in state_dict["state"][idx].items()}

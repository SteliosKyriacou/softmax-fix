"""Grokking experiment (paper Sec. 6): a+b mod 31, 50% split, 1-layer 4-head
width-64 ReLU transformer without LayerNorm, full-batch AdamW
(lr 1e-3, betas (0.9, 0.98), weight decay 1), 10k steps.

Varies: loss (standard / fixed), precision of the logits fed to the loss
(bf16 / fp32 / fp64), and AdamW eps.
"""
import argparse, json, math, os, time
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from softmax_fix import standard_ce, fixed_ce, margin

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, S, D = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, S, D))
        return x + self.mlp(x)


class Transformer(nn.Module):
    def __init__(self, p, d=64, h=4, layers=1):
        super().__init__()
        self.tok = nn.Embedding(p + 1, d)       # p numbers + '=' token
        self.pos = nn.Parameter(torch.randn(3, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        self.unembed = nn.Linear(d, p, bias=False)

    def forward(self, x):
        h = self.tok(x) + self.pos
        for b in self.blocks:
            h = b(h)
        return self.unembed(h[:, -1])


def run(loss_name, prec, eps, seed, p=31, steps=10_000, lr=1e-3, wd=1.0, frac=0.5, log_every=25, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    X = torch.stack([a.flatten(), b.flatten(), torch.full((p * p,), p)], 1)
    Y = (a.flatten() + b.flatten()) % p
    perm = torch.randperm(p * p, generator=g)
    n = int(frac * p * p)
    Xtr, Ytr, Xte, Yte = [t.to(device) for t in (X[perm[:n]], Y[perm[:n]], X[perm[n:]], Y[perm[n:]])]

    torch.manual_seed(seed)
    model = Transformer(p).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=eps, weight_decay=wd)
    loss_fn = {"standard": standard_ce, "fixed": fixed_ce}[loss_name]
    dt = DTYPES[prec]

    hist = {k: [] for k in ["step", "loss", "train_acc", "test_acc", "margin", "margin_min"]}
    for step in range(steps + 1):
        logits = model(Xtr)
        z = logits.to(dt)
        loss = loss_fn(z, Ytr)
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                m = margin(logits, Ytr)
                te = model(Xte).argmax(-1)
                hist["step"].append(step)
                hist["loss"].append(F.cross_entropy(logits.double(), Ytr).item())
                hist["train_acc"].append((logits.argmax(-1) == Ytr).float().mean().item())
                hist["test_acc"].append((te == Yte).float().mean().item())
                hist["margin"].append(m.mean().item())
                hist["margin_min"].append(m.min().item())
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="fixed")
    ap.add_argument("--prec", default="fp32")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--out", default="results/grok")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    h = run(a.loss, a.prec, a.eps, a.seed, steps=a.steps)
    fn = f"{a.out}/{a.loss}_{a.prec}_eps{a.eps:g}_s{a.seed}.json"
    json.dump(dict(vars(a), hist=h), open(fn, "w"))
    print(f"{fn}: test_acc={h['test_acc'][-1]:.3f} margin={h['margin'][-1]:.2f} ({time.time()-t0:.0f}s)")

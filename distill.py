"""Self-distillation experiment (paper Sec. 7): the student starts as an exact
copy of the teacher, so the ideal gradient is exactly zero and the student
should never move. Any drift is pure numerical error (Softmax Cancellation).

a+b mod 53, 2-layer ReLU MLP, 50% split, full-batch Adam (lr 2e-3,
betas (0.9, 0.98), eps 1e-8, no weight decay), 2000 steps, temperature T=4.
Forward pass + loss run in bf16 / fp32 / fp64; master weights are fp32.
"""
import argparse, copy, json, os, time
import torch
torch.set_num_threads(1)
import torch.nn as nn
from torch.func import functional_call
from softmax_fix import standard_soft_ce, fixed_soft_ce

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}
P = 53


class MLP(nn.Module):
    def __init__(self, p=P, d=128, h=512):
        super().__init__()
        self.emb = nn.Embedding(p, d)
        self.net = nn.Sequential(nn.Linear(2 * d, h), nn.ReLU(), nn.Linear(h, p))

    def forward(self, x):
        return self.net(self.emb(x).flatten(1))


def data(seed):
    a, b = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij")
    X = torch.stack([a.flatten(), b.flatten()], 1)
    Y = (a.flatten() + b.flatten()) % P
    perm = torch.randperm(P * P, generator=torch.Generator().manual_seed(seed))
    n = P * P // 2
    return X, Y, perm[:n], perm[n:]


def train_teacher(seed, X, Y):
    """Fit the teacher on the full table until it is 100% accurate."""
    cache = f"results/distill/teacher_s{seed}.pt"
    torch.manual_seed(seed)
    m = MLP()
    if os.path.exists(cache):
        m.load_state_dict(torch.load(cache)); return m
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)
    for step in range(5000):
        loss = nn.functional.cross_entropy(m(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 and (m(X).argmax(-1) == Y).all():
            break
    assert (m(X).argmax(-1) == Y).all()
    torch.save(m.state_dict(), cache)
    return m


def logits_in(model, params, x, dt):
    return functional_call(model, {k: v.to(dt) for k, v in params.items()}, (x,))


def run(loss_name, prec, seed, steps=2000, T=4.0, log_every=10):
    X, Y, tr, te = data(seed)
    teacher = train_teacher(seed, X, Y)
    tparams = {k: v.detach().clone() for k, v in teacher.named_parameters()}
    student = copy.deepcopy(teacher)
    opt = torch.optim.Adam(student.parameters(), lr=2e-3, betas=(0.9, 0.98), eps=1e-8)
    loss_fn = {"standard": standard_soft_ce, "fixed": fixed_soft_ce}[loss_name]
    dt = DTYPES[prec]
    with torch.no_grad():
        zT_tr = logits_in(teacher, tparams, X[tr], dt)          # teacher in same precision
        pT_te = torch.softmax(logits_in(teacher, tparams, X[te], torch.float64) / T, -1)

    hist = {k: [] for k in ["step", "kl", "test_acc", "grad_max", "param_drift"]}
    for step in range(steps + 1):
        sparams = dict(student.named_parameters())
        zS = logits_in(student, sparams, X[tr], dt)
        loss = loss_fn(zS, zT_tr, T=T)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                lq = torch.log_softmax(logits_in(student, sparams, X[te], torch.float64) / T, -1)
                kl = (pT_te * (pT_te.log() - lq)).sum(-1).mean().item()
                acc = (student(X[te]).argmax(-1) == Y[te]).float().mean().item()
                gmax = max(p.grad.abs().max().item() for p in student.parameters())
                drift = max((sparams[k] - tparams[k]).abs().max().item() for k in tparams)
            for k, v in zip(hist, [step, kl, acc, gmax, drift]):
                hist[k].append(v)
        if step == steps:
            break
        opt.step()
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="fixed")
    ap.add_argument("--prec", default="fp32")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs("results/distill", exist_ok=True)
    t0 = time.time()
    h = run(a.loss, a.prec, a.seed)
    fn = f"results/distill/{a.loss}_{a.prec}_s{a.seed}.json"
    json.dump(dict(vars(a), hist=h), open(fn, "w"))
    print(f"{fn}: KL={h['kl'][-1]:.2e} acc={h['test_acc'][-1]:.3f} grad0={h['grad_max'][0]:.1e} drift={h['param_drift'][-1]:.1e} ({time.time()-t0:.0f}s)")

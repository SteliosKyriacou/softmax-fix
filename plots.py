"""Figures and tables.

Main grokking results come from results/grok_v2 (init="small", 20 seeds, 10k steps, as in the paper).
results/grok (default torch init, 5 seeds, 40k steps) is the first attempt, kept for the record.
"""
import json, glob, math, os
from math import comb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
C = {"standard": "#d1495b", "fixed": "#2e86ab"}
CLIFF = {"bf16": 5.5, "fp32": 16.6, "fp64": 36.7}
EPS = ["1e-08", "1e-12", "1e-16"]
PRECS = ["bf16", "fp32", "fp64"]
SEEDS = range(20)
D = "results/grok_v2"


def H(loss, prec, e, s):
    return json.load(open(f"{D}/{loss}_{prec}_eps{e}_small_s{s}.json"))["hist"]


def grok_step(h, th=0.95):
    return next((s for s, a in zip(h["step"], h["test_acc"]) if a > th), np.inf)


def tavg_margin(h, frac=0.1):
    """time-average over the last 10% of training (standard CE oscillates)"""
    n = max(1, int(len(h["margin"]) * frac))
    return float(np.mean(h["margin"][-n:]))


def sign_p(w, l):
    n, k = w + l, min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


# ---------- Fig 1: the slingshot --------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for loss in ["standard", "fixed"]:
    for i, s in enumerate([0, 1, 2]):
        h = H(loss, "fp32", "1e-08", s)
        kw = dict(color=C[loss], alpha=0.8, lw=1.1, label=f"{loss} CE" if i == 0 else None)
        ax[0].plot(h["step"], h["margin"], **kw)
        ax[1].plot(h["step"], h["test_acc"], **kw)
ax[0].axhline(CLIFF["fp32"], ls="--", c="k", lw=0.8)
ax[0].text(100, CLIFF["fp32"] - 1.4, "fp32 precision cliff ≈ 16.6", fontsize=8)
ax[0].axhline(math.log(1e8), ls=":", c="gray", lw=0.8)
ax[0].text(100, math.log(1e8) + 0.4, "AdamW equilibrium log(1/ε) ≈ 18.4", fontsize=8, color="gray")
ax[0].set_ylim(top=20.5)
ax[0].set_ylabel("mean train margin m"); ax[1].set_ylabel("test accuracy"); ax[1].set_xlabel("step")
ax[0].legend(loc="lower right"); ax[0].set_title("a+b mod 31, fp32 logits, AdamW ε=1e-8, seeds 0-2")
plt.tight_layout(); plt.savefig("figures/fig1_slingshot.png", dpi=150); plt.close()

# ---------- Fig 2 + table: equilibrium margins ---------------------------------
rows = []
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for j, prec in enumerate(PRECS):
    for k, loss in enumerate(["standard", "fixed"]):
        ms = [[tavg_margin(H(loss, prec, e, s)) for s in SEEDS] for e in EPS]
        pos = np.arange(3) + (k - 0.5) * 0.3
        axs[j].bar(pos, [np.mean(m) for m in ms], width=0.3, color=C[loss], yerr=[np.std(m) for m in ms], label=f"{loss} CE")
        for e, m in zip(EPS, ms):
            rows.append((prec, loss, e, np.mean(m), np.std(m)))
    for i, e in enumerate(EPS):
        axs[j].hlines(math.log(1 / float(e)), i - 0.35, i + 0.35, colors="k", lw=1.5)
    axs[j].axhline(CLIFF[prec], ls="--", c="gray", lw=0.8)
    axs[j].set_xticks(range(3)); axs[j].set_xticklabels(["ε=1e-8", "1e-12", "1e-16"])
    axs[j].set_title(f"{prec} logits (dashed: precision cliff)")
axs[0].set_ylabel("margin, mean of last 10% of training"); axs[0].legend(fontsize=8)
plt.suptitle("Black ticks: predicted AdamW equilibrium m* = log(1/ε). 20 seeds per bar.", fontsize=10)
plt.tight_layout(); plt.savefig("figures/fig2_equilibrium.png", dpi=150); plt.close()

with open("results/table_margin.md", "w") as f:
    f.write("| logits | loss | " + " | ".join(f"ε = {e}" for e in EPS) + " |\n|---|---|---|---|---|\n")
    for prec in PRECS:
        for loss in ["standard", "fixed"]:
            vals = [f"{m:.1f} ± {sd:.1f}" for p_, l_, e, m, sd in rows if p_ == prec and l_ == loss]
            f.write(f"| {prec} | {loss} | " + " | ".join(vals) + " |\n")
    f.write("| predicted log(1/ε) | | " + " | ".join(f"{math.log(1/float(e)):.1f}" for e in EPS) + " |\n")

# ---------- Fig 4 + table: paired grokking analysis ----------------------------
lines = ["| logits | ε | fix groks sooner / later / tie | p | median grok step std → fixed (both grokked) | grokked by 10k std / fixed | final acc fix higher / lower | p | median final acc std / fixed |",
         "|---|---|---|---|---|---|---|---|---|"]
fig, axs = plt.subplots(3, 3, figsize=(11, 10))
for i, prec in enumerate(PRECS):
    for j, e in enumerate(EPS):
        gs, gf, as_, af = [], [], [], []
        for s in SEEDS:
            hs, hf = H("standard", prec, e, s), H("fixed", prec, e, s)
            gs.append(grok_step(hs)); gf.append(grok_step(hf))
            as_.append(hs["test_acc"][-1] * 100); af.append(hf["test_acc"][-1] * 100)
        gs, gf, as_, af = map(np.array, (gs, gf, as_, af))
        sooner, later = int((gf < gs).sum()), int((gf > gs).sum())
        both = np.isfinite(gs) & np.isfinite(gf)
        hi, lo = int((af - as_ > 1).sum()), int((af - as_ < -1).sum())
        med = f"{np.median(gs[both]):.0f} → {np.median(gf[both]):.0f}" if both.any() else "–"
        lines.append(f"| {prec} | {e} | {sooner} / {later} / {20-sooner-later} | {sign_p(sooner, later):.3f} | {med} | "
                     f"{int(np.isfinite(gs).sum())} / {int(np.isfinite(gf).sum())} | {hi} / {lo} | {sign_p(hi, lo):.2f} | {np.median(as_):.1f} / {np.median(af):.1f} |")
        ax = axs[i, j]
        cap = 11000
        x, y = np.minimum(gs, cap), np.minimum(gf, cap)
        ax.scatter(x, y, s=22, c=["#2e86ab" if a < b else ("#d1495b" if a > b else "gray") for a, b in zip(gf, gs)])
        ax.plot([0, cap], [0, cap], c="k", lw=0.6)
        ax.axhline(10500, c="gray", lw=0.4, ls=":"); ax.axvline(10500, c="gray", lw=0.4, ls=":")
        ax.set_xlim(0, cap + 500); ax.set_ylim(0, cap + 500)
        ax.set_title(f"{prec}, ε={e}: fix sooner {sooner}, later {later}", fontsize=9)
        if i == 2: ax.set_xlabel("standard CE: step to 95% test acc")
        if j == 0: ax.set_ylabel("fixed CE: step to 95% test acc")
plt.suptitle("Paired by seed (same init + split). Below diagonal = fix groks sooner. Points at 11000 = not grokked by 10k.", fontsize=10)
plt.tight_layout(); plt.savefig("figures/fig4_paired_grok.png", dpi=150); plt.close()
open("results/table_grok_paired.md", "w").write("\n".join(lines) + "\n")

# ---------- Fig 3 + table: distillation -------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for j, prec in enumerate(PRECS):
    for loss in ["standard", "fixed"]:
        for i, f in enumerate(sorted(glob.glob(f"results/distill/{loss}_{prec}_s*.json"))):
            h = json.load(open(f))["hist"]
            axs[j].semilogy(h["step"], np.maximum(h["kl"], 1e-20), color=C[loss], alpha=0.8, label=f"{loss} CE" if i == 0 else None)
    axs[j].set_title(f"{prec} forward + loss"); axs[j].set_xlabel("step")
axs[0].set_ylabel("held-out KL(teacher || student)"); axs[0].legend()
plt.suptitle("Student initialised = teacher. Ideal: KL stays at 0 forever.", fontsize=10)
plt.tight_layout(); plt.savefig("figures/fig3_distill.png", dpi=150); plt.close()

with open("results/distill.md", "w") as f:
    f.write("| precision | loss | |grad| at step 0 | max weight drift | final held-out KL | test acc (%) |\n|---|---|---|---|---|---|\n")
    for prec in PRECS:
        for loss in ["standard", "fixed"]:
            rs = [json.load(open(x))["hist"] for x in sorted(glob.glob(f"results/distill/{loss}_{prec}_s*.json"))]
            f.write(f"| {prec} | {loss} | {np.mean([r['grad_max'][0] for r in rs]):.1e} | {np.mean([r['param_drift'][-1] for r in rs]):.1e} | "
                    f"{np.mean([r['kl'][-1] for r in rs]):.1e} | {100*np.mean([r['test_acc'][-1] for r in rs]):.1f} |\n")

for t in ["table_margin", "table_grok_paired", "distill"]:
    print(open(f"results/{t}.md").read())

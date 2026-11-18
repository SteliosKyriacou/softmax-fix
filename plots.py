import json, glob, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
C = {"standard": "#d1495b", "fixed": "#2e86ab"}
CLIFF = {"bf16": 5.5, "fp32": 16.6, "fp64": 36.7}


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


# ---------- Fig 1: the slingshot, fp32 eps=1e-8, margin + test acc -----------
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for loss in ["standard", "fixed"]:
    for i, r in enumerate(load(f"results/grok/{loss}_fp32_eps1e-08_s*.json")[:3]):
        h = r["hist"]
        kw = dict(color=C[loss], alpha=0.8, lw=1.1, label=f"{loss} CE" if i == 0 else None)
        ax[0].plot(h["step"], h["margin"], **kw)
        ax[1].plot(h["step"], h["test_acc"], **kw)
ax[0].axhline(CLIFF["fp32"], ls="--", c="k", lw=0.8); ax[0].text(200, CLIFF["fp32"] + 0.4, "fp32 precision cliff c32 ≈ 16.6", fontsize=8)
ax[0].axhline(math.log(1e8), ls=":", c="gray", lw=0.8); ax[0].text(200, math.log(1e8) + 0.4, "AdamW equilibrium log(1/eps) ≈ 18.4", fontsize=8, color="gray")
ax[0].set_ylabel("mean train margin m"); ax[1].set_ylabel("test accuracy"); ax[1].set_xlabel("step")
ax[0].legend(loc="lower right"); ax[0].set_title("a+b mod 31, fp32 logits, AdamW eps=1e-8 (3 seeds)")
plt.tight_layout(); plt.savefig("figures/fig1_slingshot.png", dpi=150); plt.close()

# ---------- Fig 2: final margin vs predicted equilibrium, all precisions/eps ---
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
rows = []
for j, prec in enumerate(["bf16", "fp32", "fp64"]):
    ax = axs[j]
    for k, loss in enumerate(["standard", "fixed"]):
        xs, ms, accs = [], [], []
        for e in ["1e-08", "1e-12", "1e-16"]:
            rs = load(f"results/grok/{loss}_{prec}_eps{e}_s*.json")
            if not rs: continue
            # average margin over last 10% of training (standard CE oscillates)
            fm = [np.mean(r["hist"]["margin"][-160:]) for r in rs]
            fa = [r["hist"]["test_acc"][-1] for r in rs]
            ms.append(fm); accs.append(fa); xs.append(e)
            rows.append((prec, loss, e, np.mean(fm), np.std(fm), 100 * np.mean(fa), 100 * np.std(fa), len(rs)))
        pos = np.arange(len(xs)) + (k - 0.5) * 0.3
        ax.bar(pos, [np.mean(m) for m in ms], width=0.3, color=C[loss], yerr=[np.std(m) for m in ms], label=f"{loss} CE")
    for i, e in enumerate(["1e-08", "1e-12", "1e-16"]):
        ax.hlines(math.log(1 / float(e)), i - 0.35, i + 0.35, colors="k", lw=1.5)
    ax.axhline(CLIFF[prec], ls="--", c="gray", lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels(["eps=1e-8", "1e-12", "1e-16"]); ax.set_title(f"{prec} logits (dashed: precision cliff)")
axs[0].set_ylabel("final mean margin"); axs[0].legend(fontsize=8)
plt.suptitle("Black ticks: predicted AdamW equilibrium m* = log(1/eps)", fontsize=10)
plt.tight_layout(); plt.savefig("figures/fig2_equilibrium.png", dpi=150); plt.close()

with open("results/table1.md", "w") as f:
    f.write("| precision | loss | AdamW eps | final margin | predicted log(1/eps) | test acc (%) | seeds |\n|---|---|---|---|---|---|---|\n")
    for prec, loss, e, m, ms, a, as_, n in rows:
        f.write(f"| {prec} | {loss} | {e} | {m:.1f} ± {ms:.1f} | {math.log(1/float(e)):.1f} | {a:.1f} ± {as_:.1f} | {n} |\n")
print(open("results/table1.md").read())

# ---------- Fig 3: distillation drift -------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for j, prec in enumerate(["bf16", "fp32", "fp64"]):
    for loss in ["standard", "fixed"]:
        for i, r in enumerate(load(f"results/distill/{loss}_{prec}_s*.json")):
            h = r["hist"]
            axs[j].semilogy(h["step"], np.maximum(h["kl"], 1e-20), color=C[loss], alpha=0.8, label=f"{loss} CE" if i == 0 else None)
    axs[j].set_title(f"{prec} forward + loss"); axs[j].set_xlabel("step")
axs[0].set_ylabel("held-out KL(teacher || student)"); axs[0].legend()
plt.suptitle("Student initialised = teacher. Ideal: KL stays at 0 forever.", fontsize=10)
plt.tight_layout(); plt.savefig("figures/fig3_distill.png", dpi=150); plt.close()

with open("results/distill.md", "w") as f:
    f.write("| precision | loss | |grad| at step 0 | max weight drift | final held-out KL | test acc (%) |\n|---|---|---|---|---|---|\n")
    for prec in ["bf16", "fp32", "fp64"]:
        for loss in ["standard", "fixed"]:
            rs = load(f"results/distill/{loss}_{prec}_s*.json")
            if not rs: continue
            g = np.mean([r["hist"]["grad_max"][0] for r in rs]); d = np.mean([r["hist"]["param_drift"][-1] for r in rs])
            kl = np.mean([r["hist"]["kl"][-1] for r in rs]); a = 100 * np.mean([r["hist"]["test_acc"][-1] for r in rs])
            f.write(f"| {prec} | {loss} | {g:.1e} | {d:.1e} | {kl:.1e} | {a:.1f} |\n")
print(open("results/distill.md").read())

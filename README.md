# Recreating "A Fix for Softmax and Cross-Entropy" (arXiv:2610.31642)

PyTorch reproduction of Barron, Gnanapragasam & Tschannen (2026). Write-up: [blog_post.md](blog_post.md).

| file | what |
|---|---|
| `softmax_fix.py` | standard + fixed log-softmax, hard CE, soft (distillation) CE; `python softmax_fix.py` checks the paper's worked examples |
| `grok.py` | a+b mod 31 grokking, sweep over loss × logit precision × AdamW eps |
| `distill.py` | self-distillation (student = teacher) on a+b mod 53 |
| `plots.py` | figures in `figures/`, tables in `results/table_margin.md`, `results/table_grok_paired.md`, `results/distill.md` |

Reproduce:
```bash
python softmax_fix.py
cat results/grid_v2.txt | xargs -P 28 -L 1 python grok.py   # main grid: 360 runs (20 seeds, init=small, 10k steps), ~35 min on 32 cores
for s in 0 1 2; do for p in bf16 fp32 fp64; do for l in standard fixed; do python distill.py --loss $l --prec $p --seed $s & done; done; done; wait
python plots.py
```

Main results use `--init small` (embeddings at 1/sqrt(d)), which groks on the paper's 10k-step timescale.
`results/grok/` (default torch init, 5 seeds, 40k steps), `results/kick/` (artificial-kick test) and `results/initprobe/` are earlier runs kept for the record.

Deviations from the paper: CPU / PyTorch rather than JAX, own transformer details and distillation teacher where the paper is unspecified, margin = mean over training examples averaged over the last 10% of training.

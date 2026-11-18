# Recreating "A Fix for Softmax and Cross-Entropy" (arXiv:2610.31642)

PyTorch reproduction of Barron, Gnanapragasam & Tschannen (2026). Write-up: [blog_post.md](blog_post.md).

| file | what |
|---|---|
| `softmax_fix.py` | standard + fixed log-softmax, hard CE, soft (distillation) CE; `python softmax_fix.py` checks the paper's worked examples |
| `grok.py` | a+b mod 31 grokking, sweep over loss × logit precision × AdamW eps |
| `distill.py` | self-distillation (student = teacher) on a+b mod 53 |
| `plots.py` | figures in `figures/`, tables in `results/table1.md`, `results/distill.md` |

Reproduce:
```bash
python softmax_fix.py
cat results/grid.txt | xargs -P 28 -L 1 python grok.py      # 90 runs, ~40 min on 32 CPU cores
for s in 0 1 2; do for p in bf16 fp32 fp64; do for l in standard fixed; do python distill.py --loss $l --prec $p --seed $s & done; done; done; wait
python plots.py
```

Deviations from the paper: 40k steps instead of 10k (my transformer groks slower), CPU, own transformer and teacher details where the paper is unspecified, margin reported as mean over training examples.

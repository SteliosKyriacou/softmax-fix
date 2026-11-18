| precision | loss | |grad| at step 0 | max weight drift | final held-out KL | test acc (%) |
|---|---|---|---|---|---|
| bf16 | standard | 8.1e-06 | 6.5e-01 | 5.4e-03 | 100.0 |
| bf16 | fixed | 0.0e+00 | 0.0e+00 | 2.1e-18 | 100.0 |
| fp32 | standard | 1.2e-10 | 1.5e-01 | 8.8e-05 | 100.0 |
| fp32 | fixed | 0.0e+00 | 0.0e+00 | 2.1e-18 | 100.0 |
| fp64 | standard | 2.6e-19 | 0.0e+00 | 2.1e-18 | 100.0 |
| fp64 | fixed | 0.0e+00 | 0.0e+00 | 2.1e-18 | 100.0 |

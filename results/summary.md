# GraphRAG vs vector-only baseline

## Overall

| system   |   accuracy |   bleu |   rouge_l |   latency_s |   context_tokens |
|:---------|-----------:|-------:|----------:|------------:|-----------------:|
| baseline |          0 |  0.043 |     0.046 |       0.104 |          345.278 |
| graphrag |          1 |  0.073 |     0.125 |       0.02  |           49.5   |

## By category

| system   | category     |   accuracy |   bleu |   rouge_l |   latency_s |   context_tokens |   n |
|:---------|:-------------|-----------:|-------:|----------:|------------:|-----------------:|----:|
| baseline | aggregation  |          0 |  0.049 |     0.069 |       0.103 |          352.2   |  10 |
| baseline | ambiguous    |          0 |  0.027 |     0     |       0.104 |          381.667 |   6 |
| baseline | comparison   |          0 |  0.049 |     0.069 |       0.105 |          338.125 |   8 |
| baseline | relationship |          0 |  0.034 |     0     |       0.106 |          307     |   6 |
| baseline | trend        |          0 |  0.048 |     0.069 |       0.103 |          345.167 |   6 |
| graphrag | aggregation  |          1 |  0.089 |     0.112 |       0.012 |           19.1   |  10 |
| graphrag | ambiguous    |          1 |  0.063 |     0.179 |       0.031 |           40     |   6 |
| graphrag | comparison   |          1 |  0.053 |     0.074 |       0.018 |           88.375 |   8 |
| graphrag | relationship |          1 |  0.109 |     0.286 |       0.031 |           42     |   6 |
| graphrag | trend        |          1 |  0.046 |     0     |       0.013 |           65.333 |   6 |

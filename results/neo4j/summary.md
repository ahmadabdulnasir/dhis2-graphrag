# GraphRAG vs vector-only baseline

## Overall

| system   |   accuracy |   bleu |   rouge_l |   latency_s |
|:---------|-----------:|-------:|----------:|------------:|
| baseline |          0 |  0.044 |     0.051 |       0.01  |
| graphrag |          1 |  0.07  |     0.113 |       0.076 |

## By category

| system   | category     |   accuracy |   bleu |   rouge_l |   latency_s |   n |
|:---------|:-------------|-----------:|-------:|----------:|------------:|----:|
| baseline | aggregation  |          0 |  0.049 |     0.07  |       0.011 |   4 |
| baseline | ambiguous    |          0 |  0.026 |     0     |       0.012 |   2 |
| baseline | comparison   |          0 |  0.049 |     0.069 |       0.01  |   4 |
| baseline | relationship |          0 |  0.036 |     0     |       0.011 |   2 |
| baseline | trend        |          0 |  0.049 |     0.069 |       0.009 |   3 |
| graphrag | aggregation  |          1 |  0.09  |     0.113 |       0.033 |   4 |
| graphrag | ambiguous    |          1 |  0.066 |     0.192 |       0.043 |   2 |
| graphrag | comparison   |          1 |  0.053 |     0.074 |       0.117 |   4 |
| graphrag | relationship |          1 |  0.109 |     0.286 |       0.171 |   2 |
| graphrag | trend        |          1 |  0.042 |     0     |       0.037 |   3 |

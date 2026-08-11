"""Evaluation metrics (S3.5).

Offline (no API): numeric correctness, categorical correctness, ambiguity
handling, BLEU and ROUGE-L (implemented here to avoid heavy dependencies),
and latency. RAGAS faithfulness/relevance run through evaluate_ragas() when
an LLM key and the ragas extra are installed.
"""
from __future__ import annotations

import math
import re
from collections import Counter

NUMERIC_TOLERANCE = 0.01  # answers within 1% of gold count as correct


# ---------- correctness ----------
def numbers_in(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"[\d][\d,]*\.?\d*", text)]


def numeric_correct(answer_text: str, gold: float) -> bool:
    for n in numbers_in(answer_text):
        if gold == 0 and n == 0:
            return True
        if gold != 0 and abs(n - gold) / abs(gold) <= NUMERIC_TOLERANCE:
            return True
    return False


def categorical_correct(answer_text: str, gold: str) -> bool:
    return gold.lower() in answer_text.lower()


def ambiguity_handled(answer_text: str) -> bool:
    """Correct behaviour = flag ambiguity / ask for specification."""
    return bool(re.search(r"ambiguous|under-specified|specify|which .*mean|clarif",
                          answer_text, re.I))


# ---------- BLEU (up to 4-gram, brevity penalty) ----------
def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    c_tok, r_tok = candidate.lower().split(), reference.lower().split()
    if not c_tok or not r_tok:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        c_ng, r_ng = _ngrams(c_tok, n), _ngrams(r_tok, n)
        overlap = sum((c_ng & r_ng).values())
        total = max(sum(c_ng.values()), 1)
        p = (overlap + 1) / (total + 1)  # add-1 smoothing
        log_p += math.log(p)
    bp = 1.0 if len(c_tok) > len(r_tok) else math.exp(1 - len(r_tok) / max(len(c_tok), 1))
    return bp * math.exp(log_p / max_n)


# ---------- ROUGE-L (F1 on longest common subsequence) ----------
def rouge_l(candidate: str, reference: str) -> float:
    a, b = candidate.lower().split(), reference.lower().split()
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] \
                else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    prec, rec = lcs / len(a), lcs / len(b)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


# ---------- RAGAS (optional, needs LLM) ----------
def evaluate_ragas(records: list[dict]) -> dict | None:
    """records: [{question, answer, contexts: [str], ground_truth}]

    Uses the ragas 0.2.x API (see the pinned `ragas` extra in pyproject.toml).
    Returns None with a printed reason when ragas cannot be imported.
    """
    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset
        from ragas.metrics import answer_relevancy, faithfulness
    except Exception as e:  # ModuleNotFoundError, or broken transitive deps
        print(f"[ragas] unavailable: {type(e).__name__}: {e}")
        print("[ragas] install the pinned extra: uv sync --extra ragas --extra llm")
        return None
    data = [{"user_input": r["question"], "response": r["answer"],
             "retrieved_contexts": list(r["contexts"]),
             "reference": r["ground_truth"]} for r in records]
    ds = EvaluationDataset.from_list(data)
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
    df = result.to_pandas()
    return {c: round(float(df[c].mean()), 4)
            for c in ("faithfulness", "answer_relevancy") if c in df.columns}

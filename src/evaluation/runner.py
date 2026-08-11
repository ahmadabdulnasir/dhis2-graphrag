"""Comparative experiment runner: GraphRAG vs vector-only baseline (S3.5).

For every evaluation query, both systems answer under identical conditions;
correctness (by category), BLEU/ROUGE-L against a gold reference sentence,
and latency are recorded. Results go to results/ as JSON + a markdown table
ready for Chapter 5.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.evaluation import metrics
from src.evaluation.queries import build_queries


def _is_correct(category: str, answer_text: str, gold_num, gold_text) -> bool:
    if category == "ambiguous":
        return metrics.ambiguity_handled(answer_text)
    if gold_num is not None:
        return metrics.numeric_correct(answer_text, gold_num)
    if category == "trend":
        return gold_text.lower() in answer_text.lower()
    return metrics.categorical_correct(answer_text, gold_text)


def _gold_reference(category, gold_num, gold_text, question) -> str:
    """A short reference sentence for BLEU/ROUGE."""
    if gold_num is not None:
        return f"The total is {gold_num:,.0f}."
    if category == "ambiguous":
        return "The question is ambiguous, please specify the disease or indicator."
    return f"The answer is {gold_text}."


def run_comparison(graphrag_answer_fn, baseline_answer_fn, df: pd.DataFrame,
                   out_dir: str = "results") -> dict:
    queries = build_queries()
    systems = {"graphrag": graphrag_answer_fn, "baseline": baseline_answer_fn}
    rows = []

    for q in queries:
        gold_num, gold_text = q.gold_fn(df)
        if gold_num is None and gold_text is None:
            continue  # dataset lacks the entities this query needs
        if gold_num == 0 and q.category == "aggregation":
            continue  # indicator/scope absent from this dataset
        reference = _gold_reference(q.category, gold_num, gold_text, q.question)
        for sys_name, fn in systems.items():
            t0 = time.perf_counter()
            ans = fn(q.question)
            latency = time.perf_counter() - t0
            correct = _is_correct(q.category, ans.text, gold_num, gold_text)
            rows.append({
                "qid": q.qid, "category": q.category, "system": sys_name,
                "question": q.question, "answer": ans.text,
                "gold": gold_num if gold_num is not None else gold_text,
                "correct": bool(correct),
                "bleu": round(metrics.bleu(ans.text, reference), 4),
                "rouge_l": round(metrics.rouge_l(ans.text, reference), 4),
                "latency_s": round(latency, 4),
                "n_supporting_values": ans.n_supporting_values,
                "context_chars": ans.context_chars,          # Experiment B
                "context_tokens_est": ans.context_chars // 4,
                "mode": ans.mode,
            })

    res = pd.DataFrame(rows)
    summary = (res.groupby(["system", "category"])
               .agg(accuracy=("correct", "mean"), bleu=("bleu", "mean"),
                    rouge_l=("rouge_l", "mean"), latency_s=("latency_s", "mean"),
                    context_tokens=("context_tokens_est", "mean"),
                    n=("correct", "size"))
               .round(3).reset_index())
    overall = (res.groupby("system")
               .agg(accuracy=("correct", "mean"), bleu=("bleu", "mean"),
                    rouge_l=("rouge_l", "mean"), latency_s=("latency_s", "mean"),
                    context_tokens=("context_tokens_est", "mean"))
               .round(3).reset_index())

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "query_results.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "per_category": summary.to_dict("records"),
        "overall": overall.to_dict("records"),
    }, indent=2))
    (out / "summary.md").write_text(
        "# GraphRAG vs vector-only baseline\n\n## Overall\n\n"
        + overall.to_markdown(index=False)
        + "\n\n## By category\n\n" + summary.to_markdown(index=False) + "\n")

    return {"detail": res, "summary": summary, "overall": overall}

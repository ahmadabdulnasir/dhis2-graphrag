"""LLM-mode evaluation: RAGAS faithfulness/relevance + hallucination rate.

Requires OPENAI_API_KEY in .env and the ragas extra:

    uv sync --extra ragas --extra llm
    uv run python scripts/run_ragas.py

Both systems generate through the LLM with identical grounding instructions;
only retrieval differs. Outputs to results/ragas/:
- ragas_scores.json: faithfulness + answer relevance per system
- hallucination.json: unsupported-number rate per system and the relative
  reduction (objective 5 target: >= 30%)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from src.evaluation.metrics import evaluate_ragas, numbers_in
from src.evaluation.queries import build_queries
from src.extraction.synthetic import generate


def hallucination_rate(rows: list[dict]) -> float:
    """Share of answers containing numbers absent from the retrieved context.

    A crude but transparent proxy: a number in the answer that never appears
    in the supplied context (within rounding) counts as unsupported. Ignored:
    years, percentages (usually derived arithmetic, not retrieved facts) and
    small numbers, which are typically phrasing ("top 5", "8 records").
    """
    bad = 0
    for r in rows:
        ctx_nums = set()
        for c in r["contexts"]:
            ctx_nums.update(numbers_in(c))
        # strip percentage expressions before extracting numbers
        answer_text = re.sub(r"[\d][\d,.]*\s*%", " ", r["answer"])
        answer_nums = [n for n in numbers_in(answer_text)
                       if not (1990 <= n <= 2030) and abs(n) > 20]
        unsupported = [n for n in answer_nums
                       if not any(abs(n - c) <= max(1.0, 0.005 * abs(c)) for c in ctx_nums)]
        bad += bool(unsupported)
    return bad / max(len(rows), 1)


def main():
    load_dotenv()
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — add it to .env first.")
        sys.exit(1)

    from run_pipeline import build_systems
    # build_systems picks the LLM answerer automatically when the key is set
    graphrag, baseline, clean, report, linker = build_systems(generate())

    queries = build_queries()
    out = {"graphrag": [], "baseline": []}
    for q in queries:
        gold_num, gold_text = q.gold_fn(clean)
        ground_truth = (f"The total is {gold_num:,.0f}." if gold_num is not None
                        else str(gold_text))
        for name, fn in (("graphrag", graphrag), ("baseline", baseline)):
            ans = fn(q.question)
            # the context actually supplied to the generator, for both systems
            contexts = ans.contexts or [""]
            out[name].append({"question": q.question, "answer": ans.text,
                              "contexts": contexts, "ground_truth": ground_truth,
                              "mode": ans.mode})
            print(f"  [{name}] {q.qid} done ({ans.mode})")

    res_dir = Path("results/ragas")
    res_dir.mkdir(parents=True, exist_ok=True)

    modes = {r["mode"] for rows in out.values() for r in rows}
    if not any(m.endswith("llm") for m in modes):
        print("WARNING: answers were generated in offline mode, not LLM mode. "
              "Check the OPENAI_API_KEY and rerun for the thesis numbers.")

    scores = {}
    for name, rows in out.items():
        print(f"Running RAGAS for {name}...")
        ragas_rows = [{k: r[k] for k in ("question", "answer", "contexts", "ground_truth")}
                      for r in rows]
        scores[name] = evaluate_ragas(ragas_rows) or \
            "ragas not installed — run: uv sync --extra ragas --extra llm"
    (res_dir / "ragas_scores.json").write_text(json.dumps(scores, indent=2))

    h_graph = hallucination_rate(out["graphrag"])
    h_base = hallucination_rate(out["baseline"])
    reduction = (h_base - h_graph) / h_base if h_base else 0.0
    hall = {"graphrag_hallucination_rate": round(h_graph, 3),
            "baseline_hallucination_rate": round(h_base, 3),
            "relative_reduction": round(reduction, 3),
            "target_reduction": 0.30,
            "target_met": bool(reduction >= 0.30)}
    (res_dir / "hallucination.json").write_text(json.dumps(hall, indent=2))

    print(json.dumps({"ragas": scores, "hallucination": hall}, indent=2))


if __name__ == "__main__":
    main()

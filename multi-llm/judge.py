"""Parallel stateless intermediate judging on the management node."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import json
from pathlib import Path

from pydantic import TypeAdapter
from core.schemas import DatasetRow, IntermediatePreference, Prompts, QuestionRecord, ScoringRequest
from core.topo_sort import topological_sort
from core.workflow import judge_pair, model_messages, save_json


def judge_collection(collection, workers=8):
    if workers < 1 or not (collection / "candidates.ready").exists():
        raise ValueError("Need positive concurrency and completed candidates")

    config = json.loads((collection / "collection_config.json").read_text())
    prompts = Prompts(**config["prompts"])
    rows = [DatasetRow(**row) for row in json.loads((collection / "dataset.json").read_text())["rows"]]

    (collection / "calls").mkdir(exist_ok=True)
    records, requests, preferences, failures = {}, [], [], []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {}

        for row in rows:
            record = TypeAdapter(QuestionRecord).validate_json((collection / f"question_{row.id}.json").read_text())
            record.intermediate_judgments = []
            record.preference_edges = []
            record.comparisons_accepted = False
            record.discard_reason = None
            records[row.id] = record

            candidates = {"y1": record.y1, "y2": record.y2, "y3": record.y3}
            if any(record.finish_reasons.get(name) != "stop" or not text or not text.strip()
                   for name, text in candidates.items()):
                record.discard_reason = "incomplete generation"
                continue

            ys = list(candidates.values())
            normalized = [y.strip() for y in ys]
            ref_ys = record.reference_intermediates
            if ref_ys is None:
                raise ValueError("Missing frozen-reference candidates; recollect with the new format")
            if any(not y.strip() for y in ref_ys):
                record.discard_reason = "empty reference generation"
                continue
            ref_normalized = [y.strip() for y in ref_ys]
            request = ScoringRequest(id=row.id, a_messages=record.a_messages, intermediates=ys,
                b_messages=[model_messages(row, prompts, prompts.b_instruction, y) for y in ys],
                final_answers=[row.chosen, row.rejected],
                candidate_mask=[y not in normalized[:i] for i, y in enumerate(normalized)],
                reference_intermediates=ref_ys,
                reference_b_messages=[model_messages(row, prompts, prompts.b_instruction, y) for y in ref_ys],
                reference_candidate_mask=[y not in ref_normalized[:i] for i, y in enumerate(ref_normalized)])
            requests.append(request)
            record.scoring_request = request

            if len(set(normalized)) < 3:
                record.discard_reason = "duplicate"
                continue

            for first, second in combinations(candidates, 2):
                future = pool.submit(judge_pair, collection, row, candidates[first], candidates[second],
                                     f"judge_{first}_{second}", first, second, prompts)
                jobs[future] = (row.id, first, second)

        for future in as_completed(jobs):
            row_id, first, second = jobs[future]
            record = records[row_id]
            try:
                decision = future.result()
                if decision.winner not in (first, second):
                    raise ValueError("Judge did not provide a strict preference")
                record.intermediate_judgments.append(decision)
                loser = second if decision.winner == first else first
                record.preference_edges.append((decision.winner, loser))
            except Exception as error:
                failures.append({"id": row_id, "pair": [first, second], "error": str(error)})

        for row in rows:
            record = records[row.id]
            if len(record.preference_edges) == 3:
                record.preference_edges.sort()
                record.comparisons_accepted = topological_sort(record.preference_edges)
                if record.comparisons_accepted:
                    candidates = {"y1": record.y1, "y2": record.y2, "y3": record.y3}
                    for chosen, rejected in record.preference_edges:
                        preferences.append(IntermediatePreference(id=row.id, x=row.question, system=row.system,
                            a_messages=record.a_messages, y_chosen=candidates[chosen], y_rejected=candidates[rejected],
                            finish_reasons=record.finish_reasons))
                else:
                    record.discard_reason = "cyclic preferences"

            save_json(collection / f"judged_{row.id}.json", record)

    save_json(collection / "scoring_requests.json", requests)
    save_json(collection / "intermediate_preferences.json", preferences)
    save_json(collection / "judging_summary.json", {"questions": len(rows), "system_rows": len(requests),
              "accepted_pairs": len(preferences), "errors": failures})

    if failures:
        raise RuntimeError(f"{len(failures)} judge calls failed; saved calls are retained")

    (collection / "preferences.ready").write_text("complete\n")
    print(f"Judged {len(rows)} questions; system rows={len(requests)}; accepted pairs={len(preferences)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()

    judge_collection(args.collection, args.workers)

"""Length-only preference filter rules."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core.workflow import save_json
from multi_llm.filter import drop_reasons, filter_split


def row(chosen, rejected, system="", id=0):
    return {"id": id, "system": system, "question": "q", "chosen": chosen, "rejected": rejected}


class FilterTests(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(drop_reasons(row("Yes", "Sure! Yes, the review is negative because it is dull.")),
                         ["C1", "C3"])
        self.assertEqual(drop_reasons(row("Vince", "It was Paul.", "Always provide explanation.")), ["C2"])
        self.assertEqual(drop_reasons(row("(2). Poor; the family struggles to pay rent",
                                          "Based on the text, the answer is (2) Poor.")), ["C3"])
        self.assertEqual(drop_reasons(row("B: be secretive", "Based on the context, A: Strong.")), [])
        self.assertEqual(drop_reasons(row("Six games were played in the final series.",
                                          "FIVE GAMES! *counting on fingers*")), [])

    def test_split_and_report(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            source = Path(temporary) / "train.json"
            save_json(source, [row("Yes", "Sure! Yes, it is.", id=1), row("B: be secretive", "A: Strong.", id=2)])
            report = filter_split(source, Path(temporary) / "train_filtered.json")
            self.assertEqual([r["id"] for r in json.loads((Path(temporary) / "train_filtered.json").read_text())], [2])
            self.assertEqual(report["dropped_ids"], {1: ["C1", "C3"]})
            self.assertTrue((Path(temporary) / "train_filtered_report.json").exists())


    def test_label_maps_shuffled_winner_and_reuses_saved_call(self):
        from unittest.mock import patch
        from multi_llm.filter import label_pair

        with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            directory = Path(temporary)
            for choice, expected in (("option1", None), ("same", "B_same_content"), ("tie", "D_tie")):
                pair = row("Six", "Five", id=f"{choice}")
                answer = {"answers": {"preference": {"type": "choice", "choice": choice}}}
                with patch("core.model_api.call_judge", return_value=answer) as api:
                    label = label_pair(directory, pair)
                    self.assertEqual(label_pair(directory, pair), label)
                    self.assertEqual(api.call_count, 1)
                order = json.loads((directory / f"{choice}.json").read_text())["presented_order"]
                self.assertEqual(api.call_args.args[0]["candidate_1"], pair[order[0]])
                if expected is None:
                    expected = "A_keep" if order[0] == "chosen" else "C_rejected_better"
                self.assertEqual(label, expected)


if __name__ == "__main__":
    unittest.main()

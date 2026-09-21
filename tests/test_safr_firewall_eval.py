import unittest

from aml_nextstep.safr_firewall import NO_EXCEPTION
from aml_nextstep.safr_firewall_data import generate_firewall_dataset
from aml_nextstep.safr_firewall_eval import (
    FirewallPrediction,
    WordNgramNaiveBayes,
    evaluate_predictions,
    keyword_baseline,
)


class SafrFirewallEvalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        examples = generate_firewall_dataset()
        cls.rows = [example.public_row() for example in examples]
        cls.train = [row for row in cls.rows if row["split"] == "train"]
        cls.test = [row for row in cls.rows if row["split"] == "test"]

    def test_perfect_predictions_score_one(self):
        predictions = {}
        for row in self.test:
            actions = tuple(target["control_action"] for target in row["targets"])
            predictions[row["example_id"]] = FirewallPrediction(
                example_id=row["example_id"],
                actions=actions,
                citations={
                    target["control_action"]: tuple(target["supporting_span_ids"])
                    for target in row["targets"]
                },
            )
        metrics = evaluate_predictions(self.test, predictions)
        for key in (
            "valid_output_rate",
            "exact_set_accuracy",
            "micro_f1",
            "macro_f1",
            "citation_f1",
            "contrast_pair_accuracy",
        ):
            self.assertEqual(metrics[key], 1.0)
        self.assertEqual(metrics["false_clear_rate"], 0.0)
        self.assertEqual(metrics["false_hold_rate"], 0.0)

    def test_always_clear_has_maximum_false_clear_rate(self):
        predictions = {
            row["example_id"]: FirewallPrediction(
                example_id=row["example_id"],
                actions=(NO_EXCEPTION,),
                citations={NO_EXCEPTION: ("action",)},
            )
            for row in self.test
        }
        metrics = evaluate_predictions(self.test, predictions)
        self.assertEqual(metrics["false_clear_rate"], 1.0)
        self.assertEqual(metrics["false_hold_rate"], 0.0)

    def test_frozen_cheap_baselines_do_not_solve_heldout_contrasts(self):
        keyword_metrics = evaluate_predictions(self.test, keyword_baseline(self.test))
        model = WordNgramNaiveBayes().fit(self.train)
        ngram_metrics = evaluate_predictions(self.test, model.predict(self.test))
        self.assertLess(keyword_metrics["contrast_pair_accuracy"], 0.5)
        self.assertLess(ngram_metrics["contrast_pair_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()

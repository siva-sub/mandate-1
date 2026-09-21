import json
import tempfile
import unittest
from pathlib import Path

from aml_nextstep.safr_contract_fixtures import (
    CONTROL_OPTIONS,
    SPLIT_COUNTS,
    build_needle_examples,
    generate_benchmark,
    needle_example,
    validate_release,
    write_benchmark_release,
)


class SafrContractFixturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = generate_benchmark()

    def test_expected_split_counts_and_no_world_leakage(self):
        counts = {
            split: sum(scenario.split == split for scenario in self.scenarios)
            for split in SPLIT_COUNTS
        }
        self.assertEqual(counts, SPLIT_COUNTS)
        world_to_split = {}
        for scenario in self.scenarios:
            prior = world_to_split.setdefault(scenario.world_id, scenario.split)
            self.assertEqual(prior, scenario.split)

    def test_semantic_targets_only_reference_supplied_evidence(self):
        for scenario in self.scenarios:
            valid = set(scenario.signed_envelope.envelope.evidence_ids)
            self.assertEqual(set(scenario.semantic_targets), set(CONTROL_OPTIONS))
            for control_id, target in scenario.semantic_targets.items():
                self.assertIn(target.finding, CONTROL_OPTIONS[control_id])
                self.assertLessEqual(set(target.supporting_evidence_ids), valid)

    def test_needle_example_does_not_expose_gate_answer(self):
        scenario = next(item for item in self.scenarios if item.semantic_applicable)
        example = needle_example(scenario, "scope_alignment")
        query = json.loads(example["query"])
        rendered = json.dumps(query, sort_keys=True)
        self.assertNotIn("expected_disposition", rendered)
        self.assertNotIn("expected_hard_controls", rendered)
        self.assertFalse(example["metadata"]["final_disposition_exposed_to_model"])
        self.assertFalse(example["metadata"]["hard_control_result_exposed_to_model"])
        self.assertEqual(
            set(example["tools"][0]["parameters"]["properties"]["finding"]["enum"]),
            set(CONTROL_OPTIONS["scope_alignment"]),
        )

    def test_candidate_order_augmentation_stays_inside_split(self):
        examples = build_needle_examples(self.scenarios, training_permutations=3)
        train_rows = examples["train"]
        by_scenario_control = {}
        for row in train_rows:
            key = (row["metadata"]["scenario_id"], row["metadata"]["control_id"])
            by_scenario_control.setdefault(key, []).append(tuple(row["metadata"]["option_order"]))
            self.assertEqual(row["metadata"]["split"], "train")
        self.assertTrue(by_scenario_control)
        self.assertTrue(all(len(rows) == 3 for rows in by_scenario_control.values()))

    def test_hard_control_failures_never_enter_semantic_training(self):
        examples = build_needle_examples(self.scenarios)
        eligible = {scenario.scenario_id for scenario in self.scenarios if scenario.semantic_applicable}
        for rows in examples.values():
            self.assertTrue(all(row["metadata"]["scenario_id"] in eligible for row in rows))

    def test_release_is_hash_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "release"
            manifest = write_benchmark_release(root)
            result = validate_release(root)
            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(manifest["scenario_count"], sum(SPLIT_COUNTS.values()))
            self.assertGreater(manifest["needle_examples_by_split"]["train"], 0)


if __name__ == "__main__":
    unittest.main()

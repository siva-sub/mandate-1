import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from aml_nextstep.safr_firewall import EXCEPTION_ACTIONS, FIREWALL_ACTIONS, NO_EXCEPTION
from aml_nextstep.safr_firewall_data import (
    SPLIT_WORLD_COUNTS,
    build_needle_firewall_data,
    generate_firewall_dataset,
    validate_firewall_release,
    write_firewall_release,
)


class SafrFirewallDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = generate_firewall_dataset()

    def test_every_world_is_a_split_local_contrast_pair(self):
        grouped = {}
        for example in self.examples:
            grouped.setdefault(example.contrast_group_id, []).append(example)
        self.assertEqual(len(grouped), sum(SPLIT_WORLD_COUNTS.values()))
        for rows in grouped.values():
            self.assertEqual(len(rows), 2)
            self.assertEqual({row.polarity for row in rows}, {"exception", "hard_negative"})
            self.assertEqual(len({row.split for row in rows}), 1)
            self.assertEqual(len({row.world_id for row in rows}), 1)

    def test_labels_are_balanced_and_citations_are_bounded(self):
        counts = Counter(action for example in self.examples for action in example.target_actions)
        self.assertEqual(counts[NO_EXCEPTION], sum(SPLIT_WORLD_COUNTS.values()))
        for action in EXCEPTION_ACTIONS:
            self.assertGreaterEqual(counts[action], 50)
        for example in self.examples:
            allowed = set(example.firewall_input.span_ids)
            for target in example.targets:
                self.assertIn(target.control_action, FIREWALL_ACTIONS)
                self.assertLessEqual(set(target.supporting_span_ids), allowed)

    def test_near_ood_is_domain_held_out(self):
        in_domain = {example.domain for example in self.examples if example.split != "ood"}
        ood = {example.domain for example in self.examples if example.split == "ood"}
        self.assertFalse(in_domain & ood)
        self.assertEqual(ood, {"sanctions_near_ood", "trade_finance_near_ood"})

    def test_needle_training_is_action_balanced_without_split_leakage(self):
        rows = build_needle_firewall_data(
            self.examples,
            exception_training_permutations=6,
            clean_training_permutations=1,
        )
        expected_train = SPLIT_WORLD_COUNTS["train"] * 7
        self.assertEqual(len(rows["train"]), expected_train)
        by_example = {}
        action_counts = Counter()
        for row in rows["train"]:
            metadata = row["metadata"]
            self.assertEqual(metadata["split"], "train")
            by_example.setdefault(metadata["example_id"], []).append(tuple(metadata["action_order"]))
            action_counts.update(answer["arguments"]["control_action"] for answer in row["answers"])
            query = json.loads(row["query"])
            self.assertNotIn("disposition", json.dumps(query))
            self.assertFalse(metadata["final_disposition_exposed_to_model"])
        self.assertEqual(set(action_counts.values()), {SPLIT_WORLD_COUNTS["train"]})
        for example in self.examples:
            if example.split == "train":
                expected = 6 if example.polarity == "exception" else 1
                self.assertEqual(len(by_example[example.example_id]), expected)

    def test_release_hashes_and_group_integrity_validate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "release"
            manifest = write_firewall_release(root)
            validation = validate_firewall_release(root)
            self.assertTrue(validation["ok"], validation["errors"])
            self.assertEqual(manifest["example_count"], 2 * sum(SPLIT_WORLD_COUNTS.values()))
            self.assertEqual(manifest["needle_examples"]["train"], 1260)


if __name__ == "__main__":
    unittest.main()

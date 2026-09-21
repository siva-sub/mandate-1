import json
import tempfile
import unittest
from pathlib import Path

from aml_nextstep.safr_firewall import FIREWALL_ACTIONS, NO_EXCEPTION
from aml_nextstep.safr_firewall_data import generate_firewall_dataset, write_firewall_release
from aml_nextstep.safr_gliner import (
    FINDING_LABELS,
    FINDING_TASK,
    NO_FINDING,
    SEMANTIC_PRIMITIVES,
    SUPPORT_ENTITY,
    decode_finding_output,
    decode_primitive_output,
    gliner_finding_training_record,
    gliner_training_record,
    primitive_labels_for_row,
    render_envelope,
    semantic_finding_label,
    span_marker,
    write_gliner_dataset,
)


class SafrGlinerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = generate_firewall_dataset()
        cls.positive = next(example.public_row() for example in cls.examples if example.polarity == "exception")
        group = cls.positive["contrast_group_id"]
        cls.clean = next(
            example.public_row()
            for example in cls.examples
            if example.contrast_group_id == group and example.polarity == "hard_negative"
        )

    def test_contrast_pair_changes_only_one_semantic_primitive(self):
        positive = primitive_labels_for_row(self.positive)
        clean = primitive_labels_for_row(self.clean)
        differences = [task for task in positive if positive[task] != clean[task]]
        self.assertEqual(len(differences), 1)
        self.assertEqual(
            clean,
            {item.task: item.nominal_label for item in SEMANTIC_PRIMITIVES},
        )

    def test_training_record_has_no_disposition_and_markers_resolve(self):
        record = gliner_training_record(self.positive)
        serialized = json.dumps(record)
        self.assertNotIn('"disposition":', serialized)
        true_labels = {
            value
            for task in record["output"]["classifications"]
            for value in task["true_label"]
        }
        self.assertFalse(true_labels & set(FIREWALL_ACTIONS))
        self.assertEqual(len(record["output"]["classifications"]), 1)
        clean_record = gliner_training_record(self.clean)
        positive_task = record["output"]["classifications"][0]
        clean_task = clean_record["output"]["classifications"][0]
        self.assertEqual(positive_task["task"], clean_task["task"])
        self.assertNotEqual(positive_task["true_label"], clean_task["true_label"])
        full_clean = gliner_training_record(self.clean, include_all_tasks=True)
        self.assertEqual(len(full_clean["output"]["classifications"]), 6)
        self.assertTrue(
            all(
                task["true_label"] == [item.nominal_label]
                for task, item in zip(
                    full_clean["output"]["classifications"],
                    SEMANTIC_PRIMITIVES,
                    strict=True,
                )
            )
        )
        text = record["input"]
        expected_markers = {
            span_marker(span_id)
            for span_id in self.positive["targets"][0]["supporting_span_ids"]
        }
        self.assertEqual(set(record["output"]["entities"][SUPPORT_ENTITY]), expected_markers)
        for marker in expected_markers:
            self.assertEqual(text.count(marker), 1)

    def test_finding_record_is_semantic_and_maps_raise_only(self):
        positive = gliner_finding_training_record(self.positive)
        clean = gliner_finding_training_record(self.clean)
        positive_task = positive["output"]["classifications"][0]
        clean_task = clean["output"]["classifications"][0]
        self.assertEqual(positive_task["task"], FINDING_TASK)
        self.assertEqual(tuple(positive_task["labels"]), FINDING_LABELS)
        self.assertEqual(clean_task["true_label"], [NO_FINDING])
        self.assertNotEqual(semantic_finding_label(self.positive), NO_FINDING)
        serialized = json.dumps(positive)
        self.assertFalse(any(action in serialized for action in FIREWALL_ACTIONS))

        output = {
            FINDING_TASK: {
                "label": semantic_finding_label(self.positive),
                "confidence": 0.9,
            }
        }
        prediction, receipt = decode_finding_output(self.positive, output)
        self.assertEqual(
            prediction.actions,
            (self.positive["targets"][0]["control_action"],),
        )
        self.assertEqual(receipt["semantic_finding"]["confidence"], 0.9)

    def test_renderer_does_not_expose_target_or_hashes(self):
        rendered = render_envelope(self.positive)
        self.assertIn("PROPOSED ACTION TYPE:", rendered)
        self.assertIn("[[SPAN:mandate]]", rendered)
        self.assertNotIn(self.positive["targets"][0]["control_action"], rendered)
        self.assertNotIn(self.positive["input"]["input_digest"], rendered)
        self.assertNotIn(self.positive["input"]["envelope_digest"], rendered)

    def test_decoder_maps_primitives_through_raise_only_table_and_filters_citations(self):
        output = {
            item.task: {
                "label": item.exception_label if item.control_action == "request_missing_evidence" else item.nominal_label,
                "confidence": 0.9,
            }
            for item in SEMANTIC_PRIMITIVES
        }
        output["entities"] = {
            SUPPORT_ENTITY: [
                {"text": "[[SPAN:action]]", "confidence": 0.9},
                {"text": "[[SPAN:not-present]]", "confidence": 0.99},
            ]
        }
        prediction, receipt = decode_primitive_output(self.positive, output, latency_seconds=0.1)
        self.assertEqual(prediction.status, "valid")
        self.assertEqual(prediction.actions, ("request_missing_evidence",))
        self.assertEqual(prediction.citations, {"request_missing_evidence": ("action",)})
        self.assertEqual(receipt["supporting_span_ids"], ["action"])

    def test_decoder_abstains_when_any_task_is_missing(self):
        output = {item.task: item.nominal_label for item in SEMANTIC_PRIMITIVES[:-1]}
        prediction, _ = decode_primitive_output(self.clean, output)
        self.assertEqual(prediction.status, "unavailable")
        self.assertEqual(prediction.actions, ())

    def test_clean_decoder_emits_no_exception(self):
        output: dict[str, object] = {
            item.task: item.nominal_label for item in SEMANTIC_PRIMITIVES
        }
        output["entities"] = {SUPPORT_ENTITY: ["[[SPAN:action]]"]}
        prediction, _ = decode_primitive_output(self.clean, output)
        self.assertEqual(prediction.actions, (NO_EXCEPTION,))
        self.assertEqual(prediction.citations, {NO_EXCEPTION: ("action",)})

    def test_writer_is_deterministic_and_covers_every_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "release"
            write_firewall_release(root)
            first = write_gliner_dataset(root)
            second = write_gliner_dataset(root)
            self.assertEqual(first, second)
            self.assertEqual(
                first["examples"],
                {"train": 1260, "dev": 210, "calibration": 60, "test": 120, "ood": 60},
            )
            self.assertEqual(first["source_examples"]["train"], 360)
            self.assertEqual(first["task_count"], 1)
            train_records = [
                json.loads(line)
                for line in (root / "gliner25" / "train.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(train_records), 1260)
            self.assertTrue(all("entities" not in record["output"] for record in train_records))
            label_counts = {}
            for record in train_records:
                task = record["output"]["classifications"][0]
                self.assertEqual(task["task"], FINDING_TASK)
                label = task["true_label"][0]
                label_counts[label] = label_counts.get(label, 0) + 1
            self.assertEqual(set(label_counts), set(FINDING_LABELS))
            self.assertTrue(all(count == 180 for count in label_counts.values()))


if __name__ == "__main__":
    unittest.main()

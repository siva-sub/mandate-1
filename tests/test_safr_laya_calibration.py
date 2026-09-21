import copy
import unittest

from scripts.kaggle_train_laya_safr_distill import apply_choice_temperature


class LayaCalibrationTest(unittest.TestCase):
    def test_fitted_temperature_replaces_every_inherited_choice_bucket(self):
        original = {
            "temperature": [1.01, 1.03, 1.05],
            "temperature_by_options": {
                "choice:2": 1.9,
                "choice:6-10": 1.00001585,
                "choice:11+": 0.1006,
                "noul:2": 1.98,
            },
        }
        before = copy.deepcopy(original)
        updated = apply_choice_temperature(original, 0.5)
        # This is the lookup used by the pinned Laya Agent.
        effective = updated["temperature_by_options"].get(
            "choice:6-10", updated["temperature"][0]
        )
        self.assertEqual(effective, 0.5)
        self.assertEqual(updated["temperature_by_options"], {"noul:2": 1.98})
        self.assertEqual(updated["temperature"][1:], [1.03, 1.05])
        self.assertEqual(original, before)

    def test_missing_tables_use_defaults(self):
        self.assertEqual(
            apply_choice_temperature({}, 2.0),
            {"temperature": [2.0, 1.0, 1.0], "temperature_by_options": {}},
        )

    def test_invalid_temperatures_fail_closed(self):
        for temperature in (float("nan"), float("inf"), -1.0, 0.49, 5.01):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                apply_choice_temperature({}, temperature)
        with self.assertRaises(ValueError):
            apply_choice_temperature({"temperature": []}, 1.0)


if __name__ == "__main__":
    unittest.main()

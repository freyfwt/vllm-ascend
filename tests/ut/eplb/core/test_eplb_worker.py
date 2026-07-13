import unittest

import numpy as np

from vllm_ascend.eplb.core.eplb_worker import EplbWorker


class TestEplbWorkerHotnessMetrics(unittest.TestCase):
    def setUp(self):
        self.worker = EplbWorker.__new__(EplbWorker)
        self.worker.zero_load_layers = set()
        self.worker.last_current_imbalance = {}
        self.worker.last_update_imbalance = {}
        self.current_placement = np.array(
            [
                [[0, 1], [2, 3]],
                [[0, 1], [2, 3]],
            ]
        )
        self.updated_placement = np.array(
            [
                [[0, 2], [1, 3]],
                [[0, 2], [1, 3]],
            ]
        )

    def test_zero_load_layer_uses_previous_imbalance(self):
        initial_hotness = np.array([[6, 2, 1, 1], [6, 2, 1, 1]])
        self.worker._update_hotness_metrics(self.current_placement, self.updated_placement, initial_hotness)
        previous_current = self.worker.last_current_imbalance[1]
        previous_update = self.worker.last_update_imbalance[1]

        current_hotness = np.array([[6, 2, 1, 1], [0, 0, 0, 0]])
        self.worker._update_hotness_metrics(self.current_placement, self.updated_placement, current_hotness)
        metrics = self.worker.latest_expert_hotness

        self.assertEqual(metrics["zero_load_layers"], [1])
        self.assertEqual(metrics["current_zero_load_layers"], [1])
        self.assertEqual(metrics["zero_group_current_mean"], previous_current)
        self.assertEqual(metrics["zero_group_update_mean"], previous_update)
        self.assertEqual(metrics["current_mean"], self.worker.last_current_imbalance[0])
        self.assertEqual(metrics["update_mean"], self.worker.last_update_imbalance[0])
        self.assertTrue(np.isfinite(metrics["current_mean"]))
        self.assertTrue(np.isfinite(metrics["update_mean"]))

    def test_tracked_layer_uses_new_value_after_load_recovers(self):
        self.worker._update_hotness_metrics(
            self.current_placement,
            self.updated_placement,
            np.array([[6, 2, 1, 1], [0, 0, 0, 0]]),
        )
        self.worker._update_hotness_metrics(
            self.current_placement,
            self.updated_placement,
            np.array([[6, 2, 1, 1], [8, 0, 0, 0]]),
        )
        metrics = self.worker.latest_expert_hotness

        self.assertEqual(metrics["zero_load_layers"], [1])
        self.assertEqual(metrics["current_zero_load_layers"], [])
        self.assertEqual(metrics["zero_group_current_mean"], 2.0)
        self.assertEqual(self.worker.last_current_imbalance[1], 2.0)
        self.assertEqual(metrics["current_mean"], self.worker.last_current_imbalance[0])

    def test_first_zero_observation_uses_neutral_value(self):
        self.worker._update_hotness_metrics(
            self.current_placement,
            self.updated_placement,
            np.zeros((2, 4), dtype=np.int64),
        )
        metrics = self.worker.latest_expert_hotness

        self.assertEqual(metrics["zero_load_layers"], [0, 1])
        self.assertEqual(metrics["current_mean"], 1.0)
        self.assertEqual(metrics["current_max"], 1.0)
        self.assertEqual(metrics["zero_group_current_mean"], 1.0)
        self.assertEqual(metrics["zero_group_current_max"], 1.0)
        self.assertTrue(np.all(np.isfinite(metrics["current_imbalance_list"])))


if __name__ == "__main__":
    unittest.main()

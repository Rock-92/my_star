from __future__ import annotations

import unittest
import uuid
from pathlib import Path

import numpy as np
import torch

from candidate_scorer.build_dataset import ShardWriter, label_candidates
from candidate_scorer.evaluate import _aggregate
from candidate_scorer.model import CenterAwareScorer
from candidate_scorer.pipeline import (
    add_center_channels,
    parse_candidate_methods,
    resolve_data_path,
    score_nms,
)
from candidate_scorer.probe_candidates import parse_sets, select_by_budget


class CandidateScorerV2Tests(unittest.TestCase):
    def test_global_matching_assigns_unique_candidates_and_targets(self) -> None:
        candidates = np.asarray([[0.0, 0.0], [0.0, 1.9], [0.0, 4.0]], dtype=np.float32)
        targets = np.asarray([[0.0, 1.0], [0.0, 3.9]], dtype=np.float32)
        classes, quality, offsets, distances = label_candidates(
            candidates, targets, positive_radius_px=2.0, ignore_radius_px=3.0
        )
        self.assertEqual(int(np.sum(classes == 2)), 2)
        self.assertEqual(classes[2], 2)
        self.assertAlmostEqual(float(offsets[2, 1]), -0.1, places=4)
        self.assertTrue(np.all(quality[classes == 2] > 0))
        self.assertTrue(np.all(np.isfinite(distances[classes == 2])))

    def test_score_nms_keeps_highest_score(self) -> None:
        points = np.asarray([[10.0, 10.0], [10.5, 10.5], [30.0, 30.0]], dtype=np.float32)
        keep = score_nms(points, np.asarray([0.8, 0.9, 0.7]), radius_px=2.0)
        self.assertEqual(set(keep.tolist()), {1, 2})

    def test_candidate_method_parser_supports_log(self) -> None:
        self.assertEqual(
            parse_candidate_methods("daofind:2.5,log:3.0"),
            [("daofind_like", 2.5), ("multiscale_log", 3.0)],
        )

    def test_manifest_data_model_path_is_rebased(self) -> None:
        root = Path("/root/my_star/data_model")
        result = resolve_data_path(root, r"data\data_model\val\images\sample.fit")
        self.assertEqual(result, root / "val" / "images" / "sample.fit")

    def test_probe_candidate_sets_parse(self) -> None:
        self.assertEqual(parse_sets("log=log:3.2"), [("log", "log:3.2")])

    def test_tile_budget_preserves_each_region(self) -> None:
        points = np.asarray(
            [[10.0, 10.0], [20.0, 20.0], [10.0, 600.0], [20.0, 620.0]],
            dtype=np.float32,
        )
        response = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        selected = select_by_budget(points, response, budget=1, mode="tile", tile_size=512)
        self.assertEqual(set(selected.tolist()), {0, 2})

    def test_center_channels_have_expected_shape(self) -> None:
        patches = np.zeros((2, 3, 31, 31), dtype=np.float32)
        result = add_center_channels(patches)
        self.assertEqual(result.shape, (2, 6, 31, 31))
        self.assertGreater(float(result[0, 3, 15, 15]), float(result[0, 3, 0, 0]))

    def test_model_outputs_multitask_heads(self) -> None:
        model = CenterAwareScorer(input_channels=6, feature_dim=16, width=8)
        output = model(
            torch.randn(3, 6, 31, 31),
            torch.randn(3, 6, 63, 63),
            torch.randn(3, 16),
        )
        self.assertEqual(tuple(output["class_logits"].shape), (3, 3))
        self.assertEqual(tuple(output["quality_logit"].shape), (3,))
        self.assertEqual(tuple(output["offset_yx"].shape), (3, 2))

    def test_shard_writer_round_trip(self) -> None:
        root = Path(__file__).resolve().parents[2] / "data" / f"test_shards_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        try:
            writer = ShardWriter(root, "train", shard_size=10)
            arrays = {
                "classes": np.asarray([0, 2], dtype=np.int64),
                "quality": np.asarray([0.0, 1.0], dtype=np.float32),
            }
            writer.add(arrays)
            writer.flush()
            with np.load(root / "shards" / "train_000000.npz") as data:
                np.testing.assert_array_equal(data["classes"], arrays["classes"])
        finally:
            shard = root / "shards" / "train_000000.npz"
            if shard.exists():
                shard.unlink()
            shard_dir = root / "shards"
            if shard_dir.exists():
                shard_dir.rmdir()
            root.rmdir()

    def test_micro_and_macro_metrics_are_distinct(self) -> None:
        samples = [
            {"thresholds": {"0.500000": {"pred_count": 10, "target_count": 10, "matched_count": 10, "f1": 1.0}}},
            {"thresholds": {"0.500000": {"pred_count": 100, "target_count": 10, "matched_count": 0, "f1": 0.0}}},
        ]
        result = _aggregate(samples, 0.5)
        self.assertAlmostEqual(float(result["macro_f1"]), 0.5)
        self.assertLess(float(result["micro_f1"]), 0.5)


if __name__ == "__main__":
    unittest.main()

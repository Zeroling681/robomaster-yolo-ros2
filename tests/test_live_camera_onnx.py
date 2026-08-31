import unittest

import numpy as np

from scripts.live_camera_onnx import DetectionSmoother, postprocess


def model_output(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32).T[None]


class PostprocessTests(unittest.TestCase):
    def test_nms_does_not_suppress_a_different_class(self) -> None:
        output = model_output(
            [
                [50, 50, 40, 40, 0.90, 0.05],
                [50, 50, 40, 40, 0.05, 0.85],
            ]
        )

        detections = postprocess(output, (100, 100), 1.0, 0, 0, 0.5, 0.45)

        self.assertEqual({detection[0] for detection in detections}, {0, 1})

    def test_uses_a_threshold_for_each_class(self) -> None:
        output = model_output(
            [
                [25, 25, 20, 20, 0.45, 0.05],
                [75, 75, 20, 20, 0.05, 0.45],
            ]
        )

        detections = postprocess(output, (100, 100), 1.0, 0, 0, (0.4, 0.5), 0.45)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0][0], 0)

    def test_clips_a_box_before_calculating_its_size(self) -> None:
        output = model_output([[0, 50, 40, 20, 0.90, 0.05]])

        detections = postprocess(output, (100, 100), 1.0, 0, 0, 0.5, 0.45)

        self.assertEqual(detections[0][2], (0, 40, 20, 20))


class DetectionSmootherTests(unittest.TestCase):
    def test_confirms_then_holds_a_short_dropout(self) -> None:
        smoother = DetectionSmoother(confirm_frames=2, hold_frames=2, smoothing=0.5)
        detection = (0, 0.8, (10, 10, 30, 20))

        self.assertEqual(smoother.update([detection]), [])
        self.assertEqual(len(smoother.update([detection])), 1)
        self.assertEqual(len(smoother.update([])), 1)
        self.assertEqual(len(smoother.update([])), 1)
        self.assertEqual(smoother.update([]), [])


if __name__ == "__main__":
    unittest.main()

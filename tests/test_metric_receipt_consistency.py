import math
import unittest

from scripts.verify_paper_reproduction import metric_receipt_matches


class MetricReceiptConsistencyTests(unittest.TestCase):
    def test_observed_cuda_ridge_reduction_roundoff(self):
        recomputed = 53.03248390555382
        saved = 53.03249508142471
        self.assertGreater(abs(recomputed - saved), 1e-5)
        self.assertTrue(metric_receipt_matches(recomputed, saved))
        self.assertTrue(metric_receipt_matches(saved, recomputed))

    def test_meaningful_millimeter_difference_is_rejected(self):
        self.assertFalse(metric_receipt_matches(53.03248390555382, 53.03348390555382))
        self.assertFalse(metric_receipt_matches(53.03348390555382, 53.03248390555382))

    def test_absolute_floor_for_near_zero_metrics(self):
        self.assertTrue(metric_receipt_matches(0.0, 9e-6))
        self.assertFalse(metric_receipt_matches(0.0, 2e-5))

    def test_identical_finite_metrics_are_accepted(self):
        for value in (0.0, 10.0, 53.03249508142471):
            with self.subTest(value=value):
                self.assertTrue(metric_receipt_matches(value, value))

    def test_nonfinite_receipts_or_recomputed_metrics_are_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(value=bad):
                self.assertFalse(metric_receipt_matches(53.03248390555382, bad))
                self.assertFalse(metric_receipt_matches(bad, 53.03248390555382))
                self.assertFalse(metric_receipt_matches(bad, bad))


if __name__ == "__main__":
    unittest.main()

import contextlib
import csv
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["MPLBACKEND"] = "Agg"

import numpy as np

from src.runner import run_nonkey_density_sweep as sweep


class DensitySweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="swarmjam-density-")
        self.root = Path(self.temp.name)
        self.assertEqual(self.root.resolve().parent, Path(tempfile.gettempdir()).resolve())
        self.addCleanup(self.temp.cleanup)
        self.csv_path = self.root / "nonkey_density_run_metrics.csv"

    def rows(self, methods=sweep._ASSOCIATION_METHODS):
        return [
            dict(method=method, seed=seed, key_targets=3, nonkey_targets=3 * ratio,
                 nonkey_to_key_ratio=ratio,
                 mean_gospa_m=(10 if method == sweep._ASSOCIATION_METHOD else 40) + ratio + seed,
                 total_id_switches=(3 if method == sweep._ASSOCIATION_METHOD else 9) + seed)
            for method in methods for ratio in (3, 4) for seed in (0, 1)
        ]

    def write_rows(self, rows):
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sweep._RUN_METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_legacy_csv_keeps_single_method_and_seed_order(self):
        self.write_rows(list(reversed(self.rows((sweep._ASSOCIATION_METHOD,)))))
        ratios, gospa, idsw, key_num = sweep._load_run_metrics(self.csv_path)
        self.assertEqual(key_num, 3)
        np.testing.assert_array_equal(ratios, [3, 4])
        self.assertEqual(list(gospa), [sweep._ASSOCIATION_METHOD])
        np.testing.assert_array_equal(gospa[sweep._ASSOCIATION_METHOD][3], [13, 14])
        np.testing.assert_array_equal(idsw[sweep._ASSOCIATION_METHOD][3], [3, 4])

    def test_methods_are_not_pooled(self):
        self.write_rows(list(reversed(self.rows())))
        _, gospa, idsw, _ = sweep._load_run_metrics(self.csv_path)
        self.assertEqual(list(gospa), list(sweep._ASSOCIATION_METHODS))
        np.testing.assert_array_equal(gospa[sweep._ASSOCIATION_METHOD][3], [13, 14])
        np.testing.assert_array_equal(gospa["bp_jpda"][3], [43, 44])
        np.testing.assert_array_equal(idsw["bp_jpda"][3], [9, 10])

    def test_unpaired_seeds_or_missing_ratio_are_rejected(self):
        rows = self.rows()
        for incomplete in (rows[:-1], rows[:-2]):
            with self.subTest(records=len(incomplete)):
                self.write_rows(incomplete)
                with self.assertRaisesRegex(ValueError, "same Monte Carlo seed set"):
                    sweep._load_run_metrics(self.csv_path)

    def test_duplicate_records_are_rejected(self):
        rows = self.rows()
        self.write_rows(rows + rows[:1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            sweep._load_run_metrics(self.csv_path)

    def test_resume_only_adds_missing_runs_and_can_be_repeated(self):
        rows = self.rows()
        self.write_rows(rows[:5])  # Complete IA sweep and one BP run.
        original = self.csv_path.read_bytes()
        trajectory = SimpleNamespace(bounds=np.array([200., 200., 20.]))
        with (
            patch.object(sweep, "_output_root", return_value=self.root),
            patch.object(sweep, "rollout_tracking_episode", return_value=(None, trajectory, None, None, None, None)) as rollout,
            patch.object(sweep, "summarize_key_target_tracking_episode", return_value={"mean_gospa": 50., "total_id_switches": 7}),
            patch.object(sweep, "plot_nonkey_density_sweep_csv", return_value={}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            sweep.run_nonkey_density_sweep(num_seeds=2, start_seed=0, ratios=(3, 4), resume=True)
            actual = [(c.kwargs["association_method"], c.kwargs["nonkey_num"], c.kwargs["seed"])
                      for c in rollout.call_args_list]
            self.assertEqual(actual, [("bp_jpda", 9, 1), ("bp_jpda", 12, 0), ("bp_jpda", 12, 1)])
            completed = self.csv_path.read_bytes()
            self.assertTrue(completed.startswith(original))
            records, _ = sweep._read_run_metrics(self.csv_path)
            self.assertEqual(len(records), 8)
            rollout.reset_mock()
            sweep.run_nonkey_density_sweep(num_seeds=2, start_seed=0, ratios=(3, 4), resume=True)
            rollout.assert_not_called()
            self.assertEqual(self.csv_path.read_bytes(), completed)

    def test_resume_rejects_incompatible_counts_or_seed_grid_before_writing(self):
        self.write_rows(self.rows((sweep._ASSOCIATION_METHOD,)))
        original = self.csv_path.read_bytes()
        with patch.object(sweep, "_output_root", return_value=self.root):
            for kwargs in ({"key_num": 2, "start_seed": 0}, {"key_num": 3, "start_seed": 2}):
                with self.subTest(**kwargs), self.assertRaises(ValueError):
                    sweep.run_nonkey_density_sweep(num_seeds=2, ratios=(3, 4), resume=True, **kwargs)
        self.assertEqual(self.csv_path.read_bytes(), original)

    def test_plotting_writes_separate_method_summaries_and_both_curves(self):
        self.write_rows(self.rows())
        figures = []
        original_close = sweep.plt.close

        def capture_close(figure):
            if hasattr(figure, "axes"):
                figures.append(figure)
            original_close(figure)

        original = self.csv_path.read_bytes()
        with (
            patch.object(sweep, "_output_root", return_value=self.root),
            patch.object(sweep.plt, "close", side_effect=capture_close),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            outputs = sweep.plot_nonkey_density_sweep_csv(
                csv_path=self.csv_path, show_confidence_interval=False,
            )
        self.assertEqual(self.csv_path.read_bytes(), original)
        with outputs["summary_csv"].open(newline="", encoding="utf-8") as f:
            summary = {(r["method"], int(r["nonkey_to_key_ratio"])): r for r in csv.DictReader(f)}
        self.assertEqual(len(summary), 4)
        self.assertEqual(float(summary[sweep._ASSOCIATION_METHOD, 3]["mean_gospa_m"]), 13.5)
        self.assertEqual(float(summary["bp_jpda", 3]["mean_gospa_m"]), 43.5)
        self.assertTrue(all(int(r["num_runs"]) == 2 for r in summary.values()))
        for figure in figures[:2]:
            self.assertEqual([line.get_label() for line in figure.axes[0].lines], ["IA-BP-JPDA", "BP-JPDA"])
            self.assertEqual(len(figure.axes[0].collections), 0)
        self.assertEqual(len(figures[2].axes), 2)
        left, right = figures[2].axes
        self.assertEqual([line.get_label() for line in left.lines], ["IA-BP-JPDA: GOSPA", "BP-JPDA: GOSPA"])
        self.assertEqual([line.get_label() for line in right.lines], ["IA-BP-JPDA: IDSW", "BP-JPDA: IDSW"])
        np.testing.assert_array_equal(left.lines[0].get_ydata(), [13.5, 14.5])
        np.testing.assert_array_equal(left.lines[1].get_ydata(), [43.5, 44.5])
        np.testing.assert_array_equal(right.lines[0].get_ydata(), [3.5, 3.5])
        np.testing.assert_array_equal(right.lines[1].get_ydata(), [9.5, 9.5])
        self.assertEqual(left.get_ylim(), right.get_ylim())
        self.assertGreater(left.get_ylim()[1], 44.5)
        for key in ("gospa_pdf", "idsw_pdf", "combined_pdf"):
            self.assertTrue(outputs[key].is_file())


if __name__ == "__main__":
    unittest.main()

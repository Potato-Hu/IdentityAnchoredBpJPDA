import unittest

import numpy as np

from src.entities.friendly_uav import FriendlyUAV
from src.sensing.association import (
    bp_jpda_association_for_friendlies,
    covariance_anchor_bp_jpda_association_for_friendlies,
    hard_anchor_bp_jpda_association_for_friendlies,
    labeled_anchor_bp_jpda_association_for_friendlies,
)
from src.sensing.global_sensor import GlobalKeyTargetStatePrior
from src.sensing.local_sensor import LocalTargetObservation
from src.tracking.controller import build_tracking_control_commands
from src.tracking.fusion import build_fused_track_set
from src.tracking.task_assignment import build_hungarian_task_assignment


METHODS = (
    bp_jpda_association_for_friendlies,
    hard_anchor_bp_jpda_association_for_friendlies,
    covariance_anchor_bp_jpda_association_for_friendlies,
    labeled_anchor_bp_jpda_association_for_friendlies,
)


class BPJPDAInitializationTests(unittest.TestCase):
    def setUp(self):
        self.friendly = np.array([[90., 90., 8., 0., 0., 0.]])
        self.reports = np.array([[110., 110., 8., 1., 0., 0.], [115., 105., 8., 0., 1., 0.]])

    def step(self, method, mask, previous=None, model="imm", candidates=()):
        mask = np.asarray(mask, dtype=bool)
        states = self.reports[:len(mask)].copy()
        states[~mask] = 0.
        candidates = np.asarray(candidates, dtype=float).reshape(-1, 6)
        return method(
            friendly_states=self.friendly,
            global_target_priors=GlobalKeyTargetStatePrior(
                states=states, covariance_diagonal=np.array([100., 100., 4., 16., 16., .16]),
                valid_mask=mask,
            ),
            local_observations={"uav": LocalTargetObservation(
                relative_states=candidates, mask=np.ones(len(candidates)),
                covariance_diagonal=np.array([4., 4., .16, 1., 1., .04]),
            )},
            agent_ids=("uav",), previous_filter_states=previous,
            detection_radius_m=90., relative_velocity_limit_mps=9.,
            local_detection_probability=.95, local_detection_probability_min=.5,
            detection_full_probability_range_m=20., detection_density_penalty_alpha=.1,
            clutter_rate_per_second=.6, clutter_hotspot_probability=.5,
            clutter_hotspot_position_std_m=(8., 8., 2.),
            clutter_hotspot_velocity_std_mps=(2., 2., .5),
            recursive_motion_model=model, dt_seconds=.5, max_missed_steps=3,
        )

    def test_missing_reports_never_create_tracks_even_with_anonymous_candidates(self):
        for method in METHODS:
            for model in ("cv", "imm"):
                with self.subTest(method=method.__name__, model=model):
                    previous = None
                    for _ in range(8):
                        outputs, previous = self.step(
                            method, [False], previous, model,
                            candidates=[-self.friendly[0], self.reports[0] - self.friendly[0]],
                        )
                        self.assertEqual(previous["uav"], {})
                        self.assertFalse(outputs["uav"].tracks[0].valid)

    def test_first_report_initializes_without_fusing_placeholder_state(self):
        for method in METHODS:
            for model in ("cv", "imm"):
                with self.subTest(method=method.__name__, model=model):
                    previous = None
                    for _ in range(2):
                        _, previous = self.step(method, [False], previous, model)
                    outputs, previous = self.step(method, [True], previous, model)
                    expected = self.reports[0] - self.friendly[0]
                    np.testing.assert_allclose(previous["uav"][0].mean, expected, atol=1e-6)
                    np.testing.assert_allclose(
                        outputs["uav"].tracks[0].posterior_relative_state, expected, atol=1e-6,
                    )

    def test_missing_label_does_not_inherit_another_labels_initialization(self):
        for method in METHODS:
            with self.subTest(method=method.__name__):
                _, previous = self.step(method, [False, True])
                self.assertEqual(set(previous["uav"]), {1})
                _, previous = self.step(method, [False, False], previous)
                self.assertEqual(set(previous["uav"]), {1})
                _, previous = self.step(method, [True, False], previous)
                np.testing.assert_allclose(
                    previous["uav"][0].mean, self.reports[0] - self.friendly[0], atol=1e-6,
                )

    def test_established_fallback_expires_without_placeholder_rebirth(self):
        method = labeled_anchor_bp_jpda_association_for_friendlies
        _, previous = self.step(method, [True])
        for missed in range(1, 6):
            outputs, previous = self.step(method, [False], previous)
            self.assertEqual(outputs["uav"].tracks[0].valid, missed <= 3)
            self.assertEqual(bool(previous["uav"]), missed < 3)

    def test_initialized_out_of_range_track_is_retained_until_expiry(self):
        self.reports[0, :3] = [300., 300., 8.]
        for method in METHODS:
            with self.subTest(method=method.__name__):
                _, previous = self.step(method, [True])
                self.assertIn(0, previous["uav"])
                _, previous = self.step(method, [False], previous)
                self.assertIn(0, previous["uav"])

    def test_controller_waits_for_first_report_then_pursues_reported_target(self):
        previous = None
        for mask in ([False], [False], [True]):
            outputs, previous = self.step(labeled_anchor_bp_jpda_association_for_friendlies, mask, previous)
            fused = build_fused_track_set(
                timestamp=0, track_outputs=outputs, friendly_states=self.friendly, agent_ids=("uav",),
                self_position_std_m=(0., 0., 0.), min_confidence=.1,
                mahalanobis_distance_threshold=3., ci_search_steps=2,
            )
            assignment = build_hungarian_task_assignment(
                ("uav",), timestamp=0, friendly_states=self.friendly, fused_track_set=fused,
            )
            commands = build_tracking_control_commands(
                ("uav",), friendly_states=self.friendly, assignment=assignment, fused_track_set=fused,
            )
            action = commands["uav"].action
            if not mask[0]:
                self.assertEqual(action, 0)
            else:
                np.testing.assert_allclose(fused.tracks[0].fused_state_mean, self.reports[0], atol=1e-6)
                direction = FriendlyUAV.ACTION_DIRECTIONS[action]
                self.assertGreater(float(direction @ (self.reports[0, :3] - self.friendly[0, :3])), 0.)


if __name__ == "__main__":
    unittest.main()

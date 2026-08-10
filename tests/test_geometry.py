"""Tests for reactiontools.tools_geometry."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.optimize import FIRE

from reactiontools import (ConvergenceError,
                           ConvergenceWarning,
                           bonded_cluster_indices_no_anchor_hub,
                           flip_and_face_bases,
                           get_best_flip_and_face_bases,
                           get_dimer_bonded_cluster_indices,
                           optimize_with_fixed_anchors,
                           swap_bonding_configuration)
from reactiontools.tools_geometry import (_orient_normal_toward,
                                          _pca_frame,
                                          _rigid_transform)


@pytest.fixture
def dimer():
    """Two water molecules stacked along z, far enough apart not to bond.

    Atom order is O, H, H for each, so the oxygens are indices 0 and 3 and
    make the natural anchors.
    """
    lower = molecule("H2O")
    upper = molecule("H2O")
    upper.translate([0.0, 0.0, 3.0])
    return lower + upper


class TestBondedClusterIndices:
    def test_finds_one_half_of_a_dimer(self, dimer):
        """Anchored on the lower oxygen, the walk must not reach the upper."""
        cluster = bonded_cluster_indices_no_anchor_hub(dimer, 0)

        assert set(cluster) <= {0, 1, 2}

    def test_includes_the_anchor(self, dimer):
        assert 0 in bonded_cluster_indices_no_anchor_hub(dimer, 0)

    def test_returns_sorted_indices(self, dimer):
        cluster = bonded_cluster_indices_no_anchor_hub(dimer, 3)

        assert cluster == sorted(cluster)

    def test_rejects_an_out_of_range_anchor(self, dimer):
        with pytest.raises(IndexError):
            bonded_cluster_indices_no_anchor_hub(dimer, len(dimer))

    def test_rejects_a_negative_anchor(self, dimer):
        with pytest.raises(IndexError):
            bonded_cluster_indices_no_anchor_hub(dimer, -1)

    def test_a_larger_cutoff_finds_at_least_as_much(self, dimer):
        tight = bonded_cluster_indices_no_anchor_hub(dimer, 0, mult=1.0)
        loose = bonded_cluster_indices_no_anchor_hub(dimer, 0, mult=2.0)

        assert set(tight) <= set(loose)


class TestGetDimerBondedClusterIndices:
    def test_merges_both_halves(self, dimer):
        both = get_dimer_bonded_cluster_indices(dimer, [0, 3])

        assert set(both) == (set(bonded_cluster_indices_no_anchor_hub(dimer, 0))
                             | set(bonded_cluster_indices_no_anchor_hub(dimer, 3)))

    def test_has_no_duplicates(self, dimer):
        both = get_dimer_bonded_cluster_indices(dimer, [0, 3])

        assert len(both) == len(set(both))

    def test_rejects_the_wrong_number_of_anchors(self, dimer):
        with pytest.raises(ValueError, match="exactly two indices"):
            get_dimer_bonded_cluster_indices(dimer, [0])

    def test_rejects_the_wrong_number_of_mults(self, dimer):
        with pytest.raises(ValueError, match="exactly two values"):
            get_dimer_bonded_cluster_indices(dimer, [0, 3], mults=[1.0])


class TestPcaFrame:
    def test_origin_is_the_centroid(self):
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

        origin, _R = _pca_frame(pts)

        assert origin == pytest.approx(pts.mean(axis=0))

    def test_frame_is_orthonormal(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(8, 3))

        _origin, R = _pca_frame(pts)

        assert R.T @ R == pytest.approx(np.eye(3), abs=1e-10)

    def test_frame_is_right_handed(self):
        rng = np.random.default_rng(1)

        _origin, R = _pca_frame(rng.normal(size=(8, 3)))

        assert np.linalg.det(R) == pytest.approx(1.0)

    def test_normal_is_perpendicular_to_a_planar_group(self):
        """For points in the xy-plane the smallest-variance axis is z."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

        _origin, R = _pca_frame(pts)

        assert abs(abs(R[:, 2] @ np.array([0.0, 0.0, 1.0])) - 1.0) < 1e-10


class TestOrientNormalToward:
    def test_flips_a_normal_pointing_away(self):
        R = np.eye(3)  # normal is +z

        flipped = _orient_normal_toward(R, [0, 0, 0], [0, 0, -1])

        assert flipped[:, 2] == pytest.approx([0.0, 0.0, -1.0])

    def test_leaves_a_normal_already_facing_the_target(self):
        R = np.eye(3)

        kept = _orient_normal_toward(R, [0, 0, 0], [0, 0, 1])

        assert kept == pytest.approx(R)

    def test_stays_right_handed_after_flipping(self):
        flipped = _orient_normal_toward(np.eye(3), [0, 0, 0], [0, 0, -1])

        assert np.linalg.det(flipped) == pytest.approx(1.0)


class TestRigidTransform:
    def test_moves_the_anchor_onto_its_target(self):
        pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        anchor = np.array([1.0, 0.0, 0.0])

        moved = _rigid_transform(pts, anchor, np.eye(3), [5.0, 5.0, 5.0])

        assert moved[0] == pytest.approx([5.0, 5.0, 5.0])

    def test_preserves_internal_distances(self):
        rng = np.random.default_rng(2)
        pts = rng.normal(size=(5, 3))
        _origin, R = _pca_frame(rng.normal(size=(6, 3)))

        moved = _rigid_transform(pts, pts[0], R, [1.0, 2.0, 3.0])

        before = np.linalg.norm(pts[1:] - pts[0], axis=1)
        after = np.linalg.norm(moved[1:] - moved[0], axis=1)
        assert after == pytest.approx(before)


class TestFlipAndFaceBases:
    def test_swaps_the_two_halves_over(self, dimer):
        """Each fragment should land on the other's anchor."""
        base_a, base_b = [0, 1, 2], [3, 4, 5]
        anchor_a, anchor_b = dimer.positions[0].copy(), dimer.positions[3].copy()

        swapped = flip_and_face_bases(dimer, base_a, base_b, [0, 3])

        assert swapped.positions[0] == pytest.approx(anchor_b)
        assert swapped.positions[3] == pytest.approx(anchor_a)

    def test_does_not_modify_the_input(self, dimer):
        before = dimer.positions.copy()

        flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5], [0, 3])

        assert dimer.positions == pytest.approx(before)

    def test_moves_each_fragment_rigidly(self, dimer):
        """A swap is a rigid motion, so internal distances must survive."""
        swapped = flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5], [0, 3])

        for frag in ([0, 1, 2], [3, 4, 5]):
            before = dimer[frag].get_all_distances()
            after = swapped[frag].get_all_distances()
            assert after == pytest.approx(before, abs=1e-9)

    def test_leaves_atoms_outside_both_fragments_alone(self, dimer):
        spectator = dimer + Atoms("He", positions=[[8.0, 8.0, 8.0]])

        swapped = flip_and_face_bases(spectator, [0, 1, 2], [3, 4, 5], [0, 3])

        assert swapped.positions[6] == pytest.approx([8.0, 8.0, 8.0])

    def test_a_different_reflection_gives_a_different_structure(self, dimer):
        default = flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5], [0, 3])
        other = flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5], [0, 3],
                                    rot_matrix=[1.0, -1.0, -1.0])

        assert default.positions != pytest.approx(other.positions)


class TestOptimizeWithFixedAnchors:
    def test_actually_moves_the_fragment_atoms(self, calc, dimer):
        """Regression: the result was written to a throwaway copy.

        `atoms_out[selection].set_positions(...)` builds a new Atoms object
        and updates that, so the relaxed coordinates were discarded and the
        function returned its input untouched.
        """
        strained = flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5], [0, 3])

        relaxed = optimize_with_fixed_anchors(strained, [0, 1, 2], [3, 4, 5],
                                              [0, 3], calc, fmax=0.5)

        assert relaxed.positions != pytest.approx(strained.positions)

    def test_does_not_modify_the_input(self, calc, dimer):
        before = dimer.positions.copy()

        optimize_with_fixed_anchors(dimer, [0, 1, 2], [3, 4, 5], [0, 3],
                                    calc, fmax=0.5)

        assert dimer.positions == pytest.approx(before)

    def test_keeps_every_atom(self, calc, dimer):
        relaxed = optimize_with_fixed_anchors(dimer, [0, 1, 2], [3, 4, 5],
                                              [0, 3], calc, fmax=0.5)

        assert len(relaxed) == len(dimer)
        assert relaxed.get_chemical_symbols() == dimer.get_chemical_symbols()

    def test_leaves_atoms_outside_both_fragments_alone(self, calc, dimer):
        spectator = dimer + Atoms("He", positions=[[8.0, 8.0, 8.0]])

        relaxed = optimize_with_fixed_anchors(spectator, [0, 1, 2], [3, 4, 5],
                                              [0, 3], calc, fmax=0.5)

        assert relaxed.positions[6] == pytest.approx([8.0, 8.0, 8.0])

    def test_records_convergence_on_the_result(self, calc, dimer):
        relaxed = optimize_with_fixed_anchors(dimer, [0, 1, 2], [3, 4, 5],
                                              [0, 3], calc, fmax=0.5)

        assert relaxed.info["converged"] is True

    def test_uses_the_optimiser_it_is_given(self, calc, dimer, capsys):
        optimize_with_fixed_anchors(dimer, [0, 1, 2], [3, 4, 5], [0, 3], calc,
                                    fmax=0.5, optimiser=FIRE)

        log = capsys.readouterr().out
        assert "FIRE" in log
        assert "BFGS" not in log

    def test_no_logfile_silences_it(self, calc, dimer, capsys):
        optimize_with_fixed_anchors(dimer, [0, 1, 2], [3, 4, 5], [0, 3], calc,
                                    fmax=0.5, logfile=None)

        assert capsys.readouterr().out == ""

    def test_warns_and_records_when_it_runs_out_of_steps(self, calc, dimer):
        strained = flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5], [0, 3])

        with pytest.warns(ConvergenceWarning, match="Fixed-anchor relaxation"):
            relaxed = optimize_with_fixed_anchors(strained, [0, 1, 2],
                                                  [3, 4, 5], [0, 3], calc,
                                                  fmax=1e-3, steps=2)

        assert relaxed.info["converged"] is False

    def test_raises_instead_when_asked(self, calc, dimer):
        strained = flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5], [0, 3])

        with pytest.raises(ConvergenceError):
            optimize_with_fixed_anchors(strained, [0, 1, 2], [3, 4, 5], [0, 3],
                                        calc, fmax=1e-3, steps=2,
                                        raise_on_unconverged=True)


class TestGetBestFlipAndFaceBases:
    def test_returns_a_structure_without_optimising(self, dimer):
        swapped = get_best_flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5],
                                               [0, 3], optimise_after=False)

        assert len(swapped) == len(dimer)

    def test_picks_a_reflection_at_least_as_good_as_the_default(self, dimer):
        """The search exists to beat the hard-coded default sign choice."""
        base_a, base_b = [0, 1, 2], [3, 4, 5]

        best = get_best_flip_and_face_bases(dimer, base_a, base_b, [0, 3],
                                            optimise_after=False)
        default = flip_and_face_bases(dimer, base_a, base_b, [0, 3])

        def separation(atoms):
            return np.linalg.norm(atoms[base_a].get_center_of_mass()
                                  - atoms[base_b].get_center_of_mass())

        assert separation(best) <= separation(default) + 1e-9

    def test_requires_a_calculator_when_optimising(self, dimer):
        with pytest.raises(ValueError, match="needs a calculator"):
            get_best_flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5], [0, 3],
                                         optimise_after=True)

    def test_optimises_when_given_a_calculator(self, calc, dimer):
        relaxed = get_best_flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5],
                                               [0, 3], optimise_after=True,
                                               calc=calc)

        assert len(relaxed) == len(dimer)

    def test_does_not_modify_the_input(self, dimer):
        before = dimer.positions.copy()

        get_best_flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5], [0, 3],
                                     optimise_after=False)

        assert dimer.positions == pytest.approx(before)


class TestSwapBondingConfiguration:
    """O-H...O becoming O...H-O, the product end state of a proton transfer."""

    @pytest.fixture
    def h_bond(self):
        """A collinear O-H...O along x, with O-H = 1.0 and O...O = 2.8."""
        return Atoms("OHO", positions=[[0.0, 0.0, 0.0],
                                       [1.0, 0.0, 0.0],
                                       [2.8, 0.0, 0.0]])

    def test_hydrogen_moves_to_the_acceptor(self, h_bond):
        swapped = swap_bonding_configuration(h_bond, 0, 1, 2)

        # Was 1.0 from the donor; now 1.0 from the acceptor instead.
        assert np.linalg.norm(swapped.positions[1] - swapped.positions[2]) \
            == pytest.approx(1.0)

    def test_the_new_bond_length_matches_the_old_one(self, h_bond):
        h_bond.positions[1] = [0.7, 0.0, 0.0]

        swapped = swap_bonding_configuration(h_bond, 0, 1, 2)

        assert np.linalg.norm(swapped.positions[1] - swapped.positions[2]) \
            == pytest.approx(0.7)

    def test_the_hydrogen_stays_between_the_heavy_atoms(self, h_bond):
        swapped = swap_bonding_configuration(h_bond, 0, 1, 2)

        assert swapped.positions[1][0] == pytest.approx(1.8)

    def test_the_heavy_atoms_do_not_move(self, h_bond):
        swapped = swap_bonding_configuration(h_bond, 0, 1, 2)

        assert swapped.positions[0] == pytest.approx(h_bond.positions[0])
        assert swapped.positions[2] == pytest.approx(h_bond.positions[2])

    def test_does_not_modify_the_input(self, h_bond):
        before = h_bond.positions.copy()

        swap_bonding_configuration(h_bond, 0, 1, 2)

        assert h_bond.positions == pytest.approx(before)

    def test_works_off_axis(self, h_bond):
        """The donor->acceptor direction is what matters, not the frame."""
        h_bond.rotate(37, "z")
        h_bond.rotate(-19, "y")

        swapped = swap_bonding_configuration(h_bond, 0, 1, 2)

        assert np.linalg.norm(swapped.positions[1] - swapped.positions[2]) \
            == pytest.approx(1.0)

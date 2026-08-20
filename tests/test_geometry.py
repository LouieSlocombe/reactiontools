"""Tests for reactiontools.tools_geometry."""

import warnings
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import FIRE

from reactiontools import (
    ConvergenceError,
    ConvergenceWarning,
    SeedWarning,
    align_atom_sets,
    atom_set_rmsd,
    bonded_cluster_indices_no_anchor_hub,
    flip_and_face_bases,
    get_best_flip_and_face_bases,
    get_dimer_bonded_cluster_indices,
    kabsch_transform,
    optimize_with_fixed_anchors,
    seed_product_from_ts,
    swap_bonding_configuration,
)
from reactiontools.tools_geometry import (
    _orient_normal_toward,
    _pca_frame,
    _rigid_transform,
)


@pytest.fixture
def dimer() -> Atoms:
    """Two water molecules stacked along z, far enough apart not to bond.

    Atom order is O, H, H for each, so the oxygens are indices 0 and 3 and
    make the natural anchors.
    """
    lower = molecule("H2O")
    upper = molecule("H2O")
    upper.translate([0.0, 0.0, 3.0])
    return lower + upper


@pytest.fixture
def rigid_pair() -> tuple[Atoms, Atoms, np.ndarray, np.ndarray]:
    """A non-degenerate structure and an exact rigidly transformed copy."""
    mobile = Atoms(
        "CHNO",
        positions=[
            [0.0, 0.0, 0.0],
            [1.2, 0.1, -0.2],
            [-0.3, 1.1, 0.4],
            [0.2, -0.4, 1.3],
        ],
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([3.0, -2.0, 0.7])
    reference = mobile.copy()
    reference.set_positions(mobile.positions @ rotation + translation)
    return mobile, reference, rotation, translation


class TestKabschTransform:
    def test_recovers_an_exact_rigid_transform(
        self,
        rigid_pair: tuple[Atoms, Atoms, np.ndarray, np.ndarray],
    ) -> None:
        mobile, reference, expected_rotation, expected_translation = rigid_pair

        rotation, translation = kabsch_transform(
            mobile.positions, reference.positions
        )

        assert rotation == pytest.approx(expected_rotation, abs=1e-12)
        assert translation == pytest.approx(expected_translation, abs=1e-12)

    def test_returns_a_proper_orthogonal_rotation(
        self,
        rigid_pair: tuple[Atoms, Atoms, np.ndarray, np.ndarray],
    ) -> None:
        mobile, reference, _rotation, _translation = rigid_pair

        rotation, _translation = kabsch_transform(
            mobile.positions, reference.positions
        )

        assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)

    def test_does_not_use_a_reflection(self) -> None:
        mobile = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )
        reflected = mobile.copy()
        reflected[:, 0] *= -1.0

        rotation, _translation = kabsch_transform(mobile, reflected)

        assert np.linalg.det(rotation) == pytest.approx(1.0)

    def test_supports_a_single_point_translation(self) -> None:
        rotation, translation = kabsch_transform([[1, 2, 3]], [[4, 6, 8]])

        assert np.array([[1, 2, 3]]) @ rotation + translation == pytest.approx(
            np.array([[4, 6, 8]])
        )

    def test_supports_zero_weight_correspondences(self) -> None:
        mobile = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        reference = np.array([[2.0, 1.0, 0.0], [3.0, 1.0, 0.0], [-50.0, 0.0, 0.0]])

        rotation, translation = kabsch_transform(
            mobile, reference, weights=[1.0, 1.0, 0.0]
        )
        fitted = mobile @ rotation + translation

        assert fitted[:2] == pytest.approx(reference[:2], abs=1e-12)

    @pytest.mark.parametrize(
        ("mobile", "reference", "message"),
        [
            ([], [], "shape"),
            ([[0.0, 0.0]], [[0.0, 0.0]], "shape"),
            ([[0.0, 0.0, 0.0]], np.zeros((2, 3)), "same shape"),
            ([[np.nan, 0.0, 0.0]], [[0.0, 0.0, 0.0]], "finite"),
        ],
    )
    def test_rejects_invalid_positions(
        self,
        mobile: list[list[float]],
        reference: list[list[float]] | np.ndarray,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            kabsch_transform(mobile, reference)

    @pytest.mark.parametrize(
        ("weights", "message"),
        [
            ([1.0], "shape"),
            ([1.0, -1.0], "negative"),
            ([0.0, 0.0], "greater than zero"),
            ([1.0, np.inf], "finite"),
        ],
    )
    def test_rejects_invalid_weights(self, weights: list[float], message: str) -> None:
        positions = np.zeros((2, 3))

        with pytest.raises(ValueError, match=message):
            kabsch_transform(positions, positions, weights=weights)


class TestAlignAtomSets:
    def test_superposes_an_exact_rigid_copy(
        self,
        rigid_pair: tuple[Atoms, Atoms, np.ndarray, np.ndarray],
    ) -> None:
        mobile, reference, _rotation, _translation = rigid_pair

        aligned = align_atom_sets(mobile, reference)

        assert aligned.positions == pytest.approx(reference.positions, abs=1e-12)

    def test_does_not_modify_either_input(
        self,
        rigid_pair: tuple[Atoms, Atoms, np.ndarray, np.ndarray],
    ) -> None:
        mobile, reference, _rotation, _translation = rigid_pair
        mobile_before = mobile.positions.copy()
        reference_before = reference.positions.copy()

        align_atom_sets(mobile, reference)

        assert mobile.positions == pytest.approx(mobile_before)
        assert reference.positions == pytest.approx(reference_before)

    def test_moves_the_whole_structure_when_fitting_a_subset(
        self,
        rigid_pair: tuple[Atoms, Atoms, np.ndarray, np.ndarray],
    ) -> None:
        mobile, reference, _rotation, _translation = rigid_pair

        aligned = align_atom_sets(
            mobile,
            reference,
            mobile_indices=[0, 1, 2],
            reference_indices=[0, 1, 2],
        )

        assert aligned.positions == pytest.approx(reference.positions, abs=1e-12)

    def test_accepts_different_corresponding_index_orders(
        self,
        rigid_pair: tuple[Atoms, Atoms, np.ndarray, np.ndarray],
    ) -> None:
        mobile, reference, _rotation, _translation = rigid_pair
        reordered_reference = reference[[3, 1, 0, 2]]

        aligned = align_atom_sets(
            mobile,
            reordered_reference,
            mobile_indices=[0, 1, 2, 3],
            reference_indices=[2, 1, 3, 0],
        )

        assert aligned.positions == pytest.approx(reference.positions, abs=1e-12)

    def test_preserves_all_internal_distances(
        self,
        rigid_pair: tuple[Atoms, Atoms, np.ndarray, np.ndarray],
    ) -> None:
        mobile, reference, _rotation, _translation = rigid_pair

        aligned = align_atom_sets(mobile, reference, weights="masses")

        assert aligned.get_all_distances() == pytest.approx(
            mobile.get_all_distances(), abs=1e-12
        )

    def test_constraints_do_not_block_the_coordinate_frame_change(
        self,
        rigid_pair: tuple[Atoms, Atoms, np.ndarray, np.ndarray],
    ) -> None:
        mobile, reference, _rotation, _translation = rigid_pair
        mobile.set_constraint(FixAtoms(indices=[0]))

        aligned = align_atom_sets(mobile, reference)

        assert aligned.positions == pytest.approx(reference.positions, abs=1e-12)
        assert len(aligned.constraints) == 1

    def test_rejects_selections_of_different_lengths(
        self,
        rigid_pair: tuple[Atoms, Atoms, np.ndarray, np.ndarray],
    ) -> None:
        mobile, reference, _rotation, _translation = rigid_pair

        with pytest.raises(ValueError, match="same number"):
            align_atom_sets(
                mobile,
                reference,
                mobile_indices=[0, 1],
                reference_indices=[0],
            )

    @pytest.mark.parametrize(
        ("indices", "exception", "message"),
        [
            ([], ValueError, "must not be empty"),
            ([0, 0], ValueError, "repeated"),
            ([True], TypeError, "integer"),
            ([0.5], TypeError, "integer"),
            ([99], IndexError, "out of range"),
        ],
    )
    def test_rejects_invalid_indices(
        self,
        rigid_pair: tuple[Atoms, Atoms, np.ndarray, np.ndarray],
        indices: list[Any],
        exception: type[Exception],
        message: str,
    ) -> None:
        mobile, reference, _rotation, _translation = rigid_pair

        with pytest.raises(exception, match=message):
            align_atom_sets(
                mobile,
                reference,
                mobile_indices=indices,
                reference_indices=list(range(len(indices))),
            )

    def test_rejects_empty_atom_sets(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            align_atom_sets(Atoms(), Atoms())

    def test_requires_ase_atoms(
        self,
        rigid_pair: tuple[Atoms, Atoms, np.ndarray, np.ndarray],
    ) -> None:
        _mobile, reference, _rotation, _translation = rigid_pair

        with pytest.raises(TypeError, match="ase.Atoms"):
            align_atom_sets(np.zeros((4, 3)), reference)

    def test_rejects_an_unknown_weight_mode(
        self,
        rigid_pair: tuple[Atoms, Atoms, np.ndarray, np.ndarray],
    ) -> None:
        mobile, reference, _rotation, _translation = rigid_pair

        with pytest.raises(ValueError, match="masses"):
            align_atom_sets(mobile, reference, weights="heavy")


class TestAtomSetRmsd:
    def test_reports_displacement_without_alignment(
        self,
        rigid_pair: tuple[Atoms, Atoms, np.ndarray, np.ndarray],
    ) -> None:
        mobile, reference, _rotation, _translation = rigid_pair
        expected = np.sqrt(np.mean(np.sum((mobile.positions - reference.positions) ** 2, axis=1)))

        result = atom_set_rmsd(mobile, reference)

        assert result == pytest.approx(expected)

    def test_removes_rigid_motion_when_asked(
        self,
        rigid_pair: tuple[Atoms, Atoms, np.ndarray, np.ndarray],
    ) -> None:
        mobile, reference, _rotation, _translation = rigid_pair

        result = atom_set_rmsd(mobile, reference, align=True)

        assert result == pytest.approx(0.0, abs=1e-12)

    def test_calculates_a_weighted_rmsd(self) -> None:
        mobile = Atoms("HH", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        reference = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        result = atom_set_rmsd(mobile, reference, weights=[3.0, 1.0])

        assert result == pytest.approx(1.0)

    def test_supports_mass_weighting(self) -> None:
        mobile = Atoms("HO", positions=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        reference = Atoms("HO", positions=np.zeros((2, 3)))
        masses = mobile.get_masses()
        expected = np.sqrt((masses[0] + 4.0 * masses[1]) / masses.sum())

        result = atom_set_rmsd(mobile, reference, weights="masses")

        assert result == pytest.approx(expected)

    def test_uses_only_the_selected_correspondences(self) -> None:
        mobile = Atoms("HHH", positions=[[0, 0, 0], [2, 0, 0], [100, 0, 0]])
        reference = Atoms("HHH", positions=[[0, 0, 0], [1, 0, 0], [-100, 0, 0]])

        result = atom_set_rmsd(
            mobile,
            reference,
            mobile_indices=[0, 1],
            reference_indices=[0, 1],
        )

        assert result == pytest.approx(np.sqrt(0.5))


class TestBondedClusterIndices:
    def test_finds_one_half_of_a_dimer(self, dimer: Atoms) -> None:
        """Anchored on the lower oxygen, the walk must not reach the upper."""
        cluster = bonded_cluster_indices_no_anchor_hub(dimer, 0)

        assert set(cluster) <= {0, 1, 2}

    def test_includes_the_anchor(self, dimer: Atoms) -> None:
        assert 0 in bonded_cluster_indices_no_anchor_hub(dimer, 0)

    def test_returns_sorted_indices(self, dimer: Atoms) -> None:
        cluster = bonded_cluster_indices_no_anchor_hub(dimer, 3)

        assert cluster == sorted(cluster)

    def test_rejects_an_out_of_range_anchor(self, dimer: Atoms) -> None:
        with pytest.raises(IndexError):
            bonded_cluster_indices_no_anchor_hub(dimer, len(dimer))

    def test_rejects_a_negative_anchor(self, dimer: Atoms) -> None:
        with pytest.raises(IndexError):
            bonded_cluster_indices_no_anchor_hub(dimer, -1)

    def test_a_larger_cutoff_finds_at_least_as_much(self, dimer: Atoms) -> None:
        tight = bonded_cluster_indices_no_anchor_hub(dimer, 0, mult=1.0)
        loose = bonded_cluster_indices_no_anchor_hub(dimer, 0, mult=2.0)

        assert set(tight) <= set(loose)


class TestGetDimerBondedClusterIndices:
    def test_merges_both_halves(self, dimer: Atoms) -> None:
        both = get_dimer_bonded_cluster_indices(dimer, [0, 3])

        assert set(both) == (
            set(bonded_cluster_indices_no_anchor_hub(dimer, 0))
            | set(bonded_cluster_indices_no_anchor_hub(dimer, 3))
        )

    def test_has_no_duplicates(self, dimer: Atoms) -> None:
        both = get_dimer_bonded_cluster_indices(dimer, [0, 3])

        assert len(both) == len(set(both))

    def test_rejects_the_wrong_number_of_anchors(self, dimer: Atoms) -> None:
        with pytest.raises(ValueError, match="exactly two indices"):
            get_dimer_bonded_cluster_indices(dimer, [0])

    def test_rejects_the_wrong_number_of_mults(self, dimer: Atoms) -> None:
        with pytest.raises(ValueError, match="exactly two values"):
            get_dimer_bonded_cluster_indices(dimer, [0, 3], mults=[1.0])


class TestPcaFrame:
    def test_origin_is_the_centroid(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

        origin, _R = _pca_frame(pts)

        assert origin == pytest.approx(pts.mean(axis=0))

    def test_frame_is_orthonormal(self) -> None:
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(8, 3))

        _origin, R = _pca_frame(pts)

        assert R.T @ R == pytest.approx(np.eye(3), abs=1e-10)

    def test_frame_is_right_handed(self) -> None:
        rng = np.random.default_rng(1)

        _origin, R = _pca_frame(rng.normal(size=(8, 3)))

        assert np.linalg.det(R) == pytest.approx(1.0)

    def test_normal_is_perpendicular_to_a_planar_group(self) -> None:
        """For points in the xy-plane the smallest-variance axis is z."""
        pts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )

        _origin, R = _pca_frame(pts)

        assert abs(abs(R[:, 2] @ np.array([0.0, 0.0, 1.0])) - 1.0) < 1e-10


class TestOrientNormalToward:
    def test_flips_a_normal_pointing_away(self) -> None:
        R = np.eye(3)  # normal is +z

        flipped = _orient_normal_toward(R, [0, 0, 0], [0, 0, -1])

        assert flipped[:, 2] == pytest.approx([0.0, 0.0, -1.0])

    def test_leaves_a_normal_already_facing_the_target(self) -> None:
        R = np.eye(3)

        kept = _orient_normal_toward(R, [0, 0, 0], [0, 0, 1])

        assert kept == pytest.approx(R)

    def test_stays_right_handed_after_flipping(self) -> None:
        flipped = _orient_normal_toward(np.eye(3), [0, 0, 0], [0, 0, -1])

        assert np.linalg.det(flipped) == pytest.approx(1.0)


class TestRigidTransform:
    def test_moves_the_anchor_onto_its_target(self) -> None:
        pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        anchor = np.array([1.0, 0.0, 0.0])

        moved = _rigid_transform(pts, anchor, np.eye(3), [5.0, 5.0, 5.0])

        assert moved[0] == pytest.approx([5.0, 5.0, 5.0])

    def test_preserves_internal_distances(self) -> None:
        rng = np.random.default_rng(2)
        pts = rng.normal(size=(5, 3))
        _origin, R = _pca_frame(rng.normal(size=(6, 3)))

        moved = _rigid_transform(pts, pts[0], R, [1.0, 2.0, 3.0])

        before = np.linalg.norm(pts[1:] - pts[0], axis=1)
        after = np.linalg.norm(moved[1:] - moved[0], axis=1)
        assert after == pytest.approx(before)


class TestFlipAndFaceBases:
    def test_swaps_the_two_halves_over(self, dimer: Atoms) -> None:
        """Each fragment should land on the other's anchor."""
        base_a, base_b = [0, 1, 2], [3, 4, 5]
        anchor_a, anchor_b = dimer.positions[0].copy(), dimer.positions[3].copy()

        swapped = flip_and_face_bases(dimer, base_a, base_b, [0, 3])

        assert swapped.positions[0] == pytest.approx(anchor_b)
        assert swapped.positions[3] == pytest.approx(anchor_a)

    def test_does_not_modify_the_input(self, dimer: Atoms) -> None:
        before = dimer.positions.copy()

        flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5], [0, 3])

        assert dimer.positions == pytest.approx(before)

    def test_moves_each_fragment_rigidly(self, dimer: Atoms) -> None:
        """A swap is a rigid motion, so internal distances must survive."""
        swapped = flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5], [0, 3])

        for frag in ([0, 1, 2], [3, 4, 5]):
            before = dimer[frag].get_all_distances()
            after = swapped[frag].get_all_distances()
            assert after == pytest.approx(before, abs=1e-9)

    def test_leaves_atoms_outside_both_fragments_alone(self, dimer: Atoms) -> None:
        spectator = dimer + Atoms("He", positions=[[8.0, 8.0, 8.0]])

        swapped = flip_and_face_bases(spectator, [0, 1, 2], [3, 4, 5], [0, 3])

        assert swapped.positions[6] == pytest.approx([8.0, 8.0, 8.0])

    def test_a_different_reflection_gives_a_different_structure(
        self,
        dimer: Atoms,
    ) -> None:
        default = flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5], [0, 3])
        other = flip_and_face_bases(
            dimer, [0, 1, 2], [3, 4, 5], [0, 3], rot_matrix=[1.0, -1.0, -1.0]
        )

        assert default.positions != pytest.approx(other.positions)


class TestOptimizeWithFixedAnchors:
    def test_actually_moves_the_fragment_atoms(self, calc: EMT, dimer: Atoms) -> None:
        """Regression: the result was written to a throwaway copy.

        `atoms_out[selection].set_positions(...)` builds a new Atoms object
        and updates that, so the relaxed coordinates were discarded and the
        function returned its input untouched.
        """
        strained = flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5], [0, 3])

        relaxed = optimize_with_fixed_anchors(
            strained, [0, 1, 2], [3, 4, 5], [0, 3], calc, fmax=0.5
        )

        assert relaxed.positions != pytest.approx(strained.positions)

    def test_does_not_modify_the_input(self, calc: EMT, dimer: Atoms) -> None:
        before = dimer.positions.copy()

        optimize_with_fixed_anchors(dimer, [0, 1, 2], [3, 4, 5], [0, 3], calc, fmax=0.5)

        assert dimer.positions == pytest.approx(before)

    def test_keeps_every_atom(self, calc: EMT, dimer: Atoms) -> None:
        relaxed = optimize_with_fixed_anchors(
            dimer, [0, 1, 2], [3, 4, 5], [0, 3], calc, fmax=0.5
        )

        assert len(relaxed) == len(dimer)
        assert relaxed.get_chemical_symbols() == dimer.get_chemical_symbols()

    def test_leaves_atoms_outside_both_fragments_alone(
        self,
        calc: EMT,
        dimer: Atoms,
    ) -> None:
        spectator = dimer + Atoms("He", positions=[[8.0, 8.0, 8.0]])

        relaxed = optimize_with_fixed_anchors(
            spectator, [0, 1, 2], [3, 4, 5], [0, 3], calc, fmax=0.5
        )

        assert relaxed.positions[6] == pytest.approx([8.0, 8.0, 8.0])

    def test_records_convergence_on_the_result(self, calc: EMT, dimer: Atoms) -> None:
        relaxed = optimize_with_fixed_anchors(
            dimer, [0, 1, 2], [3, 4, 5], [0, 3], calc, fmax=0.5
        )

        assert relaxed.info["converged"] is True

    def test_uses_the_optimiser_it_is_given(
        self,
        calc: EMT,
        dimer: Atoms,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        optimize_with_fixed_anchors(
            dimer, [0, 1, 2], [3, 4, 5], [0, 3], calc, fmax=0.5, optimiser=FIRE
        )

        log = capsys.readouterr().out
        assert "FIRE" in log
        assert "BFGS" not in log

    def test_no_logfile_silences_it(
        self,
        calc: EMT,
        dimer: Atoms,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        optimize_with_fixed_anchors(
            dimer, [0, 1, 2], [3, 4, 5], [0, 3], calc, fmax=0.5, logfile=None
        )

        assert capsys.readouterr().out == ""

    def test_warns_and_records_when_it_runs_out_of_steps(
        self,
        calc: EMT,
        dimer: Atoms,
    ) -> None:
        strained = flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5], [0, 3])

        with pytest.warns(ConvergenceWarning, match="Fixed-anchor relaxation"):
            relaxed = optimize_with_fixed_anchors(
                strained, [0, 1, 2], [3, 4, 5], [0, 3], calc, fmax=1e-3, steps=2
            )

        assert relaxed.info["converged"] is False

    def test_raises_instead_when_asked(self, calc: EMT, dimer: Atoms) -> None:
        strained = flip_and_face_bases(dimer, [0, 1, 2], [3, 4, 5], [0, 3])

        with pytest.raises(ConvergenceError):
            optimize_with_fixed_anchors(
                strained,
                [0, 1, 2],
                [3, 4, 5],
                [0, 3],
                calc,
                fmax=1e-3,
                steps=2,
                raise_on_unconverged=True,
            )


class TestGetBestFlipAndFaceBases:
    def test_returns_a_structure_without_optimising(self, dimer: Atoms) -> None:
        swapped = get_best_flip_and_face_bases(
            dimer, [0, 1, 2], [3, 4, 5], [0, 3], optimise_after=False
        )

        assert len(swapped) == len(dimer)

    def test_picks_a_reflection_at_least_as_good_as_the_default(
        self,
        dimer: Atoms,
    ) -> None:
        """The search exists to beat the hard-coded default sign choice."""
        base_a, base_b = [0, 1, 2], [3, 4, 5]

        best = get_best_flip_and_face_bases(
            dimer, base_a, base_b, [0, 3], optimise_after=False
        )
        default = flip_and_face_bases(dimer, base_a, base_b, [0, 3])

        def separation(atoms: Atoms) -> np.floating:
            return np.linalg.norm(
                atoms[base_a].get_center_of_mass() - atoms[base_b].get_center_of_mass()
            )

        assert separation(best) <= separation(default) + 1e-9

    def test_requires_a_calculator_when_optimising(self, dimer: Atoms) -> None:
        with pytest.raises(ValueError, match="needs a calculator"):
            get_best_flip_and_face_bases(
                dimer, [0, 1, 2], [3, 4, 5], [0, 3], optimise_after=True
            )

    def test_optimises_when_given_a_calculator(self, calc: EMT, dimer: Atoms) -> None:
        relaxed = get_best_flip_and_face_bases(
            dimer, [0, 1, 2], [3, 4, 5], [0, 3], optimise_after=True, calc=calc
        )

        assert len(relaxed) == len(dimer)

    def test_does_not_modify_the_input(self, dimer: Atoms) -> None:
        before = dimer.positions.copy()

        get_best_flip_and_face_bases(
            dimer, [0, 1, 2], [3, 4, 5], [0, 3], optimise_after=False
        )

        assert dimer.positions == pytest.approx(before)


class TestSwapBondingConfiguration:
    """O-H...O becoming O...H-O, the product end state of a proton transfer."""

    @pytest.fixture
    def h_bond(self) -> Atoms:
        """A collinear O-H...O along x, with O-H = 1.0 and O...O = 2.8."""
        return Atoms(
            "OHO", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.8, 0.0, 0.0]]
        )

    def test_hydrogen_moves_to_the_acceptor(self, h_bond: Atoms) -> None:
        swapped = swap_bonding_configuration(h_bond, 0, 1, 2)

        # Was 1.0 from the donor; now 1.0 from the acceptor instead.
        assert np.linalg.norm(
            swapped.positions[1] - swapped.positions[2]
        ) == pytest.approx(1.0)

    def test_the_new_bond_length_matches_the_old_one(self, h_bond: Atoms) -> None:
        h_bond.positions[1] = [0.7, 0.0, 0.0]

        swapped = swap_bonding_configuration(h_bond, 0, 1, 2)

        assert np.linalg.norm(
            swapped.positions[1] - swapped.positions[2]
        ) == pytest.approx(0.7)

    def test_the_hydrogen_stays_between_the_heavy_atoms(self, h_bond: Atoms) -> None:
        swapped = swap_bonding_configuration(h_bond, 0, 1, 2)

        assert swapped.positions[1][0] == pytest.approx(1.8)

    def test_the_heavy_atoms_do_not_move(self, h_bond: Atoms) -> None:
        swapped = swap_bonding_configuration(h_bond, 0, 1, 2)

        assert swapped.positions[0] == pytest.approx(h_bond.positions[0])
        assert swapped.positions[2] == pytest.approx(h_bond.positions[2])

    def test_does_not_modify_the_input(self, h_bond: Atoms) -> None:
        before = h_bond.positions.copy()

        swap_bonding_configuration(h_bond, 0, 1, 2)

        assert h_bond.positions == pytest.approx(before)

    def test_works_off_axis(self, h_bond: Atoms) -> None:
        """The donor->acceptor direction is what matters, not the frame."""
        h_bond.rotate(37, "z")
        h_bond.rotate(-19, "y")

        swapped = swap_bonding_configuration(h_bond, 0, 1, 2)

        assert np.linalg.norm(
            swapped.positions[1] - swapped.positions[2]
        ) == pytest.approx(1.0)

    def test_moves_multiple_protons_in_one_call(self) -> None:
        double_h_bond = Atoms(
            "OHOOHO",
            positions=[
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [2.8, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 5.1, 0.0],
                [0.0, 7.0, 0.0],
            ],
        )

        swapped = swap_bonding_configuration(
            double_h_bond, [0, 3], [1, 4], [2, 5]
        )

        assert swapped.get_distance(1, 2) == pytest.approx(0.8)
        assert swapped.get_distance(4, 5) == pytest.approx(1.1)
        assert swapped.positions[1] == pytest.approx([2.0, 0.0, 0.0])
        assert swapped.positions[4] == pytest.approx([0.0, 5.9, 0.0])

    def test_a_scalar_donor_is_shared_by_multiple_protons(self) -> None:
        branched_h_bonds = Atoms(
            "OHOHO",
            positions=[
                [0.0, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 1.1, 0.0],
                [0.0, 3.0, 0.0],
            ],
        )

        swapped = swap_bonding_configuration(
            branched_h_bonds, 0, np.array([1, 3]), (2, 4)
        )

        assert swapped.get_distance(1, 2) == pytest.approx(0.9)
        assert swapped.get_distance(3, 4) == pytest.approx(1.1)

    def test_multiple_protons_do_not_modify_the_input(self) -> None:
        double_h_bond = Atoms(
            "OHOOHO",
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.8, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 6.8, 0.0],
            ],
        )
        before = double_h_bond.positions.copy()

        swap_bonding_configuration(double_h_bond, [0, 3], [1, 4], [2, 5])

        assert double_h_bond.positions == pytest.approx(before)

    def test_rejects_a_donor_count_that_does_not_match_the_protons(self) -> None:
        atoms = Atoms("OHOHOHO", positions=np.zeros((7, 3)))

        with pytest.raises(ValueError, match="one index per hydrogen"):
            swap_bonding_configuration(atoms, [0, 2], [1, 3, 5], 6)

    def test_rejects_a_repeated_hydrogen(self, h_bond: Atoms) -> None:
        with pytest.raises(ValueError, match="repeated"):
            swap_bonding_configuration(h_bond, 0, [1, 1], 2)

    @pytest.mark.parametrize("invalid", [1.0, None, [1.0], [True]])
    def test_rejects_non_integer_hydrogen_indices(
        self,
        h_bond: Atoms,
        invalid: Any,
    ) -> None:
        with pytest.raises(TypeError, match="integer"):
            swap_bonding_configuration(h_bond, 0, invalid, 2)

    def test_rejects_an_empty_hydrogen_list(self, h_bond: Atoms) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            swap_bonding_configuration(h_bond, 0, [], 2)

    def test_rejects_an_out_of_range_index(self, h_bond: Atoms) -> None:
        with pytest.raises(IndexError, match="out of range"):
            swap_bonding_configuration(h_bond, 0, 1, len(h_bond))

    def test_rejects_reusing_an_atom_within_a_transfer(self, h_bond: Atoms) -> None:
        with pytest.raises(ValueError, match="distinct"):
            swap_bonding_configuration(h_bond, 0, 1, 0)

    def test_rejects_a_non_hydrogen_index(self, h_bond: Atoms) -> None:
        with pytest.raises(ValueError, match="not H"):
            swap_bonding_configuration(h_bond, 1, 0, 2)

    def test_rejects_coincident_donor_and_acceptor_positions(
        self,
        h_bond: Atoms,
    ) -> None:
        h_bond.positions[2] = h_bond.positions[0]

        with pytest.raises(ValueError, match="positions must be different"):
            swap_bonding_configuration(h_bond, 0, 1, 2)


@pytest.fixture
def transfer() -> tuple[Atoms, Atoms]:
    """A proton transfer and a transition state for it.

    The reactant is the hydrogen-bonded triad of the `pt_atoms` fixture, with
    the proton on the donor oxygen; the transition state has it midway across.
    Returned as a pair, since seeding needs both.
    """
    reactant = Atoms(
        "OHOCCC",
        positions=[
            [0.00, 0.00, 0.00],
            [0.98, 0.00, 0.00],
            [2.65, 0.00, 0.00],
            [-0.65, 1.18, 0.00],
            [0.10, 2.40, 0.00],
            [1.55, 2.35, 0.00],
        ],
    )
    ts = reactant.copy()
    ts.positions[1] = [1.325, 0.00, 0.00]
    return reactant, ts


@pytest.fixture
def contracting() -> tuple[Atoms, Atoms]:
    """A pair whose path pulls the two oxygens together as it goes.

    Extrapolating it keeps contracting them, so the push runs into the clash
    check rather than into anything chemical.
    """
    reactant = Atoms("OHO", positions=[[0.0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]])
    ts = Atoms("OHO", positions=[[0.2, 0, 0], [1.5, 0, 0], [2.8, 0, 0]])
    return reactant, ts


class TestSeedProductFromTs:
    def test_steps_past_the_transition_state(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        reactant, ts = transfer

        seed = seed_product_from_ts(reactant, ts)

        assert atom_set_rmsd(seed, reactant, align=True) > atom_set_rmsd(
            ts, reactant, align=True
        )

    def test_carries_the_proton_towards_the_acceptor(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        """The point of the whole thing, on the reaction it was written for."""
        reactant, ts = transfer

        seed = seed_product_from_ts(reactant, ts)

        assert seed.get_distance(1, 2) < ts.get_distance(1, 2)
        assert seed.get_distance(0, 1) > ts.get_distance(0, 1)

    def test_a_bigger_push_crosses_further(self, transfer: tuple[Atoms, Atoms]) -> None:
        reactant, ts = transfer

        near = seed_product_from_ts(reactant, ts, push=1.0)
        far = seed_product_from_ts(reactant, ts, push=2.5)

        assert far.get_distance(1, 2) < near.get_distance(1, 2)

    def test_the_push_scales_the_distance_travelled(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        reactant, ts = transfer

        single = seed_product_from_ts(reactant, ts, push=1.0)
        double = seed_product_from_ts(reactant, ts, push=2.0)

        assert double.info["seed_push"] == pytest.approx(
            2 * single.info["seed_push"]
        )
        assert atom_set_rmsd(double, ts) == pytest.approx(
            2 * atom_set_rmsd(single, ts)
        )

    def test_seeds_the_reactant_when_the_ends_are_swapped(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        """Nothing about it is specific to products: passing the product seeds
        the reactant, off the other side of the same saddle."""
        reactant, ts = transfer
        product = reactant.copy()
        product.positions[1] = [1.67, 0.00, 0.00]  # the proton on the acceptor

        back = seed_product_from_ts(product, ts)

        assert back.get_distance(0, 1) < ts.get_distance(0, 1)

    def test_does_not_modify_its_inputs(self, transfer: tuple[Atoms, Atoms]) -> None:
        reactant, ts = transfer
        before = reactant.positions.copy(), ts.positions.copy()

        seed_product_from_ts(reactant, ts)

        assert np.allclose(reactant.positions, before[0])
        assert np.allclose(ts.positions, before[1])

    def test_keeps_the_atoms_the_cell_and_the_boundary_conditions(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        reactant, ts = transfer
        for atoms in (reactant, ts):
            atoms.set_cell([12.0, 12.0, 12.0])
            atoms.pbc = True

        seed = seed_product_from_ts(reactant, ts)

        assert seed.get_chemical_symbols() == ts.get_chemical_symbols()
        assert np.allclose(seed.cell, ts.cell)
        assert all(seed.pbc)

    def test_holds_a_constrained_atom_still(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        reactant, ts = transfer
        ts.set_constraint(FixAtoms(indices=[0]))

        seed = seed_product_from_ts(reactant, ts)

        assert np.allclose(seed.positions[0], ts.positions[0])
        assert seed.constraints

    def test_does_not_carry_over_the_transition_state_convergence(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        """Whether the saddle converged says nothing about a structure off it."""
        reactant, ts = transfer
        ts.info["converged"] = True

        seed = seed_product_from_ts(reactant, ts)

        assert "converged" not in seed.info

    def test_records_how_far_it_went(self, transfer: tuple[Atoms, Atoms]) -> None:
        reactant, ts = transfer

        seed = seed_product_from_ts(reactant, ts)

        assert seed.info["seeded"] is True
        assert seed.info["seed_push"] > 0
        assert seed.info["seed_alignment"] > 0.5
        assert seed.info["seed_rmsd_reactant"] > seed.info["seed_rmsd_ts"]

    def test_returns_the_whole_band_when_asked(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        reactant, ts = transfer

        seed, path = seed_product_from_ts(
            reactant, ts, n_images=8, n_steps=4, return_path=True
        )

        assert len(path) == 12
        assert path[-1] is seed

    def test_the_band_passes_through_the_transition_state_as_given(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        """The whole path is put back in the frame the caller works in."""
        reactant, ts = transfer

        _seed, path = seed_product_from_ts(reactant, ts, n_images=8, return_path=True)

        assert atom_set_rmsd(path[7], ts) == pytest.approx(0.0, abs=1e-8)

    def test_the_weighting_changes_the_direction(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        """Weighting every atom equally spreads a proton's motion over the rest."""
        reactant, ts = transfer

        weighted = seed_product_from_ts(reactant, ts)
        uniform = seed_product_from_ts(reactant, ts, weights=None)

        assert not np.allclose(weighted.positions, uniform.positions)

    def test_stops_short_of_a_clash_and_says_so(
        self,
        contracting: tuple[Atoms, Atoms],
    ) -> None:
        reactant, ts = contracting

        with pytest.warns(SeedWarning, match="Seeding stopped after"):
            seed = seed_product_from_ts(reactant, ts, push=8.0, n_steps=20)

        assert seed.info["seed_push"] > 0  # it got some of the way
        assert seed.get_distance(0, 1) > 0.7 * (0.66 + 0.31)  # and no further

    def test_the_clash_check_can_be_turned_off(
        self,
        contracting: tuple[Atoms, Atoms],
    ) -> None:
        reactant, ts = contracting

        with pytest.warns(SeedWarning, match="Seeding stopped after"):
            stopped = seed_product_from_ts(reactant, ts, push=8.0, n_steps=20)
        through = seed_product_from_ts(
            reactant, ts, push=8.0, n_steps=20, clash_scale=None
        )

        assert through.info["seed_push"] > stopped.info["seed_push"]
        assert through.get_distance(0, 2) < stopped.get_distance(0, 2)

    def test_warns_when_it_could_not_step_at_all(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        """A clash_scale nothing can satisfy leaves the seed on the saddle."""
        reactant, ts = transfer

        with pytest.warns(SeedWarning, match="Seeding stopped after 0"):
            with pytest.warns(SeedWarning, match="went nowhere useful"):
                seed = seed_product_from_ts(reactant, ts, clash_scale=2.0)

        assert seed.info["seeded"] is False
        assert seed.info["seed_push"] == 0.0

    def test_warns_when_the_two_structures_are_too_alike(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        """Interpolating between near-identical structures gives noise, not a
        direction."""
        reactant, _ts = transfer
        barely = reactant.copy()
        barely.positions[1] += [0.005, 0.0, 0.0]

        with pytest.warns(SeedWarning, match="aligned with the direction"):
            seed = seed_product_from_ts(reactant, barely)

        assert seed.info["seeded"] is False
        assert seed.info["seed_alignment"] < 0.5

    def test_a_warning_filter_can_promote_it_to_an_error(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        reactant, ts = transfer

        with warnings.catch_warnings():
            warnings.simplefilter("error", SeedWarning)

            with pytest.raises(SeedWarning):
                seed_product_from_ts(reactant, ts, clash_scale=2.0)

    def test_a_healthy_seed_warns_about_nothing(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        reactant, ts = transfer

        with warnings.catch_warnings():
            warnings.simplefilter("error", SeedWarning)

            seed_product_from_ts(reactant, ts)

    def test_rejects_structures_of_different_lengths(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        reactant, ts = transfer

        with pytest.raises(ValueError, match="same atoms"):
            seed_product_from_ts(reactant, ts[:-1])

    def test_rejects_structures_with_different_elements(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        reactant, ts = transfer
        ts.symbols[3] = "N"

        with pytest.raises(ValueError, match="same chemical symbols"):
            seed_product_from_ts(reactant, ts)

    def test_rejects_two_copies_of_the_same_structure(
        self,
        transfer: tuple[Atoms, Atoms],
    ) -> None:
        reactant, _ts = transfer

        with pytest.raises(ValueError, match="same structure"):
            seed_product_from_ts(reactant, reactant.copy())

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"n_images": 2}, "at least 3"),
            ({"tangent_images": 1}, "tangent_images must be"),
            ({"tangent_images": 99}, "tangent_images must be"),
            ({"push": 0.0}, "push must be positive"),
            ({"push": -1.0}, "push must be positive"),
            ({"n_steps": 0}, "n_steps must be at least 1"),
        ],
    )
    def test_rejects_arguments_out_of_range(
        self,
        transfer: tuple[Atoms, Atoms],
        kwargs: dict[str, float],
        message: str,
    ) -> None:
        reactant, ts = transfer

        with pytest.raises(ValueError, match=message):
            seed_product_from_ts(reactant, ts, **kwargs)

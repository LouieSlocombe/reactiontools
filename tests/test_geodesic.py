"""Tests for geodesic interpolation.

The alignment, coordinate and scaling helpers are checked against what they
claim to be -- a Kabsch fit, a Wilson B matrix, a differentiable scaling -- so
that a change to the numerics has to break an identity rather than merely a
recorded number. The path functions on top of them are checked for the
properties a reaction path has to have: the right number of images, unbroken
chemistry along it, and the same answer twice from the same seed.
"""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from reactiontools import (
    Geodesic,
    align_geom,
    align_path,
    align_path_to,
    compute_rij,
    compute_wij,
    elu_scaler,
    from_ase_atoms,
    geodesic_interpolate,
    get_bond_list,
    morse_scaler,
    read_xyz,
    redistribute,
    to_ase_atoms,
    tools_geodesic,
    write_xyz,
)


def _rotation(angle: float, axis: int = 2) -> np.ndarray:
    """A rotation matrix of ``angle`` radians about one Cartesian axis."""
    c, s = np.cos(angle), np.sin(angle)
    others = [i for i in range(3) if i != axis]
    matrix = np.eye(3)
    matrix[np.ix_(others, others)] = [[c, -s], [s, c]]
    return matrix


def _displaced(atoms: Atoms, index: int = 1, shift: float = 0.4) -> Atoms:
    """A copy of *atoms* with one atom pushed along x, as an end state."""
    other = atoms.copy()
    other.positions[index] += [shift, 0.0, 0.0]
    return other


def _distance_matrix(atoms: Atoms) -> np.ndarray:
    """All inter-atomic distances, which no rigid-body motion may change."""
    positions = atoms.get_positions()
    return np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)


class TestAlignGeom:
    def test_a_rigid_motion_is_undone_exactly(self) -> None:
        reference = molecule("CH3CH2OH").get_positions()
        moved = reference @ _rotation(0.7).T + [1.0, -2.0, 0.5]

        rmsd, aligned = align_geom(reference, moved)

        assert rmsd == pytest.approx(0.0, abs=1e-10)
        assert aligned == pytest.approx(reference, abs=1e-10)

    def test_the_fit_is_a_rotation_and_never_a_reflection(self) -> None:
        """A mirror image is not a rotation of the original, so it must not fit.

        Kabsch's unconstrained orthogonal fit would happily return the
        reflection and report an RMSD of zero, silently turning a molecule into
        its enantiomer part-way along a path.
        """
        reference = molecule("CH3CH2OH").get_positions()
        mirrored = reference * [-1.0, 1.0, 1.0]

        rmsd, _ = align_geom(reference, mirrored)

        assert rmsd > 0.1

    def test_the_reported_rmsd_is_the_one_the_fit_achieved(self) -> None:
        reference = molecule("H2O").get_positions()
        moved = _displaced(molecule("H2O")).get_positions()

        rmsd, aligned = align_geom(reference, moved)

        assert rmsd == pytest.approx(np.sqrt(np.mean((aligned - reference) ** 2)))


class TestAlignPath:
    def test_every_image_ends_up_centred_on_the_origin(self) -> None:
        water = molecule("H2O")
        raw = [water.get_positions() + [i, 2.0 * i, 0.0] for i in range(4)]

        _, aligned = align_path(raw)

        assert np.mean(aligned, axis=1) == pytest.approx(np.zeros((4, 3)), abs=1e-10)

    def test_the_input_is_left_untouched(self) -> None:
        raw = np.array([molecule("H2O").get_positions() + i for i in range(3)])
        before = raw.copy()

        align_path(raw)

        assert raw == pytest.approx(before)

    def test_the_max_rmsd_is_zero_for_a_path_that_does_not_move(self) -> None:
        still = [molecule("H2O").get_positions() for _ in range(3)]

        max_rmsd, _ = align_path(still)

        assert max_rmsd == pytest.approx(0.0, abs=1e-10)


class TestAlignPathTo:
    def test_the_first_image_lands_on_the_reference(self) -> None:
        reference = molecule("CH3CH2OH").get_positions()
        path = np.array([reference @ _rotation(1.1).T + 3.0 for _ in range(4)])

        moved = align_path_to(reference, path)

        assert moved[0] == pytest.approx(reference, abs=1e-10)

    def test_the_shape_of_the_path_survives_the_move(self) -> None:
        """One rigid motion for the whole path, so no image moves relative to
        any other."""
        reference = molecule("H2O").get_positions()
        path = np.array(
            [molecule("H2O").get_positions() + [0.0, 0.0, 0.1 * i] for i in range(5)]
        )

        moved = align_path_to(reference, path)

        before = np.diff(path, axis=0)
        after = np.diff(moved, axis=0)
        assert np.linalg.norm(after, axis=-1) == pytest.approx(
            np.linalg.norm(before, axis=-1), abs=1e-10
        )


class TestGetBondList:
    def test_a_single_geometry_is_promoted_to_a_path(self) -> None:
        water = molecule("H2O")

        pairs, _ = get_bond_list(water.get_positions(), water.get_chemical_symbols())

        assert pairs == [(0, 1), (0, 2), (1, 2)]

    def test_equilibrium_distances_are_summed_covalent_radii(self) -> None:
        water = molecule("H2O")

        pairs, re = get_bond_list(water.get_positions(), water.get_chemical_symbols())

        # O-H twice then H-H, in the sorted pair order above
        assert re[0] == pytest.approx(re[1])
        assert re[2] < re[0]

    def test_without_symbols_every_pair_gets_a_nominal_distance(self) -> None:
        water = molecule("H2O")

        pairs, re = get_bond_list(water.get_positions())

        assert re == pytest.approx(np.full(len(pairs), 2.0))

    def test_enforced_pairs_survive_however_far_apart_they_are(self) -> None:
        far = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])

        pairs, _ = get_bond_list(far, min_neighbors=0, enforce=[(0, 1)])

        assert pairs == [(0, 1)]

    def test_an_under_connected_atom_is_topped_up_to_min_neighbors(self) -> None:
        """An atom too far from everything still needs coordinates of its own,
        or the optimiser has nothing to hold it in place with."""
        ethanol = molecule("CH3CH2OH")
        positions = ethanol.get_positions()
        positions[-1] += [30.0, 0.0, 0.0]

        pairs, _ = get_bond_list(
            positions, ethanol.get_chemical_symbols(), min_neighbors=4
        )

        stranded = len(ethanol) - 1
        assert sum(stranded in pair for pair in pairs) >= 4

    def test_atoms_within_three_bonds_beat_the_distance_cut_off(self) -> None:
        """A chain of four atoms, each bonded to the next, stretched past the
        cut-off end to end."""
        chain = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.5, 0.0, 0.0]])

        pairs, _ = get_bond_list(chain, threshold=2.0, min_neighbors=0)

        assert (0, 3) in pairs


class TestComputeCoordinates:
    def test_distances_are_the_ones_the_geometry_has(self) -> None:
        ethanol = molecule("CH3CH2OH")
        pairs, _ = get_bond_list(
            ethanol.get_positions(), ethanol.get_chemical_symbols()
        )

        rij, _ = compute_rij(ethanol.get_positions(), pairs)

        expected = ethanol.get_all_distances()
        assert rij == pytest.approx([expected[i, j] for i, j in pairs])

    def test_the_b_matrix_matches_finite_differences(self) -> None:
        water = molecule("H2O")
        pairs, re = get_bond_list(water.get_positions(), water.get_chemical_symbols())
        scaler = morse_scaler(re=re)
        geom = water.get_positions()

        _, b_mat = compute_wij(geom, pairs, scaler)

        step = 1e-6
        numerical = np.zeros_like(b_mat)
        for k in range(geom.size):
            shift = np.zeros(geom.size)
            shift[k] = step
            up, _ = compute_wij(geom.ravel() + shift, pairs, scaler)
            down, _ = compute_wij(geom.ravel() - shift, pairs, scaler)
            numerical[:, k] = (up - down) / (2 * step)
        assert b_mat == pytest.approx(numerical, abs=1e-6)

    def test_the_sparse_and_dense_branches_agree(self) -> None:
        ethanol = molecule("CH3CH2OH")
        pairs, re = get_bond_list(
            ethanol.get_positions(), ethanol.get_chemical_symbols()
        )
        scaler = morse_scaler(re=re)

        dense_w, dense_b = compute_wij(ethanol.get_positions(), pairs, scaler)
        sparse_w, sparse_b = compute_wij(
            ethanol.get_positions(), pairs, scaler, sparse=True
        )

        assert sparse_w == pytest.approx(dense_w)
        assert sparse_b.toarray() == pytest.approx(dense_b)

    def test_a_flattened_geometry_is_accepted(self) -> None:
        water = molecule("H2O")
        pairs, re = get_bond_list(water.get_positions(), water.get_chemical_symbols())
        scaler = morse_scaler(re=re)

        shaped, _ = compute_wij(water.get_positions(), pairs, scaler)
        flat, _ = compute_wij(water.get_positions().ravel(), pairs, scaler)

        assert flat == pytest.approx(shaped)


class TestScalers:
    @pytest.mark.parametrize("build", [morse_scaler, elu_scaler])
    def test_the_derivative_matches_finite_differences(self, build) -> None:
        scaler = build(re=1.4)
        r = np.linspace(0.8, 6.0, 40)
        step = 1e-7

        _, derivative = scaler(r)

        up, _ = scaler(r + step)
        down, _ = scaler(r - step)
        assert derivative == pytest.approx((up - down) / (2 * step), rel=1e-5)

    @pytest.mark.parametrize("build", [morse_scaler, elu_scaler])
    def test_the_scaled_distance_falls_away_monotonically(self, build) -> None:
        """The metric has to shrink with distance, or far-apart atoms would
        pull on the path harder than bonded ones."""
        scaler = build(re=1.4)
        r = np.linspace(0.8, 8.0, 60)

        values, _ = scaler(r)

        assert np.all(np.diff(values) < 0)

    def test_the_elu_scaler_is_continuous_where_it_switches(self) -> None:
        scaler = elu_scaler(re=2.0)
        just_below, just_above = np.array([2.0 - 1e-9]), np.array([2.0 + 1e-9])

        below, _ = scaler(just_below)
        above, _ = scaler(just_above)

        assert below == pytest.approx(above, abs=1e-8)

    def test_alpha_sets_how_fast_the_morse_metric_decays(self) -> None:
        r = np.array([4.0])

        gentle, _ = morse_scaler(re=1.4, alpha=0.7)(r)
        sharp, _ = morse_scaler(re=1.4, alpha=3.0)(r)

        assert sharp < gentle


class TestGeodesic:
    @pytest.fixture
    def three_image_path(self) -> tuple[list[str], np.ndarray]:
        """A water molecule and a displaced copy, with a crude midpoint."""
        start = molecule("H2O")
        end = _displaced(start)
        middle = (start.get_positions() + end.get_positions()) / 2
        path = np.array([start.get_positions(), middle, end.get_positions()])
        return start.get_chemical_symbols(), path

    def test_a_two_dimensional_path_is_refused(self) -> None:
        water = molecule("H2O")

        with pytest.raises(ValueError, match="3 dimensions"):
            Geodesic(water.get_chemical_symbols(), water.get_positions())

    def test_smoothing_shortens_a_badly_placed_image(self) -> None:
        """Pushed off the path rather than sitting on it, so there is real
        length for the optimiser to take out.

        Not the fixture: its midpoint is the Cartesian average of two nearly
        identical geometries, which is already all but the geodesic, and the
        friction term can leave the length a part in a million longer.
        """
        start = molecule("H2O")
        end = _displaced(start)
        middle = (start.get_positions() + end.get_positions()) / 2
        middle[2] += [0.0, 0.0, 0.6]
        path = np.array([start.get_positions(), middle, end.get_positions()])
        smoother = Geodesic(start.get_chemical_symbols(), path)
        smoother.compute_displacements()
        before = smoother.length

        smoother.smooth(tol=1e-4, max_iter=50)
        smoother.compute_displacements()

        assert smoother.length < before

    def test_smoothing_leaves_the_end_points_where_they_were(
        self, three_image_path: tuple[list[str], np.ndarray]
    ) -> None:
        """Only the interior images are free; the two end states are the
        answer the path has to connect."""
        symbols, path = three_image_path
        smoother = Geodesic(symbols, path)
        _, aligned = align_path(path)

        smoothed = smoother.smooth(tol=1e-4, max_iter=50)

        assert align_geom(aligned[0], smoothed[0])[0] == pytest.approx(0.0, abs=1e-8)
        assert align_geom(aligned[-1], smoothed[-1])[0] == pytest.approx(0.0, abs=1e-8)

    def test_sweeping_reaches_the_same_place_as_smoothing(
        self, three_image_path: tuple[list[str], np.ndarray]
    ) -> None:
        symbols, path = three_image_path

        swept = Geodesic(symbols, path.copy()).sweep(tol=1e-5, max_iter=50)
        smoothed = Geodesic(symbols, path.copy()).smooth(tol=1e-5, max_iter=100)

        assert align_geom(smoothed[1], swept[1])[0] == pytest.approx(0.0, abs=1e-3)

    def test_an_explicit_scaler_is_used_as_given(
        self, three_image_path: tuple[list[str], np.ndarray]
    ) -> None:
        symbols, path = three_image_path

        smoother = Geodesic(symbols, path, scaler=elu_scaler(re=1.4))

        assert smoother.scaler is not None
        smoother.compute_displacements()
        assert np.isfinite(smoother.length)

    def test_a_section_outside_the_fixed_end_points_is_refused(
        self, three_image_path: tuple[list[str], np.ndarray]
    ) -> None:
        symbols, path = three_image_path
        smoother = Geodesic(symbols, path)

        with pytest.raises(ValueError, match="must lie between"):
            smoother.compute_displacements(start=0, end=2)

    def test_moving_an_image_to_where_it_already_is_changes_nothing(
        self, three_image_path: tuple[list[str], np.ndarray]
    ) -> None:
        """The optimiser asks for the same geometry twice in a row all the
        time; recomputing the coordinates for it would double the work."""
        symbols, path = three_image_path
        smoother = Geodesic(symbols, path)
        smoother.update_intc()

        moved = smoother.update_geometry(smoother.path[1:2].copy(), 1, 2)

        assert moved is False
        assert smoother.w[1] is not None

    def test_moving_an_image_invalidates_the_midpoints_either_side(
        self, three_image_path: tuple[list[str], np.ndarray]
    ) -> None:
        symbols, path = three_image_path
        smoother = Geodesic(symbols, path)
        smoother.update_intc()

        moved = smoother.update_geometry(smoother.path[1:2] + 0.1, 1, 2)

        assert moved is True
        assert smoother.w[1] is None
        assert smoother.w_mid == [None, None]

    def test_the_friction_term_is_dropped_when_there_is_no_reference(
        self, three_image_path: tuple[list[str], np.ndarray]
    ) -> None:
        """Nothing to pull back towards, so the target function is the path
        length alone."""
        symbols, path = three_image_path
        smoother = Geodesic(symbols, path)

        smoother.compute_target_func()

        assert smoother.displacements[-smoother.path[1:2].size :] == pytest.approx(0.0)

    def test_a_section_can_be_swept_with_an_explicit_end(
        self, three_image_path: tuple[list[str], np.ndarray]
    ) -> None:
        symbols, path = three_image_path
        smoother = Geodesic(symbols, path)

        swept = smoother.sweep(tol=1e-4, max_iter=5, start=1, end=2)

        assert swept.shape == path.shape

    def test_the_stored_path_and_the_returned_one_are_the_same(
        self, three_image_path: tuple[list[str], np.ndarray]
    ) -> None:
        symbols, path = three_image_path
        smoother = Geodesic(symbols, path)

        returned = smoother.smooth(tol=1e-4, max_iter=20)

        assert returned is smoother.path


class TestRedistribute:
    def test_images_are_added_until_there_are_enough(self) -> None:
        water = molecule("H2O")
        symbols, geoms = from_ase_atoms([water, _displaced(water)])

        path = redistribute(symbols, geoms, n_images=6, rng=np.random.default_rng(0))

        assert len(path) == 6

    def test_images_are_dropped_until_there_are_few_enough(self) -> None:
        water = molecule("H2O")
        end = _displaced(water)
        frames = [
            Atoms(
                symbols=water.get_chemical_symbols(),
                positions=water.get_positions()
                + t * (end.get_positions() - water.get_positions()),
            )
            for t in np.linspace(0.0, 1.0, 9)
        ]
        symbols, geoms = from_ase_atoms(frames)

        path = redistribute(symbols, geoms, n_images=4, rng=np.random.default_rng(0))

        assert len(path) == 4

    def test_the_bisection_works_without_a_seeded_generator(self) -> None:
        """``rng`` is optional, so the default has to be a usable source."""
        water = molecule("H2O")
        symbols, geoms = from_ase_atoms([water, _displaced(water)])

        path = redistribute(symbols, geoms, n_images=4)

        assert len(path) == 4

    def test_a_path_that_is_already_the_right_length_is_only_aligned(self) -> None:
        water = molecule("H2O")
        symbols, geoms = from_ase_atoms([water, _displaced(water)])

        path = redistribute(symbols, geoms, n_images=2, rng=np.random.default_rng(0))

        assert len(path) == 2
        assert np.mean(path[0], axis=0) == pytest.approx(np.zeros(3), abs=1e-10)


class TestAseConversion:
    def test_symbols_and_coordinates_survive_a_round_trip(self) -> None:
        frames = [molecule("H2O"), _displaced(molecule("H2O"))]

        symbols, coords = from_ase_atoms(frames)
        back = to_ase_atoms(symbols, coords)

        assert [a.get_chemical_symbols() for a in back] == [
            f.get_chemical_symbols() for f in frames
        ]
        assert back[1].get_positions() == pytest.approx(frames[1].get_positions())

    def test_a_single_frame_is_accepted(self) -> None:
        water = molecule("H2O")

        back = to_ase_atoms(water.get_chemical_symbols(), water.get_positions())

        assert len(back) == 1

    def test_a_template_carries_everything_the_path_does_not_touch(self) -> None:
        water = molecule("H2O")
        water.set_cell([10.0, 10.0, 10.0])
        water.set_pbc(True)
        water.set_tags([1, 2, 3])
        water.calc = EMT()

        back = to_ase_atoms(
            water.get_chemical_symbols(), water.get_positions(), template=water
        )

        assert back[0].cell.array == pytest.approx(water.cell.array)
        assert list(back[0].pbc) == [True, True, True]
        assert list(back[0].get_tags()) == [1, 2, 3]
        # As with Atoms.copy: the results belong to the geometry they were
        # computed for, not to the image that replaced it.
        assert back[0].calc is None

    def test_a_constraint_comes_along_without_moving_the_atoms(self) -> None:
        """Applying it would drag the interpolated image back onto the
        template and quietly corrupt the path."""
        water = molecule("H2O")
        water.set_constraint(FixAtoms(indices=[0]))
        moved = _displaced(water, index=0).get_positions()

        back = to_ase_atoms(water.get_chemical_symbols(), moved, template=water)

        assert back[0].get_positions() == pytest.approx(moved)
        assert len(back[0].constraints) == 1

    def test_a_template_for_a_different_molecule_is_refused(self) -> None:
        water = molecule("H2O")

        with pytest.raises(ValueError, match="not the same system"):
            to_ase_atoms(["C", "O"], np.zeros((2, 3)), template=water)


class TestXyzFiles:
    def test_coordinates_survive_a_round_trip(self, tmp_path: Path) -> None:
        water = molecule("H2O")
        symbols = water.get_chemical_symbols()
        frames = np.array([water.get_positions(), _displaced(water).get_positions()])
        path = tmp_path / "path.xyz"

        write_xyz(path, symbols, frames)
        back_symbols, back_frames = read_xyz(path)

        assert back_symbols == symbols
        assert np.array(back_frames) == pytest.approx(frames, abs=1e-6)

    def test_a_single_frame_is_accepted(self, tmp_path: Path) -> None:
        water = molecule("H2O")
        path = tmp_path / "one.xyz"

        write_xyz(path, water.get_chemical_symbols(), water.get_positions())

        _, frames = read_xyz(path)
        assert len(frames) == 1

    def test_blank_lines_between_frames_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "gappy.xyz"
        path.write_text("1\nfirst\nH 0.0 0.0 0.0\n\n1\nsecond\nH 1.0 0.0 0.0\n\n")

        symbols, frames = read_xyz(path)

        assert symbols == ["H"]
        assert len(frames) == 2

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.xyz"
        path.write_text("")

        with pytest.raises(ValueError, match="empty"):
            read_xyz(path)

    def test_a_file_that_is_not_xyz_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("this is not a structure file\n")

        with pytest.raises(ValueError, match="Incorrect XYZ file format"):
            read_xyz(path)

    def test_a_truncated_frame_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "short.xyz"
        path.write_text("3\ncomment\nO 0.0 0.0 0.0\n")

        with pytest.raises(ValueError, match="Incorrect XYZ file format"):
            read_xyz(path)


class TestGeodesicInterpolate:
    def test_the_path_has_the_images_that_were_asked_for(self, water: Atoms) -> None:
        path = geodesic_interpolate([water, _displaced(water)], n_images=7)

        assert len(path) == 7
        assert all(isinstance(image, Atoms) for image in path)

    def test_the_path_still_describes_the_same_molecule(self, water: Atoms) -> None:
        path = geodesic_interpolate([water, _displaced(water)], n_images=7)

        assert all(
            image.get_chemical_symbols() == water.get_chemical_symbols()
            for image in path
        )

    def test_no_bond_is_stretched_or_crushed_along_the_way(
        self, water: Atoms
    ) -> None:
        """The whole point of the internal-coordinate metric: interpolating in
        Cartesians instead is what lets atoms run through each other."""
        path = geodesic_interpolate([water, _displaced(water)], n_images=9)

        shortest = min(
            image.get_all_distances()[np.triu_indices(len(image), k=1)].min()
            for image in path
        )
        assert shortest > 0.5

    def test_the_end_points_are_the_ones_that_were_handed_in(
        self, water: Atoms
    ) -> None:
        """Up to the rigid motion the optimiser is free to apply, which is why
        this compares distances rather than coordinates."""
        product = _displaced(water)

        path = geodesic_interpolate([water, product], n_images=7)

        assert _distance_matrix(path[0]) == pytest.approx(
            _distance_matrix(water), abs=1e-6
        )
        assert _distance_matrix(path[-1]) == pytest.approx(
            _distance_matrix(product), abs=1e-6
        )

    def test_the_same_seed_gives_the_same_path_twice(self, water: Atoms) -> None:
        """The bisection is stochastic, so without the seed a workflow would
        not reproduce from one run to the next."""
        product = _displaced(water)

        first = geodesic_interpolate([water, product], n_images=9, seed=3)
        second = geodesic_interpolate([water, product], n_images=9, seed=3)

        assert np.array([i.get_positions() for i in first]) == pytest.approx(
            np.array([i.get_positions() for i in second])
        )

    def test_the_callers_own_random_state_is_left_alone(self, water: Atoms) -> None:
        np.random.seed(0)
        before = np.random.random()
        np.random.seed(0)

        geodesic_interpolate([water, _displaced(water)], n_images=5)

        assert np.random.random() == before

    def test_an_intermediate_geometry_is_used_when_one_is_given(
        self, water: Atoms
    ) -> None:
        product = _displaced(water)
        middle = _displaced(water, shift=0.2)

        path = geodesic_interpolate([water, middle, product], n_images=7)

        assert len(path) == 7

    def test_a_filename_is_interpolated_onto_disk(
        self, water: Atoms, tmp_path: Path
    ) -> None:
        source = tmp_path / "ends.xyz"
        write_xyz(
            source,
            water.get_chemical_symbols(),
            np.array([water.get_positions(), _displaced(water).get_positions()]),
        )
        output = tmp_path / "band.xyz"

        result = geodesic_interpolate(source, n_images=5, output=output)

        assert result is None
        _, frames = read_xyz(output)
        assert len(frames) == 5

    def test_anything_that_is_neither_is_refused(self) -> None:
        with pytest.raises(TypeError, match="ASE Atoms object or a filename"):
            geodesic_interpolate(42)

    def test_one_geometry_is_not_a_path(self, water: Atoms) -> None:
        with pytest.raises(ValueError, match="at least two"):
            geodesic_interpolate([water])

    def test_the_template_properties_reach_every_image(self, water: Atoms) -> None:
        water.set_tags([1, 2, 3])
        water.info["label"] = "reactant"

        path = geodesic_interpolate([water, _displaced(water)], n_images=5)

        assert all(list(image.get_tags()) == [1, 2, 3] for image in path)
        assert all(image.info["label"] == "reactant" for image in path)

    def test_a_periodic_path_comes_back_in_the_frame_it_went_in(
        self, water: Atoms
    ) -> None:
        """The optimisation centres and rotates the path into a frame of its
        own. A cell only says where the atoms are in the frame it came with, so
        leaving it there would put every atom in the wrong place."""
        water.set_cell([12.0, 12.0, 12.0])
        water.set_pbc(True)
        water.translate([4.0, 4.0, 4.0])
        product = _displaced(water)

        path = geodesic_interpolate([water, product], n_images=5)

        assert path[0].get_positions() == pytest.approx(
            water.get_positions(), abs=1e-6
        )
        assert all(image.pbc.all() for image in path)

    def test_an_isolated_molecule_is_not_moved_back(self, water: Atoms) -> None:
        """Nothing outside the molecule to be in the wrong place relative to,
        so the optimiser's own centred frame is kept."""
        water.translate([5.0, 5.0, 5.0])

        path = geodesic_interpolate([water, _displaced(water)], n_images=5)

        assert np.mean(path[0].get_positions(), axis=0) == pytest.approx(
            np.zeros(3), abs=1e-8
        )

    def test_large_systems_are_swept_one_image_at_a_time(
        self, water: Atoms, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sweep is what keeps big systems tractable, and it is chosen by
        atom count alone -- so lowering the threshold exercises it on a
        molecule small enough for a test."""
        monkeypatch.setattr(tools_geodesic, "SWEEP_ABOVE_N_ATOMS", 1)
        called: list[str] = []
        original = tools_geodesic.Geodesic.sweep

        def spy(self, *args, **kwargs):
            called.append("sweep")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(tools_geodesic.Geodesic, "sweep", spy)

        path = geodesic_interpolate([water, _displaced(water)], n_images=5)

        assert called == ["sweep"]
        assert len(path) == 5

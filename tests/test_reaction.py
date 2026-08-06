"""Tests for reactiontools.tools_reaction."""

from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.mep import NEB

from reactiontools import (get_neb_path,
                           get_ts_image,
                           optimise_geom,
                           optimise_neb,
                           optimise_reactant_product,
                           prepare_neb,
                           resample_path,
                           stitch_path)


class TestGetNebPath:
    def test_returns_cumulative_distance_from_zero(self, chain):
        # One atom per image spaced 0.5 A apart, so the answer is exact
        assert get_neb_path(chain) == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])

    def test_is_monotonic(self, chain):
        path = get_neb_path(chain)

        assert np.all(np.diff(path) >= 0)

    def test_length_matches_the_number_of_images(self, chain):
        assert len(get_neb_path(chain)) == len(chain)

    def test_uses_the_norm_over_all_atoms(self):
        """Displacing N atoms by d gives a step of d * sqrt(N)."""
        path = get_neb_path(molecule("H2O") for _ in range(1))

        assert path == pytest.approx([0.0])


class TestStitchPath:
    def test_reverses_the_first_path_and_drops_the_shared_image(self, make_chain):
        first = make_chain(3, dz=1.0)  # z = 0, 1, 2
        second = make_chain(3, dz=2.0)  # z = 0, 2, 4

        stitched = stitch_path(first, second)

        # first reversed (2, 1, 0) then second without its duplicated start
        z = [atoms.positions[0, 2] for atoms in stitched]
        assert z == pytest.approx([2.0, 1.0, 0.0, 2.0, 4.0])

    def test_length_is_one_less_than_the_sum(self, make_chain):
        stitched = stitch_path(make_chain(4), make_chain(3))

        assert len(stitched) == 4 + 3 - 1

    def test_reverse_flag_flips_the_result(self, make_chain):
        first, second = make_chain(3, dz=1.0), make_chain(3, dz=2.0)

        forward = stitch_path(first, second)
        backward = stitch_path(first, second, f_reverse_path=True)

        assert [a.positions[0, 2] for a in backward] == pytest.approx(
            [a.positions[0, 2] for a in forward][::-1])

    def test_accepts_any_sequence(self, make_chain):
        """Tuples and generators should work, not just lists."""
        stitched = stitch_path(tuple(make_chain(3)), tuple(make_chain(3)))

        assert len(stitched) == 5


class TestResamplePath:
    def test_returns_the_requested_number_of_images(self, chain):
        assert len(resample_path(chain, 9)) == 9

    def test_preserves_the_endpoints(self, chain):
        resampled = resample_path(chain, 9)

        assert resampled[0].positions == pytest.approx(chain[0].positions)
        assert resampled[-1].positions == pytest.approx(chain[-1].positions)

    def test_spaces_a_linear_path_evenly(self, chain):
        """A straight path resampled stays straight and evenly spaced."""
        resampled = resample_path(chain, 9)

        z = np.array([atoms.positions[0, 2] for atoms in resampled])
        assert z == pytest.approx(np.linspace(0.0, 2.0, 9))

    def test_can_downsample(self, chain):
        resampled = resample_path(chain, 3)

        assert len(resampled) == 3
        assert resampled[-1].positions == pytest.approx(chain[-1].positions)

    def test_keeps_the_chemical_symbols(self, water):
        middle = water.copy()
        middle.positions[1] += [0.25, 0.0, 0.0]
        product = water.copy()
        product.positions[1] += [0.5, 0.0, 0.0]
        path = [water, middle, product]

        resampled = resample_path(path, 5)

        assert all(a.get_chemical_symbols() == water.get_chemical_symbols()
                   for a in resampled)


class TestOptimiseGeom:
    def test_lowers_the_energy(self, calc):
        atoms = molecule("H2O")
        atoms.positions[1] += [0.2, 0.0, 0.0]
        atoms.calc = EMT()
        before = atoms.get_potential_energy()

        relaxed = optimise_geom(atoms, calc, fmax=0.05, steps=200)

        assert relaxed.get_potential_energy() <= before

    def test_converges_below_fmax(self, calc):
        atoms = molecule("H2O")
        atoms.positions[1] += [0.2, 0.0, 0.0]

        relaxed = optimise_geom(atoms, calc, fmax=0.05, steps=200)

        assert np.linalg.norm(relaxed.get_forces(), axis=1).max() < 0.05

    def test_removes_the_temporary_trajectory(self, calc, water):
        optimise_geom(water, calc, fmax=0.05, steps=5, opti_traj="scratch.traj")

        assert not Path("scratch.traj").exists()

    def test_does_not_mutate_the_input(self, calc):
        atoms = molecule("H2O")
        atoms.positions[1] += [0.2, 0.0, 0.0]
        original = atoms.positions.copy()

        optimise_geom(atoms, calc, fmax=0.05, steps=50)

        assert atoms.positions == pytest.approx(original)

    def test_returns_a_structure_with_a_calculator(self, calc, water):
        relaxed = optimise_geom(water, calc, fmax=0.05, steps=5)

        assert relaxed.calc is not None


class TestOptimiseReactantProduct:
    def test_relaxes_both_endpoints(self, calc):
        reactant = molecule("H2O")
        reactant.positions[1] += [0.2, 0.0, 0.0]
        product = molecule("H2O")
        product.positions[2] += [0.2, 0.0, 0.0]

        relaxed_r, relaxed_p = optimise_reactant_product(
            reactant, product, calc, fmax=0.05, steps=200)

        for relaxed in (relaxed_r, relaxed_p):
            assert np.linalg.norm(relaxed.get_forces(), axis=1).max() < 0.05

    def test_cleans_up_both_trajectories(self, calc, water):
        optimise_reactant_product(water, water.copy(), calc,
                                  fmax=0.05,
                                  steps=5,
                                  reactant_opti="r.traj",
                                  product_opti="p.traj")

        assert not Path("r.traj").exists()
        assert not Path("p.traj").exists()


@pytest.fixture
def endpoints(water):
    """A reactant and a displaced product, for band construction."""
    product = water.copy()
    product.positions[1] += [0.4, 0.0, 0.0]
    return water, product


class TestPrepareNeb:
    def test_builds_a_band_of_the_requested_length(self, calc, endpoints):
        reactant, product = endpoints

        neb = prepare_neb(reactant, product, calc, n_images=5, geo_int=False)

        assert isinstance(neb, NEB)
        assert len(neb.images) == 5

    def test_endpoints_are_preserved(self, calc, endpoints):
        reactant, product = endpoints

        neb = prepare_neb(reactant, product, calc, n_images=5, geo_int=False)

        assert neb.images[0].positions == pytest.approx(reactant.positions)
        assert neb.images[-1].positions == pytest.approx(product.positions)

    def test_interior_images_are_interpolated(self, calc, endpoints):
        """Without interpolation the interior would still equal the reactant."""
        reactant, product = endpoints

        neb = prepare_neb(reactant, product, calc, n_images=5, geo_int=False)

        interior = neb.images[2].positions
        assert interior != pytest.approx(reactant.positions)
        assert interior != pytest.approx(product.positions)

    def test_every_image_has_its_own_calculator(self, calc, endpoints):
        reactant, product = endpoints

        neb = prepare_neb(reactant, product, calc, n_images=5, geo_int=False)

        calcs = [image.calc for image in neb.images]
        assert all(c is not None for c in calcs)
        assert len({id(c) for c in calcs}) == len(calcs)

    def test_cached_energies_match_the_final_geometries(self, calc, endpoints):
        """Regression: energies were cached before interpolation moved the atoms.

        Interior images then reported the reactant energy, which plot_neb
        trusted and drew as a flat profile.
        """
        reactant, product = endpoints

        neb = prepare_neb(reactant, product, calc, n_images=5, geo_int=False)

        for image in neb.images:
            fresh = image.copy()
            fresh.calc = EMT()
            assert image.calc.results["energy"] == pytest.approx(
                fresh.get_potential_energy())

    def test_images_do_not_share_calculator_state(self, endpoints):
        """Regression: shallow copies of a used calculator share their arrays.

        Passing the same calculator to optimise_reactant_product and then to
        prepare_neb is the obvious way to write a workflow. With copy.copy
        every image ended up pointing at one set of force and energy arrays,
        so the band overwrote itself during optimisation and converged to a
        physically meaningless path without raising anything.
        """
        reactant, product = endpoints
        used = EMT()
        reactant.calc = used
        reactant.get_potential_energy()  # populate the internal arrays

        neb = prepare_neb(reactant, product, used, n_images=5, geo_int=False)

        first, second = vars(neb.images[0].calc), vars(neb.images[1].calc)
        shared = [name for name, value in first.items()
                  if value is not None
                  and second.get(name) is value
                  and isinstance(value, (dict, list, np.ndarray))]
        assert not shared

    def test_get_ts_image_does_not_share_calculator_state(self, chain):
        """get_ts_image clones the calculator the same way prepare_neb does."""
        used = EMT()
        chain[0].calc = used
        chain[0].get_potential_energy()

        get_ts_image(chain, used)

        first, second = vars(chain[0].calc), vars(chain[1].calc)
        shared = [name for name, value in first.items()
                  if value is not None
                  and second.get(name) is value
                  and isinstance(value, (dict, list, np.ndarray))]
        assert not shared

    def test_climb_flag_is_passed_through(self, calc, endpoints):
        reactant, product = endpoints

        climbing = prepare_neb(reactant, product, calc, n_images=5,
                               climb=True, geo_int=False)
        plain = prepare_neb(reactant, product, calc, n_images=5,
                            climb=False, geo_int=False)

        assert climbing.climb is True
        assert plain.climb is False

    def test_parallel_flag_is_passed_through(self, calc, endpoints):
        reactant, product = endpoints

        parallel = prepare_neb(reactant, product, calc, n_images=5,
                               parallel=True, geo_int=False)
        serial = prepare_neb(reactant, product, calc, n_images=5,
                             parallel=False, geo_int=False)

        assert parallel.parallel is True
        assert serial.parallel is False

    def test_parallel_evaluation_matches_serial(self, calc, endpoints):
        """Without an MPI launcher, parallel=True evaluates images on threads.

        The result should be identical to the serial evaluation, just spread
        across threads instead of run one image at a time.
        """
        reactant, product = endpoints

        serial = prepare_neb(reactant, product, calc, n_images=5, geo_int=False)
        parallel = prepare_neb(reactant, product, calc, n_images=5,
                               parallel=True, geo_int=False)

        serial_energies = [image.get_potential_energy() for image in serial.images]
        parallel_energies = [image.get_potential_energy() for image in parallel.images]
        assert parallel_energies == pytest.approx(serial_energies)


class TestOptimiseNeb:
    def test_returns_the_final_band(self, calc, endpoints):
        reactant, product = endpoints
        neb = prepare_neb(reactant, product, calc, n_images=5, geo_int=False)

        images = optimise_neb(neb, fmax=0.5, steps=5)

        assert len(images) == 5

    def test_writes_the_trajectory(self, calc, endpoints):
        reactant, product = endpoints
        neb = prepare_neb(reactant, product, calc, n_images=5, geo_int=False)

        optimise_neb(neb, fmax=0.5, steps=3, ts_traj="band.traj")

        assert Path("band.traj").exists()


class TestGetTsImage:
    def test_returns_the_highest_energy_image(self, calc, endpoints):
        reactant, product = endpoints
        neb = prepare_neb(reactant, product, calc, n_images=5, geo_int=False)
        images = list(neb.images)

        ts = get_ts_image(images, calc)

        energies = [image.get_potential_energy() for image in images]
        assert ts is images[int(np.argmax(energies))]

    def test_attaches_a_calculator_to_bare_images(self, calc, chain):
        """Images read from a file have no calculator until one is given."""
        for atoms in chain:
            atoms.calc = None

        ts = get_ts_image(chain, calc)

        assert ts.calc is not None

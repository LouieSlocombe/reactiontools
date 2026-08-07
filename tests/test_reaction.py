"""Tests for reactiontools.tools_reaction."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import add_adsorbate, fcc100, molecule
from ase.calculators.emt import EMT
from ase.calculators.socketio import PySocketIOClient, SocketIOCalculator
from ase.constraints import FixAtoms
from ase.mep import NEB

from reactiontools import (get_fmax,
                           get_neb_path,
                           get_ts_image,
                           get_vibrations,
                           optimise_geom,
                           optimise_irc,
                           optimise_neb,
                           optimise_reactant_product,
                           optimise_ts,
                           prepare_neb,
                           prepare_parallel_neb,
                           resample_path,
                           socket_calculators,
                           stitch_path)
from reactiontools import tools_reaction

# Skip the saddle-point tests rather than the whole module: sella is the
# optional [ts] extra, and everything else here works without it.
sella_required = pytest.mark.skipif(
    importlib.util.find_spec("sella") is None,
    reason="sella is the optional [ts] extra")


class _FakeSocketIOCalculator:
    """Stand-in for ase.calculators.socketio.SocketIOCalculator.

    A real one launches an external process and speaks the i-PI protocol,
    which needs a calculator that knows how to run as a socket client (e.g.
    Espresso, Aims, Siesta) - EMT doesn't. This records how it was
    constructed and hands the wrapped calculator straight back, so the
    socket wiring in optimise_geom can be checked without an external
    process.
    """

    instances = []

    def __init__(self, calc, port=None, unixsocket=None, log=None):
        self.calc = calc
        self.port = port
        self.unixsocket = unixsocket
        self.log = log
        self.closed = False
        _FakeSocketIOCalculator.instances.append(self)

    def __enter__(self):
        return self.calc

    def __exit__(self, *exc_info):
        self.closed = True
        return False


@pytest.fixture
def fake_socketio(monkeypatch):
    """Patch tools_reaction's SocketIOCalculator with a recording stub."""
    _FakeSocketIOCalculator.instances = []
    monkeypatch.setattr(tools_reaction, "SocketIOCalculator", _FakeSocketIOCalculator)
    return _FakeSocketIOCalculator


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

    def test_uses_socketio_when_requested(self, calc, water, fake_socketio):
        optimise_geom(water, calc, fmax=0.05, steps=5,
                      use_socket=True, socket_port=12345, socket_log="sock.log")

        assert len(fake_socketio.instances) == 1
        used = fake_socketio.instances[0]
        assert used.calc is calc
        assert used.port == 12345
        assert used.log == "sock.log"
        assert used.closed is True

    def test_relaxes_through_the_socket_wrapper(self, calc, water, fake_socketio):
        """The fake hands EMT straight back, so relaxation still works."""
        atoms = water.copy()
        atoms.positions[1] += [0.2, 0.0, 0.0]

        relaxed = optimise_geom(atoms, calc, fmax=0.05, steps=200,
                                use_socket=True)

        assert np.linalg.norm(relaxed.get_forces(), axis=1).max() < 0.05

    def test_does_not_touch_socketio_by_default(self, calc, water, fake_socketio):
        optimise_geom(water, calc, fmax=0.05, steps=5)

        assert fake_socketio.instances == []


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

    def test_forwards_socket_options_to_both_optimisations(
            self, calc, water, fake_socketio):
        optimise_reactant_product(water, water.copy(), calc,
                                  fmax=0.05, steps=5,
                                  use_socket=True, socket_unixsocket="rt-test")

        assert len(fake_socketio.instances) == 2
        assert all(i.unixsocket == "rt-test" for i in fake_socketio.instances)


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

    def test_geodesic_interpolation_builds_a_band(self, calc, endpoints):
        """Regression: geo_int=True, the default, raised NameError.

        Extracting _build_band left prepare_neb referring to a local the
        extraction had taken with it. Every other test here passes
        geo_int=False, so the default path went unexercised and the function
        was unusable as documented.
        """
        reactant, product = endpoints

        neb = prepare_neb(reactant, product, calc, n_images=5, geo_int=True)

        assert len(neb.images) == 5
        assert all(image.calc is not None for image in neb.images)
        assert np.all(np.isfinite([image.get_potential_energy()
                                   for image in neb.images]))

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


def emt_launcher(index):
    """Launch EMT in its own process, as a stand-in for an external code.

    ``PySocketIOClient`` pickles the factory into a fresh interpreter that
    then speaks the i-PI protocol back over the socket, which is exactly the
    arrangement a DFT client uses. It always asks for the stress, so the test
    systems below need a three-dimensional cell for EMT to supply one.
    """
    return PySocketIOClient(EMT)


@pytest.fixture
def evaluated_endpoints(endpoints):
    """Endpoints carrying energies, as optimise_reactant_product leaves them.

    Bands built from these never reach for a socket to price an endpoint, so
    the tests can run servers with no clients behind them.
    """
    reactant, product = endpoints
    for atoms in (reactant, product):
        atoms.calc = EMT()
        atoms.get_potential_energy()
    return reactant, product


@pytest.fixture
def slab_endpoints():
    """A gold adatom hopping between hollow sites on Al(100).

    Periodic, so EMT can report a stress, and small enough that a full
    climbing-image band converges in a couple of seconds.
    """
    slab = fcc100("Al", size=(2, 2, 3))
    add_adsorbate(slab, "Au", 1.7, "hollow")
    slab.center(axis=2, vacuum=4.0)
    slab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in slab]))

    reactant = slab.copy()
    product = slab.copy()
    product.positions[-1, 0] += product.cell[0, 0] / 2
    for atoms in (reactant, product):
        atoms.calc = EMT()
        atoms.get_potential_energy()
    return reactant, product


class TestSocketCalculators:
    def test_opens_one_calculator_per_image(self):
        with socket_calculators(4) as calcs:
            assert len(calcs) == 4
            assert all(isinstance(c, SocketIOCalculator) for c in calcs)

    def test_each_calculator_listens_on_its_own_socket(self):
        with socket_calculators(3) as calcs:
            names = [c.server.unixsocket for c in calcs]

        assert len(set(names)) == 3

    def test_numbers_ports_from_the_base_upwards(self):
        with socket_calculators(3, port=31500) as calcs:
            assert [c.server.port for c in calcs] == [31500, 31501, 31502]

    def test_default_socket_names_are_unique_to_the_process(self):
        """Two jobs on one node must not fight over the same socket file."""
        with socket_calculators(2) as calcs:
            assert all(str(os.getpid()) in c.server.unixsocket for c in calcs)

    def test_passes_the_image_index_to_the_factory(self):
        seen = []

        def make_calc(index):
            seen.append(index)
            return EMT()

        with socket_calculators(3, make_calc):
            pass

        assert seen == [0, 1, 2]

    def test_closes_the_calculators_on_exit(self):
        with socket_calculators(3) as calcs:
            assert all(c.server is not None for c in calcs)

        assert all(c.server is None for c in calcs)

    def test_closes_the_calculators_when_the_block_raises(self):
        """A band that blows up mid-optimisation must not leak its clients."""
        with pytest.raises(RuntimeError, match="band diverged"):
            with socket_calculators(3) as calcs:
                raise RuntimeError("band diverged")

        assert all(c.server is None for c in calcs)

    def test_releases_the_socket_files(self):
        with socket_calculators(2) as calcs:
            files = [Path(f"/tmp/ipi_{c.server.unixsocket}") for c in calcs]
            assert all(f.exists() for f in files)

        assert not [f for f in files if f.exists()]

    def test_rejects_both_a_port_and_a_unixsocket(self):
        with pytest.raises(ValueError, match="only one"):
            with socket_calculators(2, port=31500, unixsocket="rt-test"):
                pass

    def test_rejects_both_factories(self):
        with pytest.raises(ValueError, match="only one"):
            with socket_calculators(2, lambda i: EMT(), make_launcher=emt_launcher):
                pass

    def test_rejects_a_non_positive_count(self):
        with pytest.raises(ValueError, match="must be positive"):
            with socket_calculators(0):
                pass


class TestPrepareParallelNeb:
    def test_builds_a_band_of_the_requested_length(self, evaluated_endpoints):
        reactant, product = evaluated_endpoints

        with prepare_parallel_neb(reactant, product, None,
                                  n_images=5, geo_int=False) as neb:
            assert isinstance(neb, NEB)
            assert len(neb.images) == 5

    def test_asks_ase_to_spread_the_images(self, evaluated_endpoints):
        """Without this ASE walks the band one image at a time, and the
        sockets would sit idle in turn rather than working together."""
        reactant, product = evaluated_endpoints

        with prepare_parallel_neb(reactant, product, None,
                                  n_images=5, geo_int=False) as neb:
            assert neb.parallel is True

    def test_gives_each_interior_image_its_own_socket(self, evaluated_endpoints):
        reactant, product = evaluated_endpoints

        with prepare_parallel_neb(reactant, product, None,
                                  n_images=6, geo_int=False) as neb:
            interior = [image.calc for image in neb.images[1:-1]]

        assert len(interior) == 4
        assert all(isinstance(c, SocketIOCalculator) for c in interior)
        assert len({id(c) for c in interior}) == 4

    def test_does_not_give_the_endpoints_sockets(self, evaluated_endpoints):
        """Two more clients would idle through the whole run for energies
        that are already known."""
        reactant, product = evaluated_endpoints

        with prepare_parallel_neb(reactant, product, None,
                                  n_images=5, geo_int=False) as neb:
            ends = [neb.images[0].calc, neb.images[-1].calc]

        assert not any(isinstance(c, SocketIOCalculator) for c in ends)

    def test_reuses_the_endpoint_energies_it_is_given(self, evaluated_endpoints):
        reactant, product = evaluated_endpoints
        expected = [reactant.get_potential_energy(),
                    product.get_potential_energy()]

        with prepare_parallel_neb(reactant, product, None,
                                  n_images=5, geo_int=False) as neb:
            got = [neb.images[0].get_potential_energy(),
                   neb.images[-1].get_potential_energy()]

        assert got == pytest.approx(expected)

    def test_pins_the_endpoint_energy_against_rigid_motion(self,
                                                           evaluated_endpoints):
        """Regression: rm_ro_trans re-aligns the final image every force call.

        A SinglePointCalculator holding the endpoint energy refuses to give
        it back once the atoms have moved, so the band died partway through
        with a bare "the property energy is not available". Rigid-body motion
        cannot change the energy, so the pinned value stays correct.
        """
        reactant, product = evaluated_endpoints

        with prepare_parallel_neb(reactant, product, None,
                                  n_images=5, geo_int=False,
                                  rm_ro_trans=True) as neb:
            endpoint = neb.images[-1]
            before = endpoint.get_potential_energy()

            endpoint.rotate(30, "z")
            endpoint.positions += [1.0, 2.0, 3.0]

            assert endpoint.get_potential_energy() == pytest.approx(before)

    def test_leaves_the_callers_endpoints_alone(self, evaluated_endpoints):
        reactant, product = evaluated_endpoints
        calcs = (reactant.calc, product.calc)
        positions = (reactant.positions.copy(), product.positions.copy())

        with prepare_parallel_neb(reactant, product, None,
                                  n_images=5, geo_int=False):
            pass

        assert (reactant.calc, product.calc) == calcs
        assert reactant.positions == pytest.approx(positions[0])
        assert product.positions == pytest.approx(positions[1])

    def test_closes_the_sockets_on_exit(self, evaluated_endpoints):
        reactant, product = evaluated_endpoints

        with prepare_parallel_neb(reactant, product, None,
                                  n_images=5, geo_int=False) as neb:
            calcs = [image.calc for image in neb.images[1:-1]]

        assert all(c.server is None for c in calcs)

    def test_rejects_a_band_with_no_interior(self, evaluated_endpoints):
        reactant, product = evaluated_endpoints

        with pytest.raises(ValueError, match="at least 3"):
            with prepare_parallel_neb(reactant, product, None, n_images=2):
                pass

    def test_prices_endpoints_that_arrive_without_an_energy(self,
                                                            slab_endpoints):
        """Endpoints read back from a file have no calculator, so the first
        socket has to evaluate them once before the band starts."""
        reactant, product = slab_endpoints
        expected = [reactant.get_potential_energy(),
                    product.get_potential_energy()]
        for atoms in (reactant, product):
            atoms.calc = None

        with prepare_parallel_neb(reactant, product, None,
                                  make_launcher=emt_launcher,
                                  n_images=4, geo_int=False,
                                  rm_ro_trans=False, timeout=120) as neb:
            got = [neb.images[0].get_potential_energy(),
                   neb.images[-1].get_potential_energy()]

        assert got == pytest.approx(expected)

    def test_ignores_an_energy_left_over_from_the_other_endpoint(self,
                                                                 slab_endpoints):
        """Regression: optimise_reactant_product sends both endpoints through
        one calculator, so its cached result belongs to whichever went last.

        Trusting it blind handed the reactant the product's energy. Asking the
        calculator for a fresh one was no better: after the ``with`` block that
        opened it, the socket behind an endpoint is shut, and re-evaluating
        through it died on a closed log file rather than on anything legible.
        """
        reactant, product = slab_endpoints
        # The hop is symmetric, so lift the adatom to tell the two ends apart:
        # otherwise the reactant reading the product's energy looks correct.
        product.positions[-1, 2] += 0.2
        product.calc = EMT()
        expected = [reactant.get_potential_energy(),
                    product.get_potential_energy()]
        assert expected[0] != pytest.approx(expected[1])

        # What the documented workflow does: price both endpoints through one
        # socket, then build the band once that socket has been closed.
        with socket_calculators(1, make_launcher=emt_launcher,
                                timeout=120) as (calc,):
            for atoms in (reactant, product):  # product priced last, and cached
                atoms.calc = calc
                atoms.get_potential_energy()

        with prepare_parallel_neb(reactant, product, None,
                                  make_launcher=emt_launcher,
                                  n_images=4, geo_int=False,
                                  rm_ro_trans=False, timeout=120) as neb:
            got = [neb.images[0].get_potential_energy(),
                   neb.images[-1].get_potential_energy()]

        assert got == pytest.approx(expected)

    def test_relaxes_the_same_band_as_the_serial_route(self, slab_endpoints):
        """The whole point: same physics, evaluated concurrently.

        Runs the Al(100) hop from the README both ways and compares the
        converged profile image by image.
        """
        reactant, product = slab_endpoints

        with prepare_parallel_neb(reactant, product, None,
                                  make_launcher=emt_launcher,
                                  n_images=5, climb=True, rm_ro_trans=False,
                                  geo_int=False, timeout=120) as neb:
            parallel = optimise_neb(neb, fmax=0.05, steps=200,
                                    ts_traj="parallel.traj")

        serial = optimise_neb(
            prepare_neb(reactant, product, EMT(), n_images=5, climb=True,
                        rm_ro_trans=False, geo_int=False),
            fmax=0.05, steps=200, ts_traj="serial.traj")

        assert ([image.get_potential_energy() for image in parallel]
                == pytest.approx([image.get_potential_energy()
                                  for image in serial], abs=1e-6))


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

    def test_uses_the_energies_the_images_carry(self, slab_endpoints):
        """A band from prepare_parallel_neb comes back after its sockets have
        closed, so there is no calculator left to hand over."""
        reactant, product = slab_endpoints
        neb = prepare_neb(reactant, product, EMT(), n_images=5,
                          rm_ro_trans=False, geo_int=False)
        images = optimise_neb(neb, fmax=0.5, steps=3, ts_traj="band.traj")

        ts = get_ts_image(images)

        energies = [image.get_potential_energy() for image in images]
        assert ts is images[int(np.argmax(energies))]

    def test_a_given_calculator_still_overrides(self, calc, chain):
        """Passing one must re-evaluate, not defer to a stale cached energy."""
        for atoms in chain:
            atoms.calc = EMT()
            atoms.get_potential_energy()
        originals = [atoms.calc for atoms in chain]

        get_ts_image(chain, calc)

        assert all(atoms.calc is not original
                   for atoms, original in zip(chain, originals))


class TestGetFmax:
    def test_matches_the_largest_per_atom_force(self, calc, water):
        water.calc = calc

        fmax = get_fmax(water)

        forces = water.get_forces()
        assert fmax == pytest.approx(np.linalg.norm(forces, axis=1).max())

    def test_is_zero_for_a_relaxed_structure(self, calc, water):
        relaxed = optimise_geom(water, calc, fmax=0.01)
        relaxed.calc = calc

        assert get_fmax(relaxed) < 0.01


class TestGetVibrations:
    def test_returns_one_frequency_per_degree_of_freedom(self, calc, water):
        freqs = get_vibrations(water, calc)

        assert len(freqs) == 3 * len(water)

    def test_does_not_modify_the_input(self, calc, water):
        before = water.positions.copy()

        get_vibrations(water, calc)

        assert water.positions == pytest.approx(before)
        assert water.calc is None

    def test_leaves_no_cached_displacements_behind(self, calc, water, tmp_path):
        """A stale cache from an earlier geometry would be reused silently.

        ASE empties the cache but keeps the directory, which is harmless: it
        is the displacement files that would be picked up again.
        """
        get_vibrations(water, calc)

        assert list((tmp_path / "vib").iterdir()) == []


@sella_required
class TestOptimiseTs:
    def test_returns_a_structure_and_keeps_the_trajectory(self, calc, water):
        ts = optimise_ts(water, calc, fmax=0.5, steps=2)

        assert len(ts) == len(water)
        assert Path("sella.traj").exists()

    def test_does_not_modify_the_input(self, calc, water):
        before = water.positions.copy()

        optimise_ts(water, calc, fmax=0.5, steps=2)

        assert water.positions == pytest.approx(before)


@sella_required
class TestOptimiseIrc:
    def test_returns_both_directions(self, calc, water):
        forward, reverse = optimise_irc(water, calc, fmax=0.5, steps=2)

        assert len(forward) >= 1
        assert len(reverse) >= 1
        assert Path("irc_f.traj").exists()
        assert Path("irc_r.traj").exists()

    def test_the_halves_stitch_into_one_path(self, calc, water):
        """The point of returning them: stitch_path takes it from here."""
        forward, reverse = optimise_irc(water, calc, fmax=0.5, steps=2)

        path = stitch_path(reverse, forward)

        assert len(path) == len(forward) + len(reverse) - 1

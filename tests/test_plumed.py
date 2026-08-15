"""Tests for reactiontools.tools_plumed."""

import importlib.util
import subprocess

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin

from reactiontools import (
    PLUMED_ASE_UNITS,
    find_molecules,
    plumed_calculator,
    plumed_metad_input,
    plumed_selection,
    run_sum_hills,
    sum_hills_files,
)

# Skip only the biased runs: the input builder is string handling and works
# without PLUMED, as does everything else in the module.
plumed_required = pytest.mark.skipif(
    importlib.util.find_spec("plumed") is None,
    reason="needs the plumed Python module (conda install -c conda-forge py-plumed)",
)


class TestPlumedSelection:
    @pytest.mark.parametrize(
        "indices, expected",
        [
            ([0], "1"),
            ([0, 1, 2], "1-3"),
            ([0, 2, 4], "1,3,5"),
            ([0, 1, 2, 5], "1-3,6"),
            ([0, 1, 5, 6, 7, 20], "1-2,6-8,21"),
        ],
    )
    def test_formats_runs_compactly(self, indices, expected):
        assert plumed_selection(indices) == expected

    def test_converts_to_one_based_indexing(self):
        """PLUMED counts atoms from 1, ASE from 0."""
        assert plumed_selection([0]) == "1"

    def test_sorts_the_input(self):
        assert plumed_selection([4, 3, 0, 1, 2]) == "1-5"

    def test_collapses_duplicates(self):
        assert plumed_selection([0, 0, 1, 1, 2]) == "1-3"

    def test_accepts_a_numpy_array(self):
        """find_molecules returns integer arrays, which must feed straight in."""
        assert plumed_selection(np.array([0, 1, 2])) == "1-3"

    def test_accepts_a_generator(self):
        assert plumed_selection(i for i in range(3)) == "1-3"

    def test_rejects_an_empty_selection(self):
        with pytest.raises(ValueError, match="empty atom selection"):
            plumed_selection([])


class TestFindMolecules:
    def test_finds_a_single_molecule(self, water):
        groups = find_molecules(water)

        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1, 2]

    def test_separates_two_distant_molecules(self):
        first = molecule("H2O")
        second = molecule("H2O")
        second.positions += [20.0, 0.0, 0.0]

        groups = find_molecules(first + second)

        assert len(groups) == 2
        assert sorted(len(g) for g in groups) == [3, 3]

    def test_groups_partition_every_atom(self):
        atoms = molecule("H2O") + molecule("CH4")
        atoms.positions[3:] += [20.0, 0.0, 0.0]

        groups = find_molecules(atoms)

        assert sorted(np.concatenate(groups)) == list(range(len(atoms)))

    def test_isolated_atoms_are_their_own_groups(self):
        atoms = Atoms("H3", positions=[[0, 0, 0], [20, 0, 0], [40, 0, 0]])

        assert len(find_molecules(atoms)) == 3

    def test_output_feeds_plumed_selection(self, water):
        """The two helpers are meant to be chained."""
        group = find_molecules(water)[0]

        assert plumed_selection(group) == "1-3"


class TestRunSumHills:
    @pytest.fixture
    def recorded(self, monkeypatch):
        """Capture the argv that would have been handed to plumed."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        return calls

    def test_builds_the_expected_command(self, recorded):
        run_sum_hills(hills="HILLS", outfile="fes.dat", verbose=False)

        cmd, _ = recorded[0]
        assert cmd == [
            "plumed",
            "sum_hills",
            "--hills",
            "HILLS",
            "--outfile",
            "fes.dat",
            "--mintozero",
        ]

    def test_mintozero_is_optional(self, recorded):
        run_sum_hills(mintozero=False, verbose=False)

        cmd, _ = recorded[0]
        assert "--mintozero" not in cmd

    def test_accepts_path_objects(self, recorded, tmp_path):
        run_sum_hills(
            hills=tmp_path / "HILLS", outfile=tmp_path / "fes.dat", verbose=False
        )

        cmd, _ = recorded[0]
        assert all(isinstance(part, str) for part in cmd)
        assert str(tmp_path / "HILLS") in cmd

    def test_checks_the_exit_status(self, recorded):
        """A failed sum_hills must not pass silently."""
        run_sum_hills(verbose=False)

        _, kwargs = recorded[0]
        assert kwargs["check"] is True

    def test_returns_the_command_line(self, recorded):
        returned = run_sum_hills(
            hills="HILLS", outfile="fes.dat", mintozero=False, verbose=False
        )

        assert returned == "plumed sum_hills --hills HILLS --outfile fes.dat"

    def test_verbose_prints_the_command(self, recorded, capsys):
        run_sum_hills(verbose=True)

        assert "plumed sum_hills" in capsys.readouterr().out

    def test_quiet_by_default_when_verbose_is_false(self, recorded, capsys):
        run_sum_hills(verbose=False)

        assert capsys.readouterr().out == ""

    def test_propagates_a_plumed_failure(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(subprocess, "run", fake_run)

        with pytest.raises(subprocess.CalledProcessError):
            run_sum_hills(verbose=False)


class TestPlumedMetadInput:
    def test_builds_the_expected_lines(self):
        lines = plumed_metad_input(
            cvs=["d1: DISTANCE ATOMS=1,2"],
            sigma=0.05,
            height=0.02,
            pace=100,
            biasfactor=10,
            temperature=300,
        )

        assert lines == [
            PLUMED_ASE_UNITS,
            "d1: DISTANCE ATOMS=1,2",
            (
                "METAD ARG=d1 SIGMA=0.05 HEIGHT=0.02 PACE=100 FILE=HILLS "
                "BIASFACTOR=10 TEMP=300"
            ),
            "PRINT ARG=d1 FILE=COLVAR STRIDE=10",
        ]

    def test_declares_ase_units_first(self):
        """Without it PLUMED reads and writes nm and kJ/mol, not A and eV."""
        lines = plumed_metad_input(
            cvs=["d: DISTANCE ATOMS=1,2"], sigma=0.1, height=0.01, pace=10
        )

        assert lines[0] == PLUMED_ASE_UNITS

    def test_units_can_be_turned_off(self):
        lines = plumed_metad_input(
            cvs=["d: DISTANCE ATOMS=1,2"], sigma=0.1, height=0.01, pace=10, units=False
        )

        assert PLUMED_ASE_UNITS not in lines

    def test_biases_every_collective_variable(self):
        lines = plumed_metad_input(
            cvs=["d1: DISTANCE ATOMS=1,2", "t1: TORSION ATOMS=1,2,3,4"],
            sigma=[0.05, 0.1],
            height=0.02,
            pace=100,
        )

        metad = next(line for line in lines if line.startswith("METAD"))
        assert "ARG=d1,t1" in metad
        assert "SIGMA=0.05,0.1" in metad

    def test_one_sigma_covers_every_variable(self):
        lines = plumed_metad_input(
            cvs=["a: DISTANCE ATOMS=1,2", "b: DISTANCE ATOMS=3,4"],
            sigma=0.07,
            height=0.02,
            pace=100,
        )

        assert "SIGMA=0.07,0.07" in next(
            line for line in lines if line.startswith("METAD")
        )

    def test_hills_default_matches_run_sum_hills(self):
        """The two ends of the workflow have to agree without being told to."""
        lines = plumed_metad_input(
            cvs=["d: DISTANCE ATOMS=1,2"], sigma=0.1, height=0.01, pace=10
        )

        assert "FILE=HILLS" in next(line for line in lines if line.startswith("METAD"))

    def test_plain_metadynamics_has_no_bias_factor(self):
        lines = plumed_metad_input(
            cvs=["d: DISTANCE ATOMS=1,2"], sigma=0.1, height=0.01, pace=10
        )

        assert "BIASFACTOR" not in " ".join(lines)

    def test_the_colvar_print_can_be_dropped(self):
        lines = plumed_metad_input(
            cvs=["d: DISTANCE ATOMS=1,2"], sigma=0.1, height=0.01, pace=10, colvar=None
        )

        assert not any(line.startswith("PRINT") for line in lines)

    def test_extra_keywords_land_on_the_metad_line(self):
        lines = plumed_metad_input(
            cvs=["d: DISTANCE ATOMS=1,2"],
            sigma=0.1,
            height=0.01,
            pace=10,
            metad_extra="GRID_MIN=0 GRID_MAX=8",
        )

        assert next(line for line in lines if line.startswith("METAD")).endswith(
            "GRID_MIN=0 GRID_MAX=8"
        )

    def test_extra_lines_are_appended(self):
        lines = plumed_metad_input(
            cvs=["d: DISTANCE ATOMS=1,2"],
            sigma=0.1,
            height=0.01,
            pace=10,
            extra=["FLUSH STRIDE=100"],
        )

        assert lines[-1] == "FLUSH STRIDE=100"

    def test_accepts_a_selection_from_plumed_selection(self):
        """The two are meant to be used together."""
        cv = f"c1: COORDINATION GROUPA={plumed_selection([0, 1, 2])} GROUPB=4"

        lines = plumed_metad_input(cvs=[cv], sigma=0.1, height=0.01, pace=10)

        assert "c1: COORDINATION GROUPA=1-3 GROUPB=4" in lines
        assert "ARG=c1" in next(line for line in lines if line.startswith("METAD"))

    def test_rejects_an_empty_variable_list(self):
        with pytest.raises(ValueError, match="at least one collective"):
            plumed_metad_input(cvs=[], sigma=0.1, height=0.01, pace=10)

    def test_rejects_an_unlabelled_variable(self):
        with pytest.raises(ValueError, match="no label"):
            plumed_metad_input(
                cvs=["DISTANCE ATOMS=1,2"], sigma=0.1, height=0.01, pace=10
            )

    def test_rejects_duplicate_labels(self):
        with pytest.raises(ValueError, match="unique"):
            plumed_metad_input(
                cvs=["d: DISTANCE ATOMS=1,2", "d: DISTANCE ATOMS=3,4"],
                sigma=[0.1, 0.1],
                height=0.01,
                pace=10,
            )

    def test_rejects_a_sigma_per_variable_mismatch(self):
        with pytest.raises(ValueError, match="2 sigmas for 1"):
            plumed_metad_input(
                cvs=["d: DISTANCE ATOMS=1,2"], sigma=[0.1, 0.2], height=0.01, pace=10
            )

    def test_well_tempered_needs_a_temperature(self):
        with pytest.raises(ValueError, match="needs temperature"):
            plumed_metad_input(
                cvs=["d: DISTANCE ATOMS=1,2"],
                sigma=0.1,
                height=0.01,
                pace=10,
                biasfactor=10,
            )

    def test_rejects_a_bias_factor_of_one_or_less(self):
        with pytest.raises(ValueError, match="greater than 1"):
            plumed_metad_input(
                cvs=["d: DISTANCE ATOMS=1,2"],
                sigma=0.1,
                height=0.01,
                pace=10,
                biasfactor=1,
                temperature=300,
            )


@pytest.fixture
def dimer():
    """Two copper atoms in a box, cheap to run biased dynamics on."""
    return Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        cell=[12.0, 12.0, 12.0],
        pbc=True,
    )


@pytest.fixture
def metad_lines():
    """Well-tempered metadynamics along the dimer's bond length."""
    return plumed_metad_input(
        cvs=["d1: DISTANCE ATOMS=1,2"],
        sigma=0.1,
        height=0.05,
        pace=5,
        biasfactor=8,
        temperature=300,
    )


@pytest.mark.integration
@plumed_required
class TestPlumedCalculator:
    def test_biases_a_dynamics_run_and_deposits_hills(
        self, dimer, metad_lines, tmp_path
    ):
        with plumed_calculator(
            dimer,
            EMT(),
            metad_lines,
            timestep=1.0 * units.fs,
            temperature=300,
            log="plumed.log",
        ):
            Langevin(dimer, 1.0 * units.fs, temperature_K=300, friction=0.02).run(30)

        hills = [
            line
            for line in (tmp_path / "HILLS").read_text().splitlines()
            if not line.startswith("#")
        ]
        assert len(hills) > 0

    def test_attaches_the_biased_calculator_for_the_block(self, dimer, metad_lines):
        with plumed_calculator(
            dimer, EMT(), metad_lines, timestep=1.0 * units.fs, log="plumed.log"
        ) as biased:
            assert dimer.calc is biased

    def test_puts_the_original_calculator_back(self, dimer, metad_lines):
        """Regression: an ASE calculator given atoms= hangs itself on them.

        Reading dimer.calc after constructing the Plumed calculator therefore
        captured the biased one, and the block restored it over itself.
        """
        original = EMT()
        dimer.calc = original

        with plumed_calculator(
            dimer, original, metad_lines, timestep=1.0 * units.fs, log="plumed.log"
        ):
            pass

        assert dimer.calc is original

    def test_leaves_atoms_without_a_calculator_as_it_found_them(
        self, dimer, metad_lines
    ):
        with plumed_calculator(
            dimer, EMT(), metad_lines, timestep=1.0 * units.fs, log="plumed.log"
        ):
            pass

        assert dimer.calc is None

    def test_restores_and_flushes_when_the_run_raises(
        self, dimer, metad_lines, tmp_path
    ):
        """A blown-up simulation must not also lose the hills it did deposit."""
        with (
            pytest.raises(RuntimeError, match="blew up"),
            plumed_calculator(
                dimer,
                EMT(),
                metad_lines,
                timestep=1.0 * units.fs,
                temperature=300,
                log="plumed.log",
            ),
        ):
            Langevin(dimer, 1.0 * units.fs, temperature_K=300, friction=0.02).run(30)
            raise RuntimeError("simulation blew up")

        assert dimer.calc is None
        assert (tmp_path / "HILLS").exists()
        assert [
            line
            for line in (tmp_path / "HILLS").read_text().splitlines()
            if not line.startswith("#")
        ]

    def test_writes_the_collective_variable_in_angstrom(
        self, dimer, metad_lines, tmp_path
    ):
        """The UNITS line is what makes this A rather than PLUMED's nm."""
        with plumed_calculator(
            dimer,
            EMT(),
            metad_lines,
            timestep=1.0 * units.fs,
            temperature=300,
            log="plumed.log",
        ):
            Langevin(dimer, 1.0 * units.fs, temperature_K=300, friction=0.02).run(20)

        rows = [
            line
            for line in (tmp_path / "COLVAR").read_text().splitlines()
            if not line.startswith("#")
        ]
        assert float(rows[0].split()[1]) == pytest.approx(2.5, abs=0.2)

    def test_without_the_units_line_it_comes_back_in_nanometres(self, dimer, tmp_path):
        """Guards the test above: 2.5 A is 0.25 nm, a factor of ten apart."""
        lines = plumed_metad_input(
            cvs=["d1: DISTANCE ATOMS=1,2"], sigma=0.01, height=0.05, pace=5, units=False
        )

        with plumed_calculator(
            dimer, EMT(), lines, timestep=1.0 * units.fs, log="plumed.log"
        ):
            Langevin(dimer, 1.0 * units.fs, temperature_K=300, friction=0.02).run(20)

        rows = [
            line
            for line in (tmp_path / "COLVAR").read_text().splitlines()
            if not line.startswith("#")
        ]
        assert float(rows[0].split()[1]) == pytest.approx(0.25, abs=0.02)

    def test_the_bias_changes_the_forces(self, dimer, metad_lines):
        """Otherwise the run is just unbiased dynamics with extra files."""
        unbiased = EMT()
        dimer.calc = unbiased
        plain = dimer.get_forces().copy()

        with plumed_calculator(
            dimer,
            unbiased,
            metad_lines,
            timestep=1.0 * units.fs,
            temperature=300,
            log="plumed.log",
        ):
            Langevin(dimer, 1.0 * units.fs, temperature_K=300, friction=0.02).run(20)
            biased_forces = dimer.get_forces().copy()

        assert not np.allclose(plain, biased_forces)


class TestRunSumHillsOptions:
    @pytest.fixture
    def recorded(self, monkeypatch):
        """Capture the argv that would have been handed to plumed."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        return calls

    def test_stride_asks_for_a_series(self, recorded):
        run_sum_hills(stride=100, verbose=False)

        cmd, _ = recorded[0]
        assert "--stride" in cmd
        assert cmd[cmd.index("--stride") + 1] == "100"

    def test_nohistory_is_optional(self, recorded):
        run_sum_hills(stride=100, nohistory=True, verbose=False)

        assert "--nohistory" in recorded[0][0]

    def test_grid_bounds_are_passed_through(self, recorded):
        run_sum_hills(grid_min=1.5, grid_max=6.0, grid_bin=500, verbose=False)

        cmd, _ = recorded[0]
        assert cmd[cmd.index("--min") + 1] == "1.5"
        assert cmd[cmd.index("--max") + 1] == "6.0"
        assert cmd[cmd.index("--bin") + 1] == "500"

    def test_grid_bounds_take_one_value_per_variable(self, recorded):
        """A two-dimensional surface needs a bound for each variable."""
        run_sum_hills(
            grid_min=[1.0, -3.14],
            grid_max=[6.0, 3.14],
            grid_bin=[200, 100],
            verbose=False,
        )

        cmd, _ = recorded[0]
        assert cmd[cmd.index("--min") + 1] == "1.0,-3.14"
        assert cmd[cmd.index("--max") + 1] == "6.0,3.14"
        assert cmd[cmd.index("--bin") + 1] == "200,100"

    def test_idw_selects_the_variables_to_keep(self, recorded):
        run_sum_hills(idw="d1", kt=0.0259, verbose=False)

        cmd, _ = recorded[0]
        assert cmd[cmd.index("--idw") + 1] == "d1"
        assert cmd[cmd.index("--kt") + 1] == "0.0259"

    def test_idw_accepts_several_labels(self, recorded):
        run_sum_hills(idw=["d1", "t1"], kt=0.0259, verbose=False)

        cmd, _ = recorded[0]
        assert cmd[cmd.index("--idw") + 1] == "d1,t1"

    def test_kt_without_idw_is_refused(self, recorded):
        """Alone it would quietly do nothing; kt needs idw to mean anything."""
        with pytest.raises(ValueError, match="only applies when idw"):
            run_sum_hills(kt=0.0259, verbose=False)

        assert recorded == []

    def test_negbias_is_optional(self, recorded):
        run_sum_hills(negbias=True, verbose=False)

        assert "--negbias" in recorded[0][0]

    def test_extra_arguments_are_appended(self, recorded):
        run_sum_hills(extra=["--fmt", "%14.9f"], verbose=False)

        assert recorded[0][0][-2:] == ["--fmt", "%14.9f"]

    def test_the_plain_command_is_unchanged(self, recorded):
        """None of the new options may appear unless they were asked for."""
        run_sum_hills(verbose=False)

        cmd, _ = recorded[0]
        assert cmd == [
            "plumed",
            "sum_hills",
            "--hills",
            "HILLS",
            "--outfile",
            "fes.dat",
            "--mintozero",
        ]


class TestSumHillsFiles:
    def _touch(self, tmp_path, names):
        for name in names:
            (tmp_path / name).write_text("#! FIELDS d1 file.free\n")

    def test_collects_a_strided_series(self, tmp_path):
        self._touch(tmp_path, [f"fes.dat{i}.dat" for i in range(4)])

        found = sum_hills_files(tmp_path / "fes.dat")

        assert [path.name for path in found] == [
            "fes.dat0.dat",
            "fes.dat1.dat",
            "fes.dat2.dat",
            "fes.dat3.dat",
        ]

    def test_orders_numerically_not_lexicographically(self, tmp_path):
        """Regression: sorted() puts fes.dat10.dat before fes.dat2.dat.

        For a convergence series the order is the entire point, so getting it
        wrong scrambles the answer without ever looking wrong.
        """
        self._touch(tmp_path, [f"fes.dat{i}.dat" for i in range(12)])

        found = sum_hills_files(tmp_path / "fes.dat")

        indices = [int(p.name[len("fes.dat") : -len(".dat")]) for p in found]
        assert indices == list(range(12))

    def test_a_stem_outfile_gives_tidier_names(self, tmp_path):
        self._touch(tmp_path, [f"fes{i}.dat" for i in range(3)])

        found = sum_hills_files(tmp_path / "fes")

        assert [path.name for path in found] == ["fes0.dat", "fes1.dat", "fes2.dat"]

    def test_an_unstrided_run_has_no_series(self, tmp_path):
        self._touch(tmp_path, ["fes.dat"])

        assert sum_hills_files(tmp_path / "fes.dat") == []

    def test_ignores_unrelated_files(self, tmp_path):
        self._touch(
            tmp_path,
            [
                "fes.dat0.dat",
                "fes.dat1.dat",
                "HILLS",
                "COLVAR",
                "other0.dat",
                "fes.datX.dat",
            ],
        )

        found = sum_hills_files(tmp_path / "fes.dat")

        assert [path.name for path in found] == ["fes.dat0.dat", "fes.dat1.dat"]

    def test_looks_in_the_working_directory_by_default(self, tmp_path):
        """The autouse fixture puts us in tmp_path, as run_sum_hills assumes."""
        self._touch(tmp_path, ["fes.dat0.dat", "fes.dat1.dat"])

        assert len(sum_hills_files()) == 2

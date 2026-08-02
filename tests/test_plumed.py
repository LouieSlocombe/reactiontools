"""Tests for reactiontools.tools_plumed."""

import subprocess

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from reactiontools import find_molecules, plumed_selection, run_sum_hills


class TestPlumedSelection:
    @pytest.mark.parametrize("indices, expected", [
        ([0], "1"),
        ([0, 1, 2], "1-3"),
        ([0, 2, 4], "1,3,5"),
        ([0, 1, 2, 5], "1-3,6"),
        ([0, 1, 5, 6, 7, 20], "1-2,6-8,21"),
    ])
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
        assert cmd == ["plumed", "sum_hills",
                       "--hills", "HILLS",
                       "--outfile", "fes.dat",
                       "--mintozero"]

    def test_mintozero_is_optional(self, recorded):
        run_sum_hills(mintozero=False, verbose=False)

        cmd, _ = recorded[0]
        assert "--mintozero" not in cmd

    def test_accepts_path_objects(self, recorded, tmp_path):
        run_sum_hills(hills=tmp_path / "HILLS",
                      outfile=tmp_path / "fes.dat",
                      verbose=False)

        cmd, _ = recorded[0]
        assert all(isinstance(part, str) for part in cmd)
        assert str(tmp_path / "HILLS") in cmd

    def test_checks_the_exit_status(self, recorded):
        """A failed sum_hills must not pass silently."""
        run_sum_hills(verbose=False)

        _, kwargs = recorded[0]
        assert kwargs["check"] is True

    def test_returns_the_command_line(self, recorded):
        returned = run_sum_hills(hills="HILLS", outfile="fes.dat",
                                 mintozero=False, verbose=False)

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

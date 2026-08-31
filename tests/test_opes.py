"""Tests for the bundled OPES post-processing scripts and the runner for them."""

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reactiontools.opes import script_path
from reactiontools.tools_plumed import (
    _opes_fes_command,
    _opes_reweighting_command,
    combine_colvar_files,
    run_opes_reweighting,
)


class TestScriptPath:
    def test_it_finds_the_bundled_script(self) -> None:
        path = script_path("FES_from_State.py")

        assert path.is_file()
        assert path.name == "FES_from_State.py"

    def test_the_default_is_the_one_the_builders_use(self) -> None:
        assert script_path().name == "FES_from_State.py"

    def test_an_unknown_script_lists_the_ones_that_are_there(self) -> None:
        with pytest.raises(FileNotFoundError, match="FES_from_State.py"):
            script_path("NoSuchScript.py")


class TestOpesFesCommand:
    def test_it_runs_under_this_interpreter(self) -> None:
        # Not "python3": the scripts need this environment's pandas, and
        # whatever python3 resolves to on PATH may not have it.
        assert _opes_fes_command()[0] == sys.executable

    def test_it_names_the_state_and_the_output(self) -> None:
        cmd = _opes_fes_command(state="STATE", outfile="fes.dat")

        assert "--state" in cmd and "STATE" in cmd
        assert "--outfile" in cmd and "fes.dat" in cmd

    def test_the_grid_is_left_out_when_no_bounds_are_given(self) -> None:
        cmd = _opes_fes_command()

        assert "--min" not in cmd
        assert "--max" not in cmd

    def test_bounds_and_bins_are_passed_through(self) -> None:
        cmd = _opes_fes_command(grid_min=-1.1, grid_max=1.1, grid_bin=200)

        assert cmd[cmd.index("--min") + 1] == "-1.1"
        assert cmd[cmd.index("--max") + 1] == "1.1"
        assert cmd[cmd.index("--bin") + 1] == "200"

    def test_a_multi_dimensional_grid_is_comma_joined(self) -> None:
        cmd = _opes_fes_command(
            grid_min=[-1.1, -1.1], grid_max=[1.1, 1.1], grid_bin=[200, 200]
        )

        assert cmd[cmd.index("--min") + 1] == "-1.1,-1.1"
        assert cmd[cmd.index("--bin") + 1] == "200,200"

    def test_kt_is_formatted_rather_than_dumped_at_full_precision(self) -> None:
        cmd = _opes_fes_command(kt=2.494338785445972)

        assert cmd[cmd.index("--kt") + 1] == "2.49434"

    def test_half_a_grid_is_rejected(self) -> None:
        # FES_from_State.py needs both bounds together to size its grid, and
        # silently ignoring one would give a surface on the wrong axis.
        with pytest.raises(ValueError, match="both grid_min and grid_max"):
            _opes_fes_command(grid_min=-1.1)

    def test_extra_arguments_are_appended(self) -> None:
        cmd = _opes_fes_command(extra=["--all_stored"])

        assert cmd[-1] == "--all_stored"


def test_two_dimensional_reweighting_supports_rectangular_grids(tmp_path: Path) -> None:
    colvar = tmp_path / "COLVAR"
    colvar.write_text(
        "#! FIELDS time cv_x cv_y\n"
        "0.0 0.0 0.0\n"
        "1.0 0.2 0.8\n"
        "2.0 0.5 0.4\n"
        "3.0 0.8 0.2\n"
        "4.0 1.0 1.0\n"
    )
    output = tmp_path / "fes.dat"

    subprocess.run(
        [
            sys.executable,
            str(script_path("FES_from_Reweighting.py")),
            "--colvar",
            str(colvar),
            "--outfile",
            str(output),
            "--cv",
            "cv_x,cv_y",
            "--bias",
            "NO",
            "--sigma",
            "0.2,0.3",
            "--kt",
            "1.0",
            "--min",
            "0.0,0.0",
            "--max",
            "1.0,1.0",
            "--bin",
            "2,3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    grid = np.loadtxt(output, comments="#!")
    assert grid.shape == (12, 3)
    assert len(np.unique(grid[:, 0])) == 3
    assert len(np.unique(grid[:, 1])) == 4


class TestOpesReweightingCommand:
    def test_it_runs_the_reweighting_script_under_this_interpreter(self) -> None:
        cmd = _opes_reweighting_command(sigma=0.2, kt=1.0)

        assert cmd[0] == sys.executable
        assert Path(cmd[1]).name == "FES_from_Reweighting.py"

    def test_the_script_requirements_are_always_named(self) -> None:
        # --sigma and one of --kt/--temp are required=True in the script's
        # argparse, which is why they lead the signature here.
        cmd = _opes_reweighting_command(sigma=0.2, kt=1.0, colvar="COLVAR",
                                        outfile="fes.dat")

        assert cmd[cmd.index("--colvar") + 1] == "COLVAR"
        assert cmd[cmd.index("--outfile") + 1] == "fes.dat"
        assert cmd[cmd.index("--sigma") + 1] == "0.2"
        assert cmd[cmd.index("--kt") + 1] == "1"

    def test_kt_is_formatted_rather_than_dumped_at_full_precision(self) -> None:
        cmd = _opes_reweighting_command(sigma=0.2, kt=2.494338785445972)

        assert cmd[cmd.index("--kt") + 1] == "2.49434"

    def test_cv_and_bias_are_left_to_the_script_by_default(self) -> None:
        cmd = _opes_reweighting_command(sigma=0.2, kt=1.0)

        assert "--cv" not in cmd
        assert "--bias" not in cmd

    def test_cv_and_bias_pass_through(self) -> None:
        cmd = _opes_reweighting_command(sigma=0.2, kt=1.0, cv="phi", bias="NO")

        assert cmd[cmd.index("--cv") + 1] == "phi"
        assert cmd[cmd.index("--bias") + 1] == "NO"

    def test_sequences_are_comma_joined(self) -> None:
        cmd = _opes_reweighting_command(
            sigma=[0.2, 0.3], kt=1.0, cv=["phi", "psi"],
            grid_min=[0.0, 0.0], grid_max=[1.0, 2.0], grid_bin=[10, 20],
        )

        assert cmd[cmd.index("--sigma") + 1] == "0.2,0.3"
        assert cmd[cmd.index("--cv") + 1] == "phi,psi"
        assert cmd[cmd.index("--min") + 1] == "0.0,0.0"
        assert cmd[cmd.index("--max") + 1] == "1.0,2.0"
        assert cmd[cmd.index("--bin") + 1] == "10,20"

    def test_a_column_number_works_as_the_cv(self) -> None:
        cmd = _opes_reweighting_command(sigma=0.2, kt=1.0, cv=2)

        assert cmd[cmd.index("--cv") + 1] == "2"

    def test_half_a_grid_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="both grid_min and grid_max"):
            _opes_reweighting_command(sigma=0.2, kt=1.0, grid_min=0.0)

    def test_blocks_and_stride_are_alternatives(self) -> None:
        with pytest.raises(ValueError, match="alternatives"):
            _opes_reweighting_command(sigma=0.2, kt=1.0, blocks=4, stride=100)

        blocked = _opes_reweighting_command(sigma=0.2, kt=1.0, blocks=4)
        assert blocked[blocked.index("--blocks") + 1] == "4"

        strided = _opes_reweighting_command(sigma=0.2, kt=1.0, stride=100)
        assert strided[strided.index("--stride") + 1] == "100"

    def test_skiprows_and_extra_are_appended(self) -> None:
        cmd = _opes_reweighting_command(sigma=0.2, kt=1.0, skiprows=500,
                                        extra=["--reverse"])

        assert cmd[cmd.index("--skiprows") + 1] == "500"
        assert cmd[-1] == "--reverse"


class TestRunOpesReweighting:
    @pytest.fixture
    def recorded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> list[tuple[list[str], dict[str, Any]]]:
        """Capture the argv that would have been handed to the script."""
        calls = []

        def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
            calls.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        return calls

    def test_runs_the_built_command_and_checks_the_exit_status(
        self,
        recorded: list[tuple[list[str], dict[str, Any]]],
    ) -> None:
        returned = run_opes_reweighting(sigma=0.2, kt=1.0, verbose=False)

        cmd, kwargs = recorded[0]
        assert kwargs["check"] is True
        assert returned == " ".join(cmd)
        assert Path(cmd[1]).name == "FES_from_Reweighting.py"

    def test_verbose_prints_the_command(
        self,
        recorded: list[tuple[list[str], dict[str, Any]]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_opes_reweighting(sigma=0.2, kt=1.0)

        assert "FES_from_Reweighting.py" in capsys.readouterr().out


def test_independent_walkers_are_combined_and_reweighted_for_real(
    tmp_path: Path,
) -> None:
    # Two walkers' COLVARs, each with its own bias column, merged
    # contiguously so --blocks can put one block on each walker.
    header = ["#! FIELDS time cv_x metad.bias"]
    first = tmp_path / "COLVAR.0"
    first.write_text("\n".join([
        *header,
        "0.0 0.10 0.00",
        "1.0 0.30 0.10",
        "2.0 0.50 0.30",
        "3.0 0.70 0.20",
        "4.0 0.90 0.05",
    ]) + "\n")
    second = tmp_path / "COLVAR.1"
    second.write_text("\n".join([
        *header,
        "0.0 0.20 0.00",
        "1.0 0.40 0.15",
        "2.0 0.60 0.25",
        "3.0 0.80 0.10",
        "4.0 0.95 0.02",
    ]) + "\n")

    merged = combine_colvar_files([first, second], tmp_path / "COLVAR",
                                  sort_by_time=False)
    run_opes_reweighting(
        sigma=0.25,
        kt=1.0,
        colvar=merged,
        outfile=tmp_path / "fes.dat",
        cv="cv_x",
        grid_min=0.0,
        grid_max=1.0,
        grid_bin=4,
        blocks=2,
        verbose=False,
    )

    grid = np.loadtxt(tmp_path / "fes.dat", comments="#!")
    assert grid.shape == (5, 3)  # cv, free energy, cross-walker uncertainty
    assert np.isfinite(grid).all()

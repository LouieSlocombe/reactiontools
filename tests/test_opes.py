"""Tests for the bundled OPES post-processing scripts and the runner for them."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from reactiontools.opes import script_path
from reactiontools.tools_plumed import _opes_fes_command


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

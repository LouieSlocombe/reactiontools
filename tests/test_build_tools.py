"""Exercise installer ordering and failure handling without installing anything."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

BUILD_TOOLS = Path(__file__).resolve().parents[1] / "build_tools"


def run_installer(
    tmp_path: Path, installer: str, *, failure: str = "", pins: str = ""
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], Path]:
    """Copy the scripts into a scratch checkout and replace external operations."""
    repo = tmp_path / "checkout with spaces"
    scripts = repo / "build_tools"
    shutil.copytree(BUILD_TOOLS, scripts, ignore=shutil.ignore_patterns("sources"))
    # Keep the real verification helper while replacing both build functions.
    with (scripts / "build_plumed.sh").open("a") as handle:
        handle.write(
            '\nbuild_plumed() { record build_plumed "$@"; }\n'
            'build_py_plumed() { record build_py_plumed "$@"; }\n'
        )
    with (scripts / "clone_repo.sh").open("a") as handle:
        handle.write('\nclone_repo() { record clone_repo "$@"; }\n')

    base = tmp_path / "conda base"
    profile = base / "etc" / "profile.d" / "conda.sh"
    profile.parent.mkdir(parents=True)
    profile.touch()
    log = tmp_path / "commands.log"
    harness = tmp_path / "commands.sh"
    harness.write_text(
        r'''
record() {
    { printf '%s' "$1"; shift; printf '\t%s' "$@"; printf '\n'; } >> "$COMMAND_LOG"
}
conda() {
    if [[ "$*" == "info --base" ]]; then
        printf '%s\n' "$STUB_CONDA_BASE"
        return
    fi
    record conda "${CONDA_PINNED_PACKAGES-}" "$@"
    if [[ "$1 $2" == "env create" && "$FAILURE" == create ]]; then return 17; fi
}
mamba() { record mamba "$@"; }
module() { record module "$@"; }
rm() { record rm "$@"; }
pip() { record pip "$@"; }
pip3() { record pip3 "$@"; }
source() {
    if [[ "$1" == activate ]]; then record source "$@"; else builtin source "$@"; fi
}
plumed() {
    record plumed "$@"
    if [[ "$FAILURE" == opes ]]; then return 17; fi
}
check_python() {
    record "$@"
    if [[ "$FAILURE" == kernel && "$*" == *'import plumed'* ]]; then return 17; fi
    if [[ "$FAILURE" == package && "$*" == *'import reactiontools'* ]]; then return 17; fi
}
python() { check_python python "$@"; }
python3() { check_python python3 "$@"; }
'''
    )
    env = dict(
        os.environ,
        BASH_ENV=str(harness),
        COMMAND_LOG=str(log),
        STUB_CONDA_BASE=str(base),
        FAILURE=failure,
        SCRATCH=str(tmp_path / "scratch"),
        SRC_DIR=str(tmp_path / "editable sources"),
        ENV_NAME="audit-env",
        CONDA_PINNED_PACKAGES=pins,
    )
    result = subprocess.run(
        ["bash", str(scripts / installer)], env=env, text=True, capture_output=True
    )
    commands = [line.split("\t") for line in log.read_text().splitlines()]
    return result, commands, repo


@pytest.mark.parametrize("pins", ["", "numpy>=2"])
def test_sol_reads_shared_yaml_with_a_scoped_python_pin(tmp_path: Path, pins: str) -> None:
    result, commands, repo = run_installer(tmp_path, "custom_install_sol.sh", pins=pins)

    assert result.returncode == 0, result.stderr
    expected_pins = f"{pins}&python=3.13" if pins else "python=3.13"
    create = next(command for command in commands if command[2:4] == ["env", "create"])
    assert create == [
        "conda", expected_pins,
        "env", "create", "-n", "reactiontools", "-f",
        str(repo / "build_tools" / "environment.yml"), "-y",
    ]
    assert commands[-1] == ["conda", pins, "deactivate"]
    assert commands[:2] == [["module", "purge"], ["module", "load", "mamba/latest"]]
    assert [command[0] for command in commands] == [
        "module", "module", "rm", "mamba", "conda", "source",
        "build_plumed", "build_py_plumed", "clone_repo", "pip3",
        "plumed", "python3", "python3", "conda",
    ]
    assert commands[2] == [
        "rm", "-rf", str(tmp_path / "scratch" / "reactiontools_sources")
    ]
    assert commands[3] == ["mamba", "env", "remove", "-n", "reactiontools", "-y"]
    assert commands[5] == ["source", "activate", "reactiontools"]
    assert commands[9] == [
        "pip3", "install", "-e", str(tmp_path / "editable sources" / "reactiontools")
    ]


def test_conda_keeps_its_environment_name_and_interpreter(tmp_path: Path) -> None:
    result, commands, repo = run_installer(tmp_path, "conda_install.sh")

    assert result.returncode == 0, result.stderr
    assert ["conda", "", "activate", "base"] in commands
    assert [
        "conda", "", "env", "create", "-n", "audit-env", "-f",
        str(repo / "build_tools" / "environment.yml"),
    ] in commands
    assert ["pip", "install", "-e", str(repo)] in commands
    assert commands[-3:] == [
        ["plumed", "--no-mpi", "config", "-q", "module", "opes"],
        ["python", "-c", "import plumed; plumed.Plumed()"],
        ["python", "-c", "import reactiontools"],
    ]
    assert "Activate with: conda activate audit-env" in result.stdout


@pytest.mark.parametrize("installer", ["conda_install.sh", "custom_install_sol.sh"])
@pytest.mark.parametrize("failure", ["create", "opes", "kernel", "package"])
def test_installers_stop_at_a_failed_step(
    tmp_path: Path, installer: str, failure: str
) -> None:
    result, commands, _repo = run_installer(tmp_path, installer, failure=failure)

    assert result.returncode == 17
    assert "Build Complete!" not in result.stdout
    if failure == "create":
        assert not any(command[0] == "build_plumed" for command in commands)
    elif failure == "opes":
        assert commands[-1][0] == "plumed"
        assert "PLUMED opes module: OK" not in result.stdout
    elif failure == "kernel":
        assert commands[-1][-1] == "import plumed; plumed.Plumed()"
        assert "py-plumed kernel load: OK" not in result.stdout
    else:
        assert commands[-1][-1] == "import reactiontools"
        assert "reactiontools: OK" not in result.stdout

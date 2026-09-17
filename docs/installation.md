# Installation

One command, from the repository root:

```bash
bash build_tools/conda_install.sh
```

That creates the `reactiontools` conda environment, compiles PLUMED with the
OPES module and the matching Python bindings into it, and installs this package
— along with `geodesic_interpolate` and `sella`, cloned next to this repository —
in editable mode. It **removes and recreates** any environment of that name;
pass `ENV_NAME=...` to install somewhere else, or `SRC_DIR=...` to keep the
checkouts elsewhere. Checkouts that already exist are used as they are and never
wiped.

If you already have a PLUMED with OPES on your `PATH`, the environment and the
package on their own are:

```bash
conda env create -f build_tools/environment.yml -y
```

```bash
conda activate reactiontools
```

```bash
pip install -e .
```

See [build_tools/README.md](https://github.com/LouieSlocombe/reactiontools/blob/main/build_tools/README.md) for the full guide, why
PLUMED is built from source, and the Sol cluster route.

## Dependencies

Python 3.12 or newer is required. Installed requirements include `numpy>=2.0`,
`scipy>=1.16`, `matplotlib>=3.8.4`, `pandas>=2.2.2`, `ase>=3.25`,
`mdtraj>=1.10.2`, pytest and Ruff, plus two that come from git:
[`sella`](https://github.com/LouieSlocombe/sella) (saddle-point refinement and
IRC) and
[`geodesic_interpolate`](https://github.com/LouieSlocombe/geodesic_interpolate)
(used by `prepare_neb`, `quick_guess_path` and `quick_guess_ts`).
`conda_install.sh` sets both up as editable checkouts beside this repository;
a plain `pip install` takes them from GitHub instead.

The Python, ASE and SciPy minimums follow the Geodesic fork's requirements.
NumPy 2 provides the trapezoidal integration used for basin free energies;
the Matplotlib, pandas and MDTraj minimums support NumPy 2.

Three dependencies fall outside `pip install` and are only needed by the
functions named:

| Dependency | Needed by | Notes |
| --- | --- | --- |
| `plumed` executable | `run_sum_hills` | Must be on `PATH`. Called as a subprocess, not imported. Compiled by `conda_install.sh`; conda-forge's `plumed` package is built without the OPES module that `f_opes=True` inputs need. |
| `py-plumed` | `plumed_calculator` | The Python bindings, compiled by `conda_install.sh` against the same PLUMED. Imported on first use; the input builder works without it. |
| [ORCA](https://www.faccts.de/orca/) | everything in `tools_orca` | Licensed separately and installed by hand; point `ORCA_PATH` at the binary. See [build_tools/README.md](https://github.com/LouieSlocombe/reactiontools/blob/main/build_tools/README.md#orca). |

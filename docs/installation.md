# Installation

## Install with pip

Requires **Python 3.12 or later**. In an activated virtual environment or Conda
environment:

```bash
python -m pip install reactiontools
python -c "import reactiontools; print(reactiontools.__version__)"
```

Pip installs NumPy, SciPy, ASE, Matplotlib, pandas and MDTraj automatically.
Sella, IRC and geodesic interpolation are included in the package and need no
additional installation. Sella's derivatives use NumPy; JAX is not required.

## Install from source

```bash
git clone https://github.com/LouieSlocombe/reactiontools.git
cd reactiontools
python -m pip install -e .
```

An editable install uses the code in your checkout, so source changes take
effect without reinstalling. Run the remaining checkout commands from this
repository root.

## Optional workflow dependencies

Install these only for the workflows that use them:

| Workflow | Requirement | Setup |
| --- | --- | --- |
| Reconstruct metadynamics free energies with `run_sum_hills` | PLUMED executable | Put `plumed` on `PATH`. |
| Run biased ASE dynamics with `plumed_calculator` | PLUMED Python bindings (`py-plumed`) and a loadable kernel | Use matching bindings and kernel; the Conda installer below builds both. OPES simulations need the `opes` module. |
| Run ORCA electronic-structure calculations | ORCA quantum chemistry executable | Install ORCA separately and set `ORCA_PATH` to its executable, or pass `orca_path` to the calculator helper. |

Building PLUMED input files and processing existing OPES output with the
bundled scripts do not require a PLUMED installation.

For ORCA setup, see the
[build guide](https://github.com/LouieSlocombe/reactiontools/blob/main/build_tools/README.md#orca).

## Conda with PLUMED and OPES

The Linux installer creates an environment, builds PLUMED with OPES and its
matching Python bindings, and installs the checkout in editable mode. It
requires Conda and Git; Python and the compiler are installed by the script.

**Each run removes and recreates the target environment and
`build_tools/sources/`.** The default environment name is `reactiontools`.

```bash
bash build_tools/conda_install.sh
conda activate reactiontools
```

To choose a different environment name:

```bash
ENV_NAME=reactiontools-opes bash build_tools/conda_install.sh
conda activate reactiontools-opes
```

The supplied Conda environment selects Python 3.13 or later. The package itself
supports Python 3.12 or later. For an environment without the PLUMED build,
verification commands, or Sol cluster setup, see the
[build guide](https://github.com/LouieSlocombe/reactiontools/blob/main/build_tools/README.md).

## Development

From the repository root, install the test and lint tools and run the checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest --cov
ruff check .
```

The core suite has no expected failures. Sella and IRC tests run with the
standard Python dependencies. Tests marked `integration` can skip for these
specific reasons:

| Tests | Requirement to run them |
| --- | --- |
| Three ORCA calculations | A separately installed ORCA quantum chemistry executable, selected with `ORCA_PATH`. |
| Eight biased PLUMED dynamics tests | PLUMED Python bindings and a loadable kernel, as described above. |
| Five OpenMM quantity conversions | `python -m pip install openmm`. CI installs it in the Python 3.12 job with current dependencies. |
| Six comparisons against upstream Sella's compiled Gram–Schmidt implementation | An importable upstream `sella` installation. The bundled NumPy implementation also has unconditional mathematical tests. |
| Two socket transport tests | A runner that permits binding local Unix sockets. Offline orchestration tests run independently of these. |

Use `python -m pytest -rs` to see the reason for each skip in your environment,
or `python -m pytest -m "not integration"` to run the core suite alone.

To build the documentation with the same dependencies as CI:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

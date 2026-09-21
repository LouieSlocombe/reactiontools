# Environment setup scripts

These scripts create a development environment, build PLUMED 2.10.1 with the
`opes` module and matching Python bindings, and install `reactiontools` in
editable mode. For a package-only install, see the
[installation guide](../docs/installation.md).

The package requires Python 3.12 or newer. The Conda environment uses Python
3.13 or newer, and the Sol installer selects Python 3.13. Sella and geodesic
interpolation are included in the package; no separate installation or JAX is
needed.

Run the examples below from the repository root.

## Conda environment

The build scripts target Linux, including WSL. Install Conda and Git first;
the environment supplies the compiler, `make`, and Cython.

**Each run removes and recreates the target environment and
`build_tools/sources/`.** The default environment name is `reactiontools`.

```bash
bash build_tools/conda_install.sh
conda activate reactiontools
```

To use a different environment name:

```bash
ENV_NAME=reactiontools-opes bash build_tools/conda_install.sh
conda activate reactiontools-opes
```

The installer creates the environment from [`environment.yml`](environment.yml),
builds PLUMED and its Python bindings, installs this checkout, and checks that
PLUMED's `opes` module, the Python bindings, and `reactiontools` load correctly.
The environment also includes pytest, coverage, and Ruff.

`environment.yml` alone installs neither PLUMED nor `reactiontools`. The shared
[`build_plumed.sh`](build_plumed.sh) file contains build functions used by both
installers and pins the PLUMED version.

### Conda without the PLUMED build

If your workflows do not need PLUMED, create the environment and install the
checkout directly:

```bash
conda env create -f build_tools/environment.yml
conda activate reactiontools
python -m pip install -e .
```

This also works with an existing PLUMED installation: `run_sum_hills` needs
the executable on `PATH`, while `plumed_calculator` needs importable Python
bindings and a loadable kernel in the active environment.

## Sol cluster

The Sol installer uses the `mamba/latest` module and requires `$SCRATCH` and
GitHub SSH access to clone `reactiontools`.

**Each run removes and recreates the `reactiontools` environment and
`$SCRATCH/reactiontools_sources`.** The editable checkout is stored separately
at `$HOME/reactiontools_src/reactiontools` and preserved on subsequent runs.
For an interactive install, export `SRC_DIR` to change its parent directory.
An existing checkout is used without pulling updates.

Submit the installer as a batch job:

```bash
(cd build_tools && sbatch sub_sol_install.sh)
```

Or start an interactive session, then run the installer from the repository root:

```bash
interactive -t 60 -p htc -c 12 --mem=64G
bash build_tools/custom_install_sol.sh
```

After the installation finishes:

```bash
module load mamba/latest
source activate reactiontools
cd "${SRC_DIR:-$HOME/reactiontools_src}/reactiontools"
```

Neither installation route requests a GPU. Any additional calculator
requirements must be installed separately.

## Check the install

With the environment activated, run these commands from the installed checkout:

```bash
python -c 'import reactiontools'
python -m pytest --cov
ruff check .
```

For an environment with PLUMED, also check the OPES module and kernel loading:

```bash
plumed --no-mpi config -q module opes
python -c 'import plumed; plumed.Plumed()'
```

Tests that need external programs may skip when those programs are unavailable.

## ORCA

ORCA is the quantum chemistry package used here for electronic-structure
calculations. It is licensed separately and is not installed by these scripts.
Install it from the [ORCA website](https://www.faccts.de/orca/) and point
`ORCA_PATH` at its executable:

```bash
export ORCA_PATH="/path/to/orca-install/orca"
```

Add the export to your shell configuration to keep it across sessions. The ORCA
helpers accept an explicit executable path; `ORCA_PATH` provides a default and
enables the tests that require ORCA.

On some Linux systems, the command `orca` belongs to the unrelated GNOME Orca
screen reader. Set the explicit path above to select the quantum chemistry
executable; the resolver checks for this name collision before launching it.

## OpenMM workflows

For path-integral and nuclear-quantum-effect simulations with OpenMM, see
[openmmnqe](https://github.com/LouieSlocombe/openmmnqe) for its environment and
plugin setup.

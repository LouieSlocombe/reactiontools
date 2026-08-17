# Installation guide

There are two ways to install `reactiontools`, depending on where you are running it:

| Route | Use when | Script |
|---|---|---|
| Conda environment | Normal use. Everything from conda-forge except PLUMED. | `conda_install.sh` |
| Sol cluster | Running on Sol. Same split, plus the module loads and SLURM wrapper. | `sub_sol_install.sh` |

Both routes compile PLUMED and the PLUMED Python bindings (py-plumed), because
conda-forge's `plumed` is built without the `opes` module — on that build,

```bash
plumed --no-mpi config -q module opes   # exits 1
```

`tools_cv`'s `f_opes=True` builders emit `OPES_METAD` and `run_opes_fes` reads
the `STATE` file it writes, so a biased run against that kernel fails at PLUMED
init with an unknown action. Building from source also gets py-plumed — the
`plumed` module that `plumed_calculator` imports on first use — matched to the
same PLUMED version, with the kernel path baked in, so `import plumed` works
without `PLUMED_KERNEL` set in your shell.

## Prerequisites

- A compatible operating system: Linux, macOS, or Windows via WSL.
- Python 3.13 or higher.
- Conda or Mamba.
- Git, to clone the PLUMED sources. The compiler and `make` come from the
  environment (`cxx-compiler`, `make`); git does not.

## Conda environment

From this directory:

```bash
bash conda_install.sh
```

The `reactiontools` environment is recreated **from scratch on every run** — any
existing environment with that name is removed first. Set `ENV_NAME` to install
into a differently named environment instead:

```bash
ENV_NAME=reactiontools2 bash conda_install.sh
```

The script creates the environment from `environment.yml`, compiles PLUMED and
py-plumed into it (sources are cloned into the gitignored `build_tools/sources/`,
wiped on each run), installs `reactiontools` in editable mode so changes to the
source are picked up without reinstalling, and finishes with import checks. It is
equivalent to running, from this directory:

```bash
conda env create -f environment.yml
conda activate reactiontools
src_dir="$(mktemp -d)"
source build_plumed.sh && build_plumed "${src_dir}" && build_py_plumed "${src_dir}"
pip install -e ..
```

(`build_plumed.sh` is a function library rather than a script; `build_py_plumed`
reuses the plumed2 checkout that `build_plumed` leaves behind, so both take the
same working directory. The PLUMED version is pinned there, in one place.)

`environment.yml` on its own installs no PLUMED at all — everything else in
`reactiontools` works without it, but `plumed_calculator` and `run_sum_hills`
do not, so skip the build only if you already have a PLUMED with `opes` on your
`PATH`.

## Sol cluster

`custom_install_sol.sh` builds the `reactiontools` environment on Sol. Most
dependencies come from conda-forge, but PLUMED is compiled from source for the
`opes` module as above. Sources are cloned into `$SCRATCH/reactiontools_sources`,
and both the environment and the sources are recreated from scratch on each run.

Submit it as a batch job from this directory:

```bash
sbatch sub_sol_install.sh
```

Or run it directly from an interactive session:

```bash
interactive -t 60 -p htc -c 12 --mem=64G
```

No GPU is requested for either: the build and `reactiontools` itself are
CPU-only, and the calculator a script brings is what decides otherwise. Unlike
the conda route this installs `reactiontools` from git rather than in editable
mode, so re-run it to pick up changes.

## Check the install

```bash
pytest --cov
```

Check formatting and lint before committing:

```bash
ruff format --check .
ruff check .
```

## ORCA

The helpers in `reactiontools.tools_orca` shell out to ORCA, which is licensed
separately and must be installed by hand:

1. Download it from the [ORCA website](https://www.faccts.de/orca/).
2. Extract it: `tar -xf orca-x.y.z.tar.gz`
3. Point `ORCA_PATH` at the `orca` binary, adding this to your `~/.bashrc`:

   ```bash
   export ORCA_PATH="/path/to/orca_6_1_1/orca"
   ```

`orca_calc_preset()`, `orca_optimise_atoms()` and `orca_calculate_goat()` read
`ORCA_PATH` when no explicit path is passed, as do the ORCA tests — which skip
rather than fail when it is unset.

## openmmnqe

Running the same collective variables under OpenMM instead of ASE — path-integral
and nuclear-quantum-effect simulations — is
[openmmnqe](https://github.com/LouieSlocombe/openmmnqe), which depends on this
package and installs it from git. Its `build_tools/` compiles the same PLUMED,
from the same pinned version, plus the `openmm-plumed` plugin; install that
environment instead of this one if you need both.

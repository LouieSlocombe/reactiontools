# Installation guide

There are two ways to install `reactiontools`, depending on where you are running it:

| Route | Use when | Script |
|---|---|---|
| Conda environment | Normal use. Everything from conda-forge except PLUMED and the two editable checkouts. | `conda_install.sh` |
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
- Git, to clone the PLUMED sources and the editable dependencies. The compiler
  and `make` come from the environment (`cxx-compiler`, `make`); git does not.

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
source are picked up without reinstalling, and finishes with import checks. It
is equivalent to running, from this directory:

```bash
conda env create -f environment.yml
conda activate reactiontools
src_dir="$(mktemp -d)"
source build_plumed.sh && build_plumed "${src_dir}" && build_py_plumed "${src_dir}"
pip install -e ..
```

(`build_plumed.sh` and `clone_repo.sh` are function libraries rather than
scripts. `build_py_plumed` reuses the plumed2 checkout that `build_plumed` leaves
behind, so both take the same working directory, and the PLUMED version is pinned
there in one place.)

`environment.yml` on its own installs no PLUMED at all — everything else in
`reactiontools` works without it, but `plumed_calculator` and `run_sum_hills`
do not, so skip the build only if you already have a PLUMED with `opes` on your
`PATH`.

### Vendored code

There are no editable git dependencies left. Both of the forks that used to be
cloned next to this repository are now ordinary modules of this package —
linted, covered by their own test files and documented alongside their siblings.
Change either the way you would change any other module: there is nothing to
copy across and no fork to keep in step.

| Module | From | Tests | Licence |
| --- | --- | --- | --- |
| `reactiontools/tools_geodesic.py` | [`geodesic-interpolate`](https://github.com/virtualzx-nad/geodesic-interpolate), Xiaolei Zhu | `tests/test_geodesic.py` | MIT, as the package |
| `reactiontools/tools_sella.py` | [Sella](https://github.com/zadorlab/sella), Eric Hermes and contributors | `tests/test_sella.py` | **LGPL-3.0-or-later** |

Both got here for the same reason: `pyproject.toml` cannot declare either as a
dependency. A `name @ git+...` requirement in the metadata would keep
reactiontools off PyPI, which rejects direct URLs, and the forks these workflows
need are not published there. The geodesic code had a second reason: the
distribution published on PyPI as `geodesic-interpolate` exposes only the
lower-level `Geodesic` and `redistribute`, not the ASE-aware entry point this
package calls.

Two things about `tools_sella.py` in particular:

**It is LGPL, not MIT.** Sella is copyright Sandia National Laboratories
(NTESS) under US Government contract DE-NA0003525, and that licence travels
with the code. The repository `LICENSE` records which parts of the distribution
it covers, `LICENSE.LGPL` and `LICENSE.GPL` carry the text, and all three have
to stay there. If you edit that module, the LGPL asks that the change be noted:
the list at the top of the file is where the existing ones are recorded.

**Its internal coordinates and derivatives use NumPy.** No JAX installation
or compilation cache is needed. Upstream also ships three Cython extension
modules; none are built here — two were unused, and of the third only
`modified_gram_schmidt` was ever called, which is reimplemented in NumPy. That
is what keeps this a pure-Python wheel with no build step.

The geodesic notice is at the foot of the repository `LICENSE` and has to stay
there too.

## Sol cluster

`custom_install_sol.sh` builds the `reactiontools` environment on Sol. Most
dependencies come from conda-forge, but PLUMED is compiled from source for the
`opes` module as above. PLUMED sources are cloned into
`$SCRATCH/reactiontools_sources`, and both the environment and those sources are
recreated from scratch on each run.

`reactiontools` itself is cloned into `$HOME/reactiontools_src` instead —
outside the build area, since that is wiped — and installed editable, so `git
pull` in the checkout is enough to update it. Set `SRC_DIR` to put it somewhere
else.

Submit it as a batch job from this directory:

```bash
sbatch sub_sol_install.sh
```

Or run it directly from an interactive session:

```bash
interactive -t 60 -p htc -c 12 --mem=64G
```

No GPU is requested for either: the build and `reactiontools` itself are
CPU-only, and the calculator a script brings is what decides otherwise.

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
package. Its `build_tools/` compiles the same PLUMED, from the same pinned
version, plus the `openmm-plumed` plugin, and installs this repository editable
— the same checkout this one uses. Install that environment instead of this one
if you need both.

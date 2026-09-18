# Installation

From PyPI, into an existing environment:

```bash
pip install reactiontools
```

That is enough for everything except `tools_orca`, which needs ORCA — licensed
separately and installed by hand; see [Dependencies](#dependencies). The
saddle-point searches and geodesic interpolation need nothing extra.

For everything at once — environment, PLUMED and the package — one command from
the repository root:

```bash
bash build_tools/conda_install.sh
```

That creates the `reactiontools` conda environment, compiles PLUMED with the
OPES module and the matching Python bindings into it, and installs this package
in editable mode. It **removes and recreates** any environment of that name; pass `ENV_NAME=...` to
install somewhere else.

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

Python 3.12 or newer is required, and `pip install reactiontools` brings in
`numpy>=2.0`, `scipy>=1.16`, `matplotlib>=3.8.4`, `pandas>=2.2.2`, `ase>=3.25`
and `mdtraj>=1.10.2`.

The Python, ASE and SciPy minimums follow what `tools_geodesic` needs.
NumPy 2 provides the trapezoidal integration used for basin free energies;
the Matplotlib, pandas and MDTraj minimums support NumPy 2.

### Geodesic interpolation

Nothing to install. `prepare_neb(geo_int=True)`, `quick_guess_path`,
`quick_guess_ts` and `seed_product_from_ts` are built on `tools_geodesic`,
which is part of the package like any other module. `geodesic_interpolate` is
its entry point, and `Geodesic`, `redistribute` and the scalers that set the
metric are exported alongside it.

It is derived from
[`geodesic-interpolate`](https://github.com/virtualzx-nad/geodesic-interpolate)
by Xiaolei Zhu, MIT licensed as the rest of the package is — the notice is at
the foot of `LICENSE` — and it is still `zhu2019geodesic` you cite when you use
it. It lives here rather than being depended on because the version on PyPI
under the name `geodesic-interpolate` exposes only the lower-level `Geodesic`
and `redistribute`, without the ASE-aware entry point these functions call, and
a git dependency on a fork that does have it would keep this package off PyPI
altogether.

### Saddle-point searches

Nothing to install either. `optimise_ts`, `optimise_irc` and `sella_ts_search`
are built on `tools_sella`, which is part of the package like any other module.
`Sella` and `IRC` are its entry points, and `Internals` and `Constraints` are
exported alongside them for driving either under constraints.

It is derived from [Sella](https://github.com/zadorlab/sella) by Eric Hermes
and contributors, and it is `hermes2022sella` you cite when you use it. It
lives here for the same PyPI reason as the interpolation: the fork these
workflows are built against is not on PyPI, and a git dependency on it would
keep this package off PyPI altogether.

Two things follow from bringing it in.

**It is LGPL, not MIT.** Sella is licensed under the GNU Lesser General Public
License v3, copyright Sandia National Laboratories (NTESS) under US Government
contract DE-NA0003525, and that licence travels with the code: `tools_sella.py`
stays LGPL-3.0-or-later while everything else here stays MIT, so the
distribution as a whole is `MIT AND LGPL-3.0-or-later`. `LICENSE` records which
is which, `LICENSE.LGPL` and `LICENSE.GPL` carry the text, and you may modify
that module and relink it against the rest of the package under those terms.

**It is why `jax` is a dependency.** Sella differentiates its internal
coordinates rather than hand-coding the derivatives, so `jax` and `jaxlib` are
installed with the package. They are used for automatic differentiation only,
never for linear algebra, so the CPU wheels are enough and no GPU build is
needed. Compiled programs are cached under
`~/.cache/reactiontools/jax_cache`; set `JAX_COMPILATION_CACHE_DIR` to move it
if the home directory is not writable.

Upstream Sella also ships three Cython extension modules. None are built here:
two were unused, and of the third only one routine was ever called from Python,
which is reimplemented in NumPy. That is what keeps `reactiontools` a
pure-Python wheel with no build step.

### Development

The test runner and the linter are not runtime dependencies either. To run the
suite from a checkout, install the `dev` extra:

```bash
pip install -e ".[dev]"
```

### Outside pip entirely

Three more fall outside `pip` altogether, and are only needed by the functions
named:

| Dependency | Needed by | Notes |
| --- | --- | --- |
| `plumed` executable | `run_sum_hills` | Must be on `PATH`. Called as a subprocess, not imported. Compiled by `conda_install.sh`; conda-forge's `plumed` package is built without the OPES module that `f_opes=True` inputs need. |
| `py-plumed` | `plumed_calculator` | The Python bindings, compiled by `conda_install.sh` against the same PLUMED. Imported on first use; the input builder works without it. |
| [ORCA](https://www.faccts.de/orca/) | everything in `tools_orca` | Licensed separately and installed by hand; point `ORCA_PATH` at the binary. See [build_tools/README.md](https://github.com/LouieSlocombe/reactiontools/blob/main/build_tools/README.md#orca). |

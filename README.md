# reactiontools

[![PyPI](https://img.shields.io/pypi/v/reactiontools)](https://pypi.org/project/reactiontools/)
[![Python versions](https://img.shields.io/pypi/pyversions/reactiontools)](https://pypi.org/project/reactiontools/)
[![Documentation Status](https://readthedocs.org/projects/reactiontools/badge/?version=latest)](https://reactiontools.readthedocs.io/en/latest/)
[![License: MIT AND LGPL-3.0-or-later](https://img.shields.io/badge/license-MIT%20AND%20LGPL--3.0--or--later-blue)](LICENSE)

A centralised set of tools for looking at transition state (TS) and nudged
elastic band (NEB) calculations.

`reactiontools` wraps the parts of an [ASE](https://wiki.fysik.dtu.dk/ase/)
reaction-path workflow that get rewritten in every project: interpolating a
band, relaxing the endpoints, pulling the TS image out and refining it to a
true saddle point, following the IRC away from it, driving PLUMED, and
producing publication-ready figures with consistent styling.

It also holds the collective variables an enhanced-sampling run is biased
along — `tools_cv` for proton transfer, `tools_path` for turning a steered
trajectory into a `PATHMSD` reference. Those are written
as text for whatever runs PLUMED, so they are equally usable from an ASE run
here or from an OpenMM one driven by
[openmmnqe](https://github.com/LouieSlocombe/openmmnqe), which depends on this
package for exactly that.

It is calculator-agnostic — anything that behaves like an ASE calculator works,
from EMT to a machine-learned potential to a DFT code.

## Installation

```bash
pip install reactiontools
```

That covers everything. The saddle-point searches and geodesic interpolation
need nothing extra: both are part of the package, in `tools_sella` and
`tools_geodesic`. The one thing still installed separately is ORCA, which
`tools_orca` shells out to and which is licensed separately.

For everything at once, including PLUMED built with the OPES module, one
command from the repository root:

```bash
bash build_tools/conda_install.sh
```

That creates the `reactiontools` conda environment, compiles PLUMED and the
matching Python bindings into it, and installs this package in editable mode.

For the environment-only route, the dependencies that fall outside `pip`
(the `plumed` executable, `py-plumed` and ORCA), and the Sol cluster, see
[Installation](https://reactiontools.readthedocs.io/en/latest/installation.html) or
[build_tools/README.md](build_tools/README.md).

## Quickstart

Relax two endpoints, run a climbing-image NEB between them, and plot the
barrier. This example is a gold adatom hopping between hollow sites on an
Al(100) surface, and runs in a few seconds with EMT:

```python
from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from reactiontools import (
    optimise_neb,
    optimise_reactant_product,
    plot_neb,
    prepare_neb,
    summarise_neb,
)

calc = EMT()

slab = fcc100("Al", size=(2, 2, 3))
add_adsorbate(slab, "Au", 1.7, "hollow")
slab.center(axis=2, vacuum=4.0)
slab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in slab]))

reactant = slab.copy()
product = slab.copy()
product.positions[-1, 0] += product.cell[0, 0] / 2  # hop to the next site

reactant, product = optimise_reactant_product(reactant, product, calc, fmax=0.05)

# rm_ro_trans=False: the slab is periodic and constrained, so the rigid-body
# degrees of freedom are already pinned -- see the NEB guide.
neb = prepare_neb(
    reactant, product, calc, n_images=7, climb=True, rm_ro_trans=False, geo_int=False
)
images = optimise_neb(neb, fmax=0.05, ts_traj="ts.traj")

print(summarise_neb(images))

plot_neb(images, calc, filename="neb")
```

This converges in 17 NEB steps and prints:

```
Barrier:         0.374 eV
Reverse barrier: 0.374 eV
Reaction energy: 0.000 eV
TS image:        3 of 6
```

`plot_neb` writes `neb.png` and `neb.pdf`, with energies referenced to the
lowest image and reported in meV.

## Documentation

Full documentation is at **[reactiontools.readthedocs.io](https://reactiontools.readthedocs.io)**.

- [Installation](https://reactiontools.readthedocs.io/en/latest/installation.html) — environment, PLUMED, ORCA
- [Nudged elastic bands](https://reactiontools.readthedocs.io/en/latest/guide/neb.html) — settings,
  reading a band, convergence, optimisers
- [Continuing a band](https://reactiontools.readthedocs.io/en/latest/guide/restarting.html)
- [Sockets and parallel bands](https://reactiontools.readthedocs.io/en/latest/guide/parallel.html)
- [Refining the transition state](https://reactiontools.readthedocs.io/en/latest/guide/transition-states.html)
- [Building end states](https://reactiontools.readthedocs.io/en/latest/guide/end-states.html)
- [Running metadynamics](https://reactiontools.readthedocs.io/en/latest/guide/metadynamics.html)
- [PLUMED post-processing](https://reactiontools.readthedocs.io/en/latest/guide/plumed-postprocessing.html)
- [Units](https://reactiontools.readthedocs.io/en/latest/guide/units.html) — eV/Å against kJ/mol/nm,
  and the factor-of-ten trap between them
- [API reference](https://reactiontools.readthedocs.io/en/latest/api/index.html) — every public function
  and class, generated from the docstrings

## Testing

The test runner and the linter are not runtime dependencies; install the `dev`
extra to get them:

```bash
pip install -e ".[dev]"
```

```bash
pytest --cov
```

The suite builds its own structures with `ase.build` and evaluates them with
EMT. Offline unit tests cover orchestration without opening sockets; a small
set of `integration` tests exercises real local socket transport when the
runner permits it. Checks needing ORCA, PLUMED or OpenMM skip when those are
unavailable, so a plain `pip install -e ".[dev]"` runs the suite green. The
saddle-point and IRC tests always run: `tests/test_sella.py` is the upstream
Sella suite, brought across with the module it exercises. Coverage is
branch-aware and enforces the configured floor.

CI tests Python 3.12–3.14 and also tests the minimum scientific, pytest and Ruff
versions together on Python 3.12. To reproduce that environment, install with
`pip install -c .github/requirements-min.txt -e ".[dev]"` in a fresh Python 3.12
environment. Keep those constraints in step with the bounds in `pyproject.toml`.

## Citing

If `reactiontools` is useful in your work, please cite it and whichever of the
codes it wraps you actually exercised. Every entry is in
[CITATIONS.bib](CITATIONS.bib), and
[Citing](https://reactiontools.readthedocs.io/en/latest/citing.html) says which to use for what.

## License

MIT, with one exception — see [LICENSE](LICENSE).

`reactiontools/tools_sella.py` is derived from
[Sella](https://github.com/zadorlab/sella) and is **LGPL-3.0-or-later**, not
MIT, as the code it derives from is: copyright Sandia National Laboratories
(NTESS) under US Government contract DE-NA0003525. The licence travels with the
package in [LICENSE.LGPL](LICENSE.LGPL), together with
[LICENSE.GPL](LICENSE.GPL), which it incorporates by reference. You may modify
that module and relink it against the rest of the package under those terms;
its header lists the changes already made relative to upstream. Everything else
here is MIT, so the distribution as a whole is `MIT AND LGPL-3.0-or-later`.

`reactiontools/tools_geodesic.py` is derived from
[`geodesic-interpolate`](https://github.com/virtualzx-nad/geodesic-interpolate),
also MIT, copyright Xiaolei Zhu; the notice is at the foot of
[LICENSE](LICENSE). `reactiontools/opes/` holds PLUMED's OPES post-processing
scripts, which carry their own upstream licence.

# reactiontools

A centralised set of tools for looking at transition state (TS) and nudged
elastic band (NEB) calculations.

`reactiontools` wraps the parts of an [ASE](https://wiki.fysik.dtu.dk/ase/)
reaction-path workflow that get rewritten in every project: interpolating a
band, relaxing the endpoints, pulling the TS image out and refining it to a
true saddle point, following the IRC away from it, driving PLUMED, and
producing publication-ready figures with consistent styling.

It also holds the collective variables an enhanced-sampling run is biased
along — `tools_cv` for proton transfer, `tools_path` for turning a steered
trajectory into a `PATHMSD` reference. Those are written as text for whatever
runs PLUMED, so they are equally usable from an ASE run here or from an OpenMM
one driven by [openmmnqe](https://github.com/LouieSlocombe/openmmnqe), which
depends on this package for exactly that.

It is calculator-agnostic — anything that behaves like an ASE calculator works,
from EMT to a machine-learned potential to a DFT code.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download` Installation
:link: installation
:link-type: doc

One command to get PLUMED, the environment and the package in place.
:::

:::{grid-item-card} {octicon}`rocket` Quickstart
:link: quickstart
:link-type: doc

A climbing-image NEB over a real barrier, start to finish, in a few seconds.
:::

:::{grid-item-card} {octicon}`book` User guide
:link: guide/index
:link-type: doc

Choosing NEB settings, restarting bands, sockets, saddle points, metadynamics.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: api/index
:link-type: doc

Every public function and class, generated from the docstrings.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
guide/index
api/index
citing
changelog
```

```{toctree}
:hidden:
:caption: Project links

Source code <https://github.com/LouieSlocombe/reactiontools>
Issue tracker <https://github.com/LouieSlocombe/reactiontools/issues>
```

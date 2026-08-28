# Quickstart

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

From here, [Nudged elastic bands](guide/neb.md) covers the settings
`prepare_neb` was given above, how to read the relaxed band, and how to tell
whether it converged.

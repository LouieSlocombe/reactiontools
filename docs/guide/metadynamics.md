# Running metadynamics

The three stages of a metadynamics run, in order: pick the atoms, build the
input, bias the dynamics with it, sum the hills.

```python
from ase import units
from ase.md.langevin import Langevin
from reactiontools import (
    find_molecules,
    plot_plumed,
    plumed_calculator,
    plumed_metad_input,
    plumed_selection,
    run_sum_hills,
)

solute, solvent = find_molecules(atoms)[:2]

lines = plumed_metad_input(
    cvs=[
        f"c1: COORDINATION GROUPA={plumed_selection(solute)} "
        f"GROUPB={plumed_selection(solvent)} R_0=3.0"
    ],
    sigma=0.05,
    height=0.02,
    pace=100,
    biasfactor=10,
    temperature=300,
)

with plumed_calculator(atoms, calc, lines, timestep=0.5 * units.fs, temperature=300):
    Langevin(atoms, 0.5 * units.fs, temperature_K=300, friction=0.01).run(100000)

run_sum_hills()
plot_plumed("fes.dat", x_label="Coordination")
```

`plumed_calculator` wraps your calculator so the integrator asks for forces as
usual and PLUMED adds the bias on top. It is a context manager because PLUMED
buffers what it writes and only flushes when finalised: run the dynamics inside
the block, or `HILLS` ends short of the hills actually deposited and the surface
summed from it is quietly wrong rather than obviously missing. The block puts
your original calculator back on the way out, exceptions included.

Three things to keep in step, none of which is checked for you:

- **The timestep** must be the one the integrator uses, since PLUMED counts its
  own steps from it.
- **The temperature** goes to `plumed_metad_input` as `TEMP`, to
  `plumed_calculator` as the thermal energy, and to the thermostat. All three
  should agree.
- **`biasfactor`** makes it well-tempered. Without one the Gaussians never stop
  piling up and there is no converged surface to read off.

`plumed_metad_input` writes `UNITS ENERGY=eV LENGTH=A TIME=fs` first, so
`sigma`, `height` and everything PLUMED writes back are in Å and eV rather than
PLUMED's own nm and kJ/mol. That is what makes `plot_plumed` right about the
file it reads. `PLUMED_ASE_UNITS` is the same line, for hand-written input.

`METAD` settings without their own argument go through `metad_extra` — the grid
ones matter for a long run — and further actions through `extra`:

```python
lines = plumed_metad_input(
    cvs=["d1: DISTANCE ATOMS=1,2"],
    sigma=0.05,
    height=0.02,
    pace=100,
    biasfactor=10,
    temperature=300,
    metad_extra="GRID_MIN=1.0 GRID_MAX=6.0 GRID_BIN=500",
    extra=["UPPER_WALLS ARG=d1 AT=6.0 KAPPA=100.0"],
)
```

CV lines go through untouched, so any PLUMED action works — and any mistake in
one is PLUMED's to report. It checks them as the calculator is built, before
the block is entered and so before anything is written, which is why a missing
`R_0` above comes back as `keyword R_0 is compulsory for this action` rather
than as a run that goes nowhere.

Only `plumed_calculator` needs the plumed Python module, and only
`run_sum_hills` needs the `plumed` executable. The rest is string handling.

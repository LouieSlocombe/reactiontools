# PLUMED post-processing

Build a selection string, sum the hills, and plot the free-energy surface:

```python
from ase.io import read
from reactiontools import find_molecules, plumed_selection, run_sum_hills, plot_plumed

atoms = read("system.xyz")
solute = find_molecules(atoms)[0]
print(plumed_selection(solute))  # e.g. "1-9,14" — one-based, as PLUMED expects

run_sum_hills(hills="HILLS", outfile="fes.dat", mintozero=True)
plot_plumed("fes.dat", x_label="CV (Å)")
```

Comparing several runs laid out as `<run>/fes.dat` — pass the parent directory
and every surface beneath it is found and labelled by run name:

```python
from reactiontools import plot_plumed_multi

plot_plumed_multi("runs/", mintozero=True, x_label="CV (Å)")
```

## Reading the numbers off a surface

`summarise_fes` measures a 1-D profile: the barrier each way, and the free
energy of one basin relative to the other. Give it the two basins as windows on
the collective variable — reading them off the plot is the point:

```python
from reactiontools import summarise_fes

print(
    summarise_fes("fes.dat", basin_a=(1.0, 2.0), basin_b=(3.0, 4.5), source_unit="eV")
)
```

```
Barrier A->B:  0.352 eV
Barrier B->A:  0.418 eV
Delta F (B-A): -0.066 eV
Minima at:     1.5, 3.75
Barrier at:    2.6
```

Windows rather than automatic minimum-finding, because a noisy surface has
local minima everywhere, and because a convergence series only means anything
if every surface is measured the same way.

By default `delta_f` is the difference between the two minima. Pass a
`temperature` and it comes instead from integrating the Boltzmann weight across
each basin, which counts a wide basin as more probable than a narrow one of the
same depth:

```python
summarise_fes(
    "fes.dat", basin_a=(1.0, 2.0), basin_b=(3.0, 4.5), source_unit="eV", temperature=300
)
```

The barriers are measured out of the bottom of each well either way, since that
is what a barrier is.

## Has it converged?

`stride` sums the hills every so often instead of once at the end, giving a
series of surfaces. The run has converged when the last few lie on top of each
other:

```python
from reactiontools import plot_fes_1d, run_sum_hills, sum_hills_files

run_sum_hills(stride=100, outfile="fes", grid_min=1.0, grid_max=6.0, grid_bin=500)

surfaces = sum_hills_files("fes")
plot_fes_1d(
    surfaces,
    source_unit="eV",
    max_datasets=5,
    labels=[(i + 1) * 100 for i in range(len(surfaces))],
    label_template="{:g} hills",
)
```

Set the grid explicitly for a series. Without it PLUMED picks bounds per
surface from the hills it has so far, and the surfaces come back on grids that
do not line up.

Surfaces lying on top of each other is the loose version of the test. The one
that matters is whether the numbers have stopped moving, which
`plot_fes_convergence` draws directly — the barrier and the basin difference
against time:

```python
from reactiontools import fes_convergence, plot_fes_convergence

hills = [(i + 1) * 100 for i in range(len(surfaces))]
plot_fes_convergence(
    surfaces,
    basin_a=(1.0, 2.0),
    basin_b=(3.0, 4.5),
    times=hills,
    source_unit="eV",
    filename="convergence",
)
```

`fes_convergence` returns the underlying `FESSummary` per surface if you want
the numbers rather than the picture.

`sum_hills_files` exists because the naming is a trap. `--stride` does not
number the file it was given — it writes `f"{outfile}{n}.dat"`, so the default
`outfile="fes.dat"` yields `fes.dat0.dat`, and the obvious glob `fes*.dat` is
right only by accident. Worse, sorting those names puts `fes.dat10.dat` before
`fes.dat2.dat`, which for a convergence series silently scrambles the very
thing being plotted. `sum_hills_files` orders them by the index PLUMED gave
them. Passing `outfile="fes"` at least gets you `fes0.dat`.

To project a surface onto some of its variables, name the ones to keep and give
the temperature the rest are integrated out at, in the energy units of the
hills — eV for a run built by `plumed_metad_input`:

```python
from ase.units import kB

run_sum_hills(idw="d1", kt=kB * 300, outfile="fes_d1.dat")
```

There has to be something left to integrate out, so this needs a run biased on
two variables or more. On a one-variable `HILLS` PLUMED does not report a
usable error — it aborts the process on a failed internal assertion,
`proj.size()<getNumberOfArguments()`, which surfaces here as a
`CalledProcessError` with a signal rather than an exit code.

Anything without its own keyword goes through `extra`, for example
`extra=["--fmt", "%14.9f"]`.

## Combining independent walkers

Several independent OPES runs of the same system — walkers — are combined at
analysis time by reweighting: every `COLVAR` row carries the bias its own
walker collected it under, so the merged samples reweight as one. Merge sorted
by time to treat them as a single series (what `stride` and `skiprows`
expect), or unsorted to keep each walker contiguous and get cross-walker error
bars from `blocks`:

```python
from reactiontools import combine_colvar_files, run_opes_reweighting

combine_colvar_files(
    [f"walker_{i:03d}/COLVAR" for i in range(4)], sort_by_time=False
)
run_opes_reweighting(
    sigma=0.1, kt=2.494, cv="phi",
    grid_min=-3.14, grid_max=3.14, grid_bin=100, blocks=4,
)
```

For a single run, `run_opes_reweighting` is instead a cross-check on
`run_opes_fes`: one surface from the samples, one from the bias's own running
estimate, and daylight between them means the run is not converged.

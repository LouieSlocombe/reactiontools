# Sockets and parallel bands

Both ways of avoiding a fresh start-up on every force call: driving one
calculator over a socket so it stays alive between steps, and spreading the
interior images of a band across a pool of them.

## Minimising with a socket calculator

`optimise_geom` and `optimise_reactant_product` take the same `use_socket`,
`socket_port`, `socket_unixsocket` and `socket_log` arguments. With
`use_socket=True`, `calc` is driven through an ASE
[`SocketIOCalculator`](https://wiki.fysik.dtu.dk/ase/ase/calculators/socketio/socketio.html)
instead of being called directly, so the external program launches once and
stays running for every BFGS step instead of restarting on each one:

```python
reactant, product = optimise_reactant_product(
    reactant, product, calc, fmax=0.05, use_socket=True
)
```

This needs a calculator ASE knows how to launch as an i-PI client — built-in
support covers `Espresso`, `Aims` and `Siesta`. A calculator without that
support, such as EMT, will fail with `use_socket=True`.

Already have a band on disk? Read it back and plot it directly — or hand it to
`restart_neb` and carry on relaxing it:

```python
from ase.calculators.emt import EMT
from ase.io import read
from reactiontools import plot_images, plot_neb

images = read("ts.traj", index="-7:")  # the last band written by optimise_neb
plot_images(images, view="side", n_cols=4, save=True)
plot_neb(images, EMT(), smooth=True)
```

Images read back from a trajectory already carry their energies, so `plot_neb`
reuses them and only falls back to the calculator you pass for images that have
none. One thing they do not carry is `info["converged"]`, which is written onto
the images `optimise_neb` returns rather than into the trajectory itself.

## Running the images in parallel

`prepare_neb` walks the band one image at a time, so a seven-image band costs
five sequential energy evaluations per step. That is fine for EMT and painful
for DFT. `prepare_parallel_neb` gives each interior image its own
[socket calculator](https://ase-lib.org/ase/calculators/socketio/socketio.html)
and asks ASE to spread the band over threads. Each thread blocks waiting on its
own socket while the external code works, which releases the GIL, so the
calculations genuinely overlap and a step costs about as much as its slowest
image:

```python
from ase.calculators.espresso import Espresso

from reactiontools import get_ts_image, optimise_neb, prepare_parallel_neb


def make_calc(index):
    # One directory per client: file-based codes write their input and output
    # relative to calc.directory, and clients sharing one overwrite each other.
    return Espresso(directory=f"image-{index}", pseudopotentials=...)


with prepare_parallel_neb(reactant, product, make_calc, n_images=7, timeout=600) as neb:
    images = optimise_neb(neb, fmax=0.05, ts_traj="ts.traj")

# The sockets are shut by now, but the band read back from ts.traj carries its
# energies, so get_ts_image and plot_neb need no calculator.
ts = get_ts_image(images)
```

It is a context manager because the sockets and the client processes behind
them have to be shut down, including when the band blows up partway through.
Optimise inside the block; once it exits the calculators are closed.

Three things to know:

- **Run it as a single process.** The parallelism is threads and sockets, so
  give the ranks to the clients, not to the driver. Under `mpirun` ASE would
  distribute the images over MPI ranks instead and every rank would try to bind
  the same sockets, so this raises rather than hanging.
- **Only the interior images get sockets.** The endpoints are evaluated once
  and pinned, reusing the energy their calculator already holds — normally the
  one `optimise_reactant_product` left behind — and otherwise pricing them
  through the first socket. Two more clients idling all run for a pair of
  energies that are already known is not worth it.
- **Set a `timeout`.** Without one, a client that dies without closing its
  socket hangs the run forever rather than raising. `log="socket"` writes the
  i-PI traffic to `socket-0.log`, `socket-1.log`, … and is the first thing to
  reach for when a run stalls with no output.

Sockets are named `/tmp/ipi_reactiontools-<pid>-<image>` by default, so
concurrent jobs on one node do not collide. Pass `unixsocket="..."` to choose
the prefix, or `port=31415` to use TCP ports counting up from there instead.

To drive a Python calculator — an ML potential, say — in separate processes
rather than an external binary, pass `make_launcher` instead of `make_calc`:

```python
from ase.calculators.socketio import PySocketIOClient

with prepare_parallel_neb(
    reactant,
    product,
    None,
    make_launcher=lambda index: PySocketIOClient(MyMLIP),
    n_images=7,
    timeout=600,
) as neb:
    images = optimise_neb(neb, fmax=0.05)
```

`socket_calculators` is the same machinery without the band, for when you want
a pool of socket calculators for something else:

```python
from reactiontools import optimise_reactant_product, socket_calculators

with socket_calculators(1, make_calc) as (calc,):
    reactant, product = optimise_reactant_product(reactant, product, calc)
```

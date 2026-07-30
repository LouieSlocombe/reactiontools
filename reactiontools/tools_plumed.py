import subprocess

import numpy as np
from ase.neighborlist import build_neighbor_list
from scipy.sparse.csgraph import connected_components


def plumed_selection(indices):
    """Format atom indices as a PLUMED ``ATOMS=`` selection string.

    Parameters
    ----------
    indices : iterable of int
        Zero-based atom indices.

    Returns
    -------
    str
        Comma-separated PLUMED selection using one-based indexing and compact
        ranges.
    """
    idx = sorted({int(i) + 1 for i in indices})
    if not idx:
        raise ValueError("empty atom selection")
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    runs.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def find_molecules(atoms):
    """Return connected atom groups identified as molecules.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure for which the bonded graph should be analysed.

    Returns
    -------
    list of numpy.ndarray
        Atom-index arrays, one per connected component.
    """
    nl = build_neighbor_list(atoms, self_interaction=False, bothways=True)
    n, labels = connected_components(nl.get_connectivity_matrix(sparse=True),
                                     directed=False)
    return [np.where(labels == k)[0] for k in range(n)]


def run_sum_hills(hills="HILLS",
                  outfile="fes.dat",
                  mintozero=True,
                  verbose=True):
    """Run ``plumed sum_hills`` to build a free-energy surface from the hills.

    The paths are resolved by the plumed executable, so this acts on the
    current working directory unless absolute paths are given.

    Parameters
    ----------
    hills : str or path-like, optional
        Hills file written by the ``METAD`` action.
    outfile : str or path-like, optional
        Free-energy surface file to write, as read by
        :func:`~bghbn.tools_plotting.plot_plumed`.
    mintozero : bool, optional
        Pass ``--mintozero`` so the surface minimum sits at zero.
    verbose : bool, optional
        Print the command being run.

    Returns
    -------
    str
        The command line that was run.

    Raises
    ------
    subprocess.CalledProcessError
        If plumed exits non-zero.
    """
    cmd_str = f"plumed sum_hills --hills {hills} --outfile {outfile}"
    if mintozero:
        cmd_str += " --mintozero"

    if verbose:
        print(f"Running: {cmd_str}", flush=True)

    subprocess.run(
        cmd_str,
        shell=True,
        executable="/bin/bash",
        check=True,
    )
    return cmd_str

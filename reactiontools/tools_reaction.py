import copy
from pathlib import Path

import geodesic_interpolate as gi
import numpy as np
from ase.io import read
from ase.mep import NEB
from ase.optimize import BFGS
from scipy.interpolate import CubicSpline


def get_neb_path(images):
    """Compute the cumulative reaction-path distance for NEB images.

    Parameters
    ----------
    images : sequence of ase.Atoms
        Images along a path.

    Returns
    -------
    numpy.ndarray
        Cumulative distance coordinate starting at zero.
    """
    positions = [atoms.positions for atoms in images]
    path = [0] + [np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]
    return np.cumsum(path)


def stitch_path(path1, path2, f_reverse_path=False):
    """Join two reaction paths into a single IRC-like sequence.

    Parameters
    ----------
    path1 : sequence of ase.Atoms
        First path, typically the reactant side.
    path2 : sequence of ase.Atoms
        Second path, typically the product side.
    f_reverse_path : bool, optional
        Reverse the stitched path before returning it.

    Returns
    -------
    list
        Concatenated path.
    """
    irc = list(path1)[::-1] + list(path2)[1:]
    if f_reverse_path:
        irc = irc[::-1]
    return irc


def resample_path(path, n_resample):
    """Resample a path to a fixed number of images using cubic splines.

    Parameters
    ----------
    path : sequence of ase.Atoms
        Path to resample.
    n_resample : int
        Number of images in the resampled path.

    Returns
    -------
    list of ase.Atoms
        Resampled path with the first and last images preserved.
    """
    path_distance = get_neb_path(path)
    path_interp = np.linspace(0, path_distance[-1], n_resample)
    positions = np.array([image.positions for image in path])
    positions_interp = CubicSpline(path_distance, positions)(path_interp)
    irc_resampled = [path[0]]
    for ii in range(1, n_resample - 1):
        atoms = path[0].copy()
        atoms.positions = positions_interp[ii, :, :]
        irc_resampled.append(atoms)
    irc_resampled.append(path[-1])
    return irc_resampled


def optimise_geom(atoms, calc,
                  fmax=0.01,
                  steps=1000,
                  opti_traj='opti.traj'):
    """Relax a structure with BFGS and return the final image.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to optimise.
    calc : ase.calculators.Calculator
        Calculator attached during the optimisation.
    fmax : float, optional
        Maximum force criterion in eV/Å.
    steps : int, optional
        Maximum number of optimiser steps.
    opti_traj : str, optional
        Temporary trajectory filename used to store the optimisation.

    Returns
    -------
    ase.Atoms
        Relaxed structure.
    """
    atoms = atoms.copy()
    atoms.calc = calc
    BFGS(atoms, trajectory=opti_traj).run(fmax=fmax, steps=steps)
    atoms = read(opti_traj, index=-1)
    Path(opti_traj).unlink()
    atoms.calc = calc
    return atoms


def optimise_reactant_product(reactant, product, calc,
                              fmax=0.01,
                              steps=1000,
                              reactant_opti='reactant_opti.traj',
                              product_opti='product_opti.traj'):
    """Optimise reactant and product structures independently.

    Parameters
    ----------
    reactant : ase.Atoms
        Reactant structure.
    product : ase.Atoms
        Product structure.
    calc : ase.calculators.Calculator
        Calculator used for both optimisations.
    fmax : float, optional
        Maximum force criterion in eV/Å.
    steps : int, optional
        Maximum number of optimiser steps.
    reactant_opti : str, optional
        Temporary trajectory filename for the reactant optimisation.
    product_opti : str, optional
        Temporary trajectory filename for the product optimisation.

    Returns
    -------
    tuple of ase.Atoms
        Optimised reactant and product structures.
    """
    print('Optimising reactant...', flush=True)
    reactant = optimise_geom(reactant, calc,
                             fmax=fmax,
                             steps=steps,
                             opti_traj=reactant_opti)

    print('Optimizing product...', flush=True)
    product = optimise_geom(product, calc,
                            fmax=fmax,
                            steps=steps,
                            opti_traj=product_opti)
    return reactant, product


def prepare_neb(reactant, product, calc,
                n_images=5,
                climb=True,
                rm_ro_trans=True,
                geo_int=True,
                k=2.0):
    """Build an ASE NEB object from reactant and product end states.

    Parameters
    ----------
    reactant : ase.Atoms
        Initial state.
    product : ase.Atoms
        Final state.
    calc : ase.calculators.Calculator
        Calculator copied onto each NEB image.
    n_images : int, optional
        Total number of images, including endpoints.
    climb : bool, optional
        Enable the climbing-image NEB variant.
    rm_ro_trans : bool, optional
        Remove rigid-body rotation and translation during interpolation.
    geo_int : bool, optional
        Use geodesic interpolation before NEB construction.
    k : float, optional
        Spring constant passed to ASE's NEB.

    Returns
    -------
    ase.mep.NEB
        Configured NEB object.
    """
    neb_images = [reactant]
    for ii in range(n_images - 2):
        neb_images.append(reactant.copy())
    neb_images.append(product)

    if geo_int:
        neb_images = gi.geodesic_interpolate(neb_images, n_images=n_images)

    for image in neb_images:
        image.calc = copy.copy(calc)
        image.get_potential_energy()

    neb = NEB(neb_images,
              climb=climb,
              remove_rotation_and_translation=rm_ro_trans,
              k=k,
              method='improvedtangent')
    if not geo_int:
        neb.interpolate()
        neb.interpolate("idpp")
    return neb


def optimise_neb(neb,
                 fmax=0.01,
                 steps=1000,
                 ts_traj='ts.traj'):
    """Optimise an NEB band and return the final trajectory images.

    Parameters
    ----------
    neb : ase.mep.NEB
        NEB object to optimise.
    fmax : float, optional
        Maximum force criterion in eV/Å.
    steps : int, optional
        Maximum number of optimiser steps.
    ts_traj : str, optional
        Output trajectory filename.

    Returns
    -------
    list of ase.Atoms
        Final NEB images read back from ``ts_traj``.
    """
    n_images = len(neb.images)
    BFGS(neb, trajectory=ts_traj).run(fmax=fmax, steps=steps)
    return read(ts_traj, index=f"-{n_images}:")


def get_ts_image(neb_images, calc):
    """Return the highest-energy image along a NEB band.

    Parameters
    ----------
    neb_images : sequence of ase.Atoms
        Images along the band.
    calc : ase.calculators.Calculator
        Calculator used to evaluate the potential energies.

    Returns
    -------
    ase.Atoms
        Image with the maximum potential energy.
    """
    for image in neb_images:
        image.calc = copy.copy(calc)
    index = np.argmax([image.get_potential_energy() for image in neb_images])
    return neb_images[index]


def quick_guess_path(reactant, product, n_images=25):
    """Generate a quick geodesic path guess between two endpoints.

    Parameters
    ----------
    reactant : ase.Atoms
        Starting structure.
    product : ase.Atoms
        Ending structure.
    n_images : int, optional
        Number of images in the interpolated path.

    Returns
    -------
    list of ase.Atoms
        Interpolated path.
    """
    return gi.geodesic_interpolate([reactant, product], n_images=n_images)


def quick_guess_ts(reactant, product, n_images=25):
    """Return the midpoint image from a quick geodesic path guess.

    Parameters
    ----------
    reactant : ase.Atoms
        Starting structure.
    product : ase.Atoms
        Ending structure.
    n_images : int, optional
        Number of images used in the interpolated path.

    Returns
    -------
    ase.Atoms
        Midpoint image of the interpolated path.
    """
    atoms_ts = gi.geodesic_interpolate([reactant, product], n_images=n_images)
    atoms_ts = atoms_ts[n_images // 2]
    return atoms_ts

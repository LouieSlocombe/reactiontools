import copy
import matplotlib.pyplot as plt
import numpy as np
import warnings
from ase import Atoms
from ase.io import read
from ase.visualize.plot import plot_atoms
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from scipy.interpolate import make_interp_spline
from typing import Any

from .tools_reaction import get_neb_path

C_CYCLE = ("#D4447E", "#2F3E56", "#5FABA2", "#E9A66C", "#7B6CA8", "#9AA5B1")

# Named camera angles, expressed as ASE rotation strings
_ATOM_VIEWS = {"top": "0x,0y,0z",
               "side": "-90x,0y,0z",
               "front": "-90x,-90y,0z",
               "tilted": "300x,0y,0z"}

# Setting plot aesthetics for better visibility
plt.rcParams['axes.linewidth'] = 2.0


def n_plot(xlab: str,
           ylab: str,
           xs: int = 14,
           ys: int = 14
           ) -> None:
    """Apply a consistent styling to the current matplotlib axes.

    This convenience function configures tick visibility, sizes and label
    fonts for the current pyplot axes and sets x/y labels.

    Parameters
    ----------
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    xs : int, optional
        Font size for axis labels on the x-axis (default 14).
    ys : int, optional
        Font size for axis labels on the y-axis (default 14).

    Returns
    -------
    None
    """
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    return None


def ax_plot(fig: Figure,
            ax: Axes,
            xlab: str,
            ylab: str,
            xs: int = 14,
            ys: int = 14
            ) -> None:
    """Apply a consistent styling to a specific matplotlib Axes object.

    This function mirrors `n_plot` but operates on a provided `Axes` and
    `Figure` so it can be used when creating subplots.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure instance that contains the axes.
    ax : matplotlib.axes.Axes
        Axes instance to style.
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    xs : int, optional
        Font size for axis labels on the x-axis (default 14).
    ys : int, optional
        Font size for axis labels on the y-axis (default 14).

    Returns
    -------
    None
    """
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    ax.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.set_xlabel(xlab, fontsize=xs)
    ax.set_ylabel(ylab, fontsize=ys)
    fig.tight_layout()
    return None


def plot_images(images,
                view="tilted",
                rotation=None,
                n_cols=4,
                titles=None,
                radii=0.8,
                show_unit_cell=2,
                fig_size=None,
                save=False,
                show=True,
                filename="images"):
    """Plot a series of structures as a grid of panels, one per image.

    Each panel is titled with the index of the image, which makes it easy to
    pick frames out of a trajectory, a NEB band or a set of local minima.

    Parameters
    ----------
    images : ase.Atoms or sequence of ase.Atoms
        Structures to draw. A single ``Atoms`` object is accepted and drawn as
        a one-panel figure.
    view : str, optional
        Camera angle: one of ``"top"``, ``"side"``, ``"front"`` or
        ``"tilted"``, or any ASE rotation string (e.g. ``"300x,0y,0z"``).
    rotation : str, optional
        Explicit ASE rotation string. Overrides ``view`` when given.
    n_cols : int, optional
        Number of panels per row. Clamped to the number of images.
    titles : sequence of str, optional
        Per-panel titles. Defaults to the index of each image.
    radii : float, optional
        Atomic radius scale passed to :func:`~ase.visualize.plot.plot_atoms`.
    show_unit_cell : int, optional
        Unit-cell drawing mode passed to
        :func:`~ase.visualize.plot.plot_atoms` (0 none, 1 behind, 2 in front).
    fig_size : tuple, optional
        Size of the created figure. Defaults to 3 inches per panel.
    save : bool, optional
        Save ``.png`` and ``.pdf`` outputs when ``True``.
    show : bool, optional
        Display the plot with ``plt.show()`` when ``True``.
    filename : str, optional
        Output filename stem for saved figures.

    Returns
    -------
    tuple
        ``(fig, axes)`` with ``axes`` as a flat array, one entry per panel.
    """
    if isinstance(images, Atoms):
        images = [images]
    images = list(images)
    if not images:
        raise ValueError("No images to plot.")

    if rotation is None:
        # Fall through to the raw string so any ASE rotation is accepted
        rotation = _ATOM_VIEWS.get(view, view)

    if titles is None:
        titles = [str(i) for i in range(len(images))]
    elif len(titles) != len(images):
        raise ValueError(f"Got {len(titles)} titles for {len(images)} images.")

    n_cols = max(1, min(n_cols, len(images)))
    n_rows = int(np.ceil(len(images) / n_cols))
    if fig_size is None:
        fig_size = (3.0 * n_cols, 3.0 * n_rows)

    fig, axes = plt.subplots(n_rows,
                             n_cols,
                             figsize=fig_size,
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, image, title in zip(axes, images, titles):
        plot_atoms(image,
                   ax,
                   rotation=rotation,
                   show_unit_cell=show_unit_cell,
                   radii=radii)
        ax.set_title(title, fontsize=12)
    # Blank the axes frames, including any unused panels in the last row
    for ax in axes:
        ax.set_axis_off()

    if save:
        fig.savefig(f"{filename}.png", dpi=600)
        fig.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, axes


def _get_energy(image, calc):
    if image.calc is not None:
        try:
            return image.calc.results['energy']
        except (AttributeError, KeyError):
            pass
    image.calc = copy.copy(calc)
    return image.get_potential_energy()


def plot_neb(images,
             calc,
             fig=None,
             ax=None,
             save=True,
             show=True,
             smooth=True,
             k=2,
             fig_size=(8, 3),
             filename="neb",
             label=None):
    """Plot a nudged elastic band energy profile.

    Parameters
    ----------
    images : sequence of ase.Atoms
        NEB images along the reaction path.
    calc : ase.calculators.Calculator
        Calculator used to evaluate any missing energies.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on. A new figure is created if ``None``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new axes is created if ``None``.
    save : bool, optional
        Save ``.png`` and ``.pdf`` outputs when ``True``.
    show : bool, optional
        Display the plot with ``plt.show()`` when ``True``.
    smooth : bool, optional
        Use spline interpolation for the path when ``True``.
    k : int, optional
        Spline order used when ``smooth`` is ``True``.
    fig_size : tuple, optional
        Size of the created figure.
    filename : str, optional
        Output filename stem for saved figures.
    label : str, optional
        Line label for the path curve.

    Returns
    -------
    tuple
        ``(fig, ax)`` containing the matplotlib figure and axes.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    # Use cached energies where available, fall back to calc otherwise
    energies = np.array([_get_energy(image, calc) for image in images])
    energies -= min(energies)
    energies *= 1000.0
    # Get the path
    path = get_neb_path(images)
    if smooth:
        spl = make_interp_spline(path, energies, k=k)
        path_smooth = np.linspace(min(path), max(path), 100)
        energies_smooth = spl(path_smooth)
        ax.scatter(path, energies)
        ax.plot(path_smooth, energies_smooth, '-', lw=2, label=label)
    else:
        ax.plot(path, energies, 'o-', lw=2, label=label)
    ax_plot(fig, ax, "Path (Å)", "Energy (meV)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_temperature(trajectories, labels=None, timestep=None, ax=None):
    """Plot temperature versus frame number or time for one or more trajectories.

    Parameters
    ----------
    trajectories : str or path-like or sequence
        Trajectory file paths readable by ASE.
    labels : sequence of str, optional
        Legend labels. Defaults to the filename for each trajectory.
    timestep : float, optional
        Time spacing between frames. When provided, the x-axis is in fs.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. A new figure and axes are created if ``None``.

    Returns
    -------
    tuple
        ``(fig, ax)`` containing the matplotlib figure and axes.
    """
    if isinstance(trajectories, (str, Path)):
        trajectories = [trajectories]

    if labels is None:
        labels = [Path(t).name for t in trajectories]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for traj_path, label in zip(trajectories, labels):
        frames = read(traj_path, index=":")
        temperatures = [atoms.get_temperature() for atoms in frames]

        if timestep is not None:
            x = [i * timestep for i in range(len(temperatures))]
        else:
            x = range(len(temperatures))

        ax.plot(x, temperatures, label=label)

    ax.set_xlabel(f"Time ({'fs' if timestep else 'frame'})" if timestep else "Frame")
    ax.set_ylabel("Temperature (K)")
    ax.legend()
    ax_plot(fig, ax, "Time (fs)" if timestep else "Frame", "Temperature (K)")

    return fig, ax


def plot_total_energy(trajectories, labels=None, timestep=None, ax=None):
    """Plot total energy versus frame number or time for one or more trajectories.

    Parameters
    ----------
    trajectories : str or path-like or sequence
        Trajectory file paths readable by ASE.
    labels : sequence of str, optional
        Legend labels. Defaults to the filename for each trajectory.
    timestep : float, optional
        Time spacing between frames. When provided, the x-axis is in fs.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. A new figure and axes are created if ``None``.

    Returns
    -------
    tuple
        ``(fig, ax)`` containing the matplotlib figure and axes.
    """
    if isinstance(trajectories, (str, Path)):
        trajectories = [trajectories]

    if labels is None:
        labels = [Path(t).name for t in trajectories]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for traj_path, label in zip(trajectories, labels):
        frames = read(traj_path, index=":")
        energies = [atoms.get_total_energy() for atoms in frames]

        if timestep is not None:
            x = [i * timestep for i in range(len(energies))]
        else:
            x = range(len(energies))

        ax.plot(x, energies, label=label)

    ax.legend()
    ax_plot(fig, ax, "Time (fs)" if timestep else "Frame", "Total energy (eV)")
    return fig, ax


def _load_fes(file):
    """Read a PLUMED ``fes.dat`` file, converting the free energy to meV.

    Parameters
    ----------
    file : str or path-like
        Path to a PLUMED ``fes.dat``-style file.

    Returns
    -------
    tuple of numpy.ndarray
        ``(cv, fes)`` with the collective variable and the free energy in meV.
    """
    cv, fes = np.loadtxt(file, usecols=(0, 1), unpack=True)
    return cv, fes * 1000.0


def _expand_fes_files(files):
    """Resolve run paths to a flat list of ``fes.dat`` files.

    Parameters
    ----------
    files : str or path-like or sequence
        A single path, or a sequence of them. Directories are expanded to every
        ``fes.dat`` they contain, recursively.

    Returns
    -------
    list of pathlib.Path
        Files to plot, in the order given.
    """
    if isinstance(files, (str, Path)):
        files = [files]

    resolved = []
    for file in (Path(f) for f in files):
        resolved.extend(sorted(file.glob("**/fes.dat")) if file.is_dir()
                        else [file])
    if not resolved:
        raise ValueError(f"no fes.dat files found in {files}")
    return resolved


def _fes_labels(files):
    """Derive legend labels for a set of PLUMED runs.

    Parameters
    ----------
    files : sequence of pathlib.Path
        Paths to the ``fes.dat`` file of each run.

    Returns
    -------
    list of str
        File stems, or the parent directory names when the runs share a
        filename, as they do for ``<run>/fes.dat`` layouts.
    """
    labels = [f.stem for f in files]
    if len(set(labels)) != len(labels):
        labels = [f.parent.name for f in files]
    return labels


def plot_plumed(file='fes.dat',
                fig=None,
                ax=None,
                save=True,
                show=True,
                fig_size=(8, 3),
                filename="fes",
                x_range=None,
                x_label="CV",
                ):
    """Plot a one-dimensional PLUMED free-energy surface.

    Parameters
    ----------
    file : str, optional
        Path to a PLUMED ``fes.dat``-style file.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on. A new figure is created if ``None``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new axes is created if ``None``.
    save : bool, optional
        Save ``.png`` and ``.pdf`` outputs when ``True``.
    show : bool, optional
        Display the plot with ``plt.show()`` when ``True``.
    fig_size : tuple, optional
        Size of the created figure.
    filename : str, optional
        Output filename stem for saved figures.
    x_range : tuple, optional
        Optional x-axis limits passed to ``Axes.set_xlim``.
    x_label : str, optional
        Optional x-axis label.

    Returns
    -------
    tuple
        ``(fig, ax)`` containing the matplotlib figure and axes.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    cv, fes = _load_fes(file)
    ax.plot(cv, fes, lw=2, color='black')
    if x_range is not None:
        ax.set_xlim(x_range)
    ax_plot(fig, ax, x_label, "Free Energy (meV)")
    if save:
        fig.savefig(f"{filename}.png", dpi=600)
        fig.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_plumed_multi(files,
                      labels=None,
                      fig=None,
                      ax=None,
                      save=True,
                      show=True,
                      fig_size=(8, 3),
                      filename="fes_multi",
                      x_range=None,
                      x_label='CV',
                      mintozero=False,
                      colors=C_CYCLE,
                      ):
    """Plot the free-energy surfaces of several PLUMED runs on one axes.

    Parameters
    ----------
    files : str or path-like or sequence
        Paths to PLUMED ``fes.dat``-style files, as written by
        ``plumed sum_hills``. A directory is expanded to every ``fes.dat``
        beneath it, matching the ``<run>/fes.dat`` layout of the metadynamics
        runs.
    labels : sequence of str, optional
        Legend labels. Defaults to the file stems, or to the parent directory
        names when the runs share a filename.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on. A new figure is created if ``None``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new axes is created if ``None``.
    save : bool, optional
        Save ``.png`` and ``.pdf`` outputs when ``True``.
    show : bool, optional
        Display the plot with ``plt.show()`` when ``True``.
    fig_size : tuple, optional
        Size of the created figure.
    filename : str, optional
        Output filename stem for saved figures.
    x_range : tuple, optional
        Optional x-axis limits passed to ``Axes.set_xlim``.
    x_label : str, optional
        Optional x-axis label.
    mintozero : bool, optional
        Shift each curve so its minimum sits at zero. Use when the runs were
        summed without ``--mintozero``, or to compare barrier heights directly.
    colors : sequence, optional
        Colours cycled over the runs.

    Returns
    -------
    tuple
        ``(fig, ax)`` containing the matplotlib figure and axes.
    """
    files = _expand_fes_files(files)
    if labels is None:
        labels = _fes_labels(files)

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    for i, (file, label) in enumerate(zip(files, labels)):
        cv, fes = _load_fes(file)
        if mintozero:
            fes = fes - fes.min()
        ax.plot(cv, fes, lw=2, color=colors[i % len(colors)], label=label)

    if x_range is not None:
        ax.set_xlim(x_range)
    ax.legend(frameon=False, fontsize=12)
    ax_plot(fig, ax, x_label, "Free Energy (meV)")
    if save:
        fig.savefig(f"{filename}.png", dpi=600)
        fig.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax

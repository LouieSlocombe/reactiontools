from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from ase.visualize.plot import plot_atoms
from scipy.interpolate import make_interp_spline

from .tools_fes import plot_fes_1d
# _get_energy lives with the reaction tools so summarise_neb can share it;
# re-exported here because callers and tests import it from this module.
from .tools_reaction import _get_energy, get_neb_path
# Re-exported: the styling helpers used to live here, and callers import them
# from this module.
from .tools_style import ax_plot, n_plot  # noqa: F401

C_CYCLE = ("#D4447E", "#2F3E56", "#5FABA2", "#E9A66C", "#7B6CA8", "#9AA5B1")

# Named camera angles, expressed as ASE rotation strings
_ATOM_VIEWS = {"top": "0x,0y,0z",
               "side": "-90x,0y,0z",
               "front": "-90x,-90y,0z",
               "tilted": "300x,0y,0z"}


def _save_and_show(fig, save, show, filename):
    """Write ``<filename>.png`` and ``.pdf`` when asked, then optionally show.

    This is the save interface shared by every plotter in this module: a
    boolean ``save`` plus a filename stem, always writing both formats.
    (:func:`~reactiontools.tools_style._finalise` is the other convention,
    keyed off ``filename`` alone.)

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    save : bool
        Write the two files when ``True``.
    show : bool
        Call ``plt.show()`` when ``True``.
    filename : str
        Output filename stem.
    """
    if save:
        fig.savefig(f"{filename}.png", dpi=600)
        fig.savefig(f"{filename}.pdf")
    if show:
        plt.show()


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

    _save_and_show(fig, save, show, filename)
    return fig, axes


def show_atoms(atoms,
               view="tilted",
               rotation=None,
               ax=None,
               save=True,
               show=True,
               filename="atoms"):
    """Draw one or more structures superimposed on a single axes.

    Unlike :func:`plot_images`, which gives every structure its own panel,
    this overlays them all in one frame. That is the view for seeing how far a
    band has moved: pass the images of a NEB and the drawings stack up, so the
    atoms that shift stand out against the ones that do not.

    Parameters
    ----------
    atoms : ase.Atoms or sequence of ase.Atoms
        Structure, or structures, to draw.
    view : str, optional
        Camera angle: one of ``"top"``, ``"side"``, ``"front"`` or
        ``"tilted"``, or any ASE rotation string (e.g. ``"300x,0y,0z"``).
    rotation : str, optional
        Explicit ASE rotation string. Overrides ``view`` when given.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure and axes are created if ``None``.
    save : bool, optional
        Save ``.png`` and ``.pdf`` outputs when ``True``.
    show : bool, optional
        Display the plot with ``plt.show()`` when ``True``.
    filename : str, optional
        Output filename stem for saved figures.

    Returns
    -------
    tuple
        ``(fig, ax)`` containing the matplotlib figure and axes.
    """
    if isinstance(atoms, Atoms):
        atoms = [atoms]

    if rotation is None:
        # Fall through to the raw string so any ASE rotation is accepted
        rotation = _ATOM_VIEWS.get(view, view)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    for atom in atoms:
        plot_atoms(atom, ax, rotation=rotation)
    ax.set_axis_off()

    _save_and_show(fig, save, show, filename)
    return fig, ax


def _plot_profile(images, calc, fig, ax, save, show, smooth, k, fig_size,
                  filename, label, color=None, annotate=False):
    """Draw a reaction-path energy profile in meV against path distance.

    Shared body of :func:`plot_neb` and :func:`plot_irc`, which differ only
    in their defaults and in whether the curve colour is pinned.

    Parameters
    ----------
    images, calc, fig, ax, save, show, smooth, k, fig_size, filename, label
        See :func:`plot_neb`.
    color : str or None, optional
        Colour for the curve and its markers. ``None`` leaves matplotlib's
        colour cycle in charge.
    annotate : bool, optional
        Write the forward barrier in the corner of the axes.

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
    energies *= 1000.0  # eV -> meV
    path = get_neb_path(images)
    color_kwargs = {} if color is None else {"c": color}
    if smooth:
        spl = make_interp_spline(path, energies, k=k)
        path_smooth = np.linspace(min(path), max(path), 100)
        ax.scatter(path, energies, **color_kwargs)
        ax.plot(path_smooth, spl(path_smooth), '-', lw=2, label=label,
                **color_kwargs)
    else:
        ax.plot(path, energies, 'o-', lw=2, label=label, **color_kwargs)
    if annotate:
        # Shift-invariant, so the barrier is the same whether it is measured
        # off these meV values or the raw energies summarise_neb reports.
        barrier = energies[np.argmax(energies)] - energies[0]
        ax.text(0.02, 0.95, f"$E_\\mathrm{{a}}$ = {barrier:.0f} meV",
                transform=ax.transAxes, va="top", ha="left", fontsize=12)
    ax_plot(fig, ax, "Path (Å)", "Energy (meV)")
    _save_and_show(fig, save, show, filename)
    return fig, ax


def plot_neb(images,
             calc=None,
             fig=None,
             ax=None,
             save=True,
             show=True,
             smooth=True,
             k=2,
             fig_size=(8, 3),
             filename="neb",
             label=None,
             annotate=False):
    """Plot a nudged elastic band energy profile.

    Parameters
    ----------
    images : sequence of ase.Atoms
        NEB images along the reaction path.
    calc : ase.calculators.Calculator or None, optional
        Calculator used to evaluate any missing energies. Not needed for a
        band read back by ``optimise_neb``, whose images already carry theirs.
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
    annotate : bool, optional
        Write the forward barrier in the top-left corner of the axes, in the
        meV the y-axis is drawn in. Off by default so that existing figures
        do not change. The value matches
        :attr:`~reactiontools.NebSummary.barrier` from
        :func:`~reactiontools.summarise_neb`, converted to meV.

    Returns
    -------
    tuple
        ``(fig, ax)`` containing the matplotlib figure and axes.
    """
    return _plot_profile(images, calc, fig, ax, save, show, smooth, k,
                         fig_size, filename, label, annotate=annotate)


def plot_irc(images,
             calc=None,
             fig=None,
             ax=None,
             save=True,
             show=True,
             smooth=True,
             k=2,
             fig_size=(8, 3),
             filename="irc",
             color="black",
             label=None):
    """Plot an intrinsic reaction coordinate energy profile.

    The same profile as :func:`plot_neb`, with defaults suited to an IRC: a
    single black curve rather than one of a labelled set, since an IRC is
    normally shown on its own. Feed it the two halves from
    :func:`~reactiontools.tools_reaction.optimise_irc` joined by
    :func:`~reactiontools.tools_reaction.stitch_path`.

    Parameters
    ----------
    images : sequence of ase.Atoms
        Images along the reaction path, ordered reactant to product.
    calc : ase.calculators.Calculator or None, optional
        Calculator used to evaluate any missing energies. Not needed for
        images read back from a trajectory, which already carry theirs.
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
    color : str, optional
        Colour of the curve and its markers.
    label : str, optional
        Line label for the path curve.

    Returns
    -------
    tuple
        ``(fig, ax)`` containing the matplotlib figure and axes.
    """
    return _plot_profile(images, calc, fig, ax, save, show, smooth, k,
                         fig_size, filename, label, color=color)


def _plot_trajectory_series(trajectories, labels, timestep, ax, frame_value,
                            y_lab):
    """Plot a per-frame quantity against frame number or time.

    Shared body of :func:`plot_temperature` and :func:`plot_total_energy`,
    which differ only in the quantity read from each frame.

    Parameters
    ----------
    trajectories, labels, timestep, ax
        See :func:`plot_temperature`.
    frame_value : callable
        Called on each frame's ``Atoms`` to get the value to plot.
    y_lab : str
        Y-axis label.

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
    else:
        fig = ax.get_figure()

    for traj_path, label in zip(trajectories, labels):
        frames = read(traj_path, index=":")
        values = [frame_value(atoms) for atoms in frames]

        if timestep is not None:
            x = [i * timestep for i in range(len(values))]
        else:
            x = range(len(values))

        ax.plot(x, values, label=label)

    ax.legend()
    ax_plot(fig, ax, "Time (fs)" if timestep else "Frame", y_lab)
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
    return _plot_trajectory_series(trajectories, labels, timestep, ax,
                                   lambda atoms: atoms.get_temperature(),
                                   "Temperature (K)")


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
    return _plot_trajectory_series(trajectories, labels, timestep, ax,
                                   lambda atoms: atoms.get_total_energy(),
                                   "Total energy (eV)")


#: Unit conversion applied to every ``fes.dat`` these wrappers read.  ASE works
#: in eV and that is what ``plumed sum_hills`` writes for an ASE-driven run, so
#: the surfaces are read as eV and plotted in meV.  A run driven from OpenMM
#: writes kJ/mol instead -- reach for :mod:`reactiontools.tools_fes` directly
#: for those, which is where these wrappers send the work anyway.
_FES_UNITS = {"source_unit": "eV", "energy_unit": "meV"}


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

    # shift_min_to_zero=False: sum_hills is normally run with --mintozero, so
    # shifting again here would hide a surface that was not.
    fig, ax = plot_fes_1d(file,
                          fig=fig,
                          ax=ax,
                          shift_min_to_zero=False,
                          x_lab=x_label,
                          y_lab="Free Energy (meV)",
                          lw=2,
                          color='black',
                          **_FES_UNITS)
    if x_range is not None:
        ax.set_xlim(x_range)
    _save_and_show(fig, save, show, filename)
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

    # One call per file rather than one call with every file, so each curve
    # keeps its own colour from C_CYCLE.
    for i, (file, label) in enumerate(zip(files, labels)):
        fig, ax = plot_fes_1d(file,
                              fig=fig,
                              ax=ax,
                              labels=[label],
                              shift_min_to_zero=mintozero,
                              x_lab=x_label,
                              y_lab="Free Energy (meV)",
                              lw=2,
                              color=colors[i % len(colors)],
                              **_FES_UNITS)

    if x_range is not None:
        ax.set_xlim(x_range)
    ax.legend(frameon=False, fontsize=12)
    _save_and_show(fig, save, show, filename)
    return fig, ax

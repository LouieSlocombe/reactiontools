"""Shared figure styling.

Every plot in the package goes through :func:`ax_plot`, so that a NEB profile,
a free-energy surface and an MD temperature trace all come out looking like
they belong in the same paper: inward ticks on all four sides, minor ticks on,
consistent label sizes.

This lives in its own module rather than in either plotting module because
:mod:`reactiontools.tools_plotting` and :mod:`reactiontools.tools_fes` both
need it, and the former imports the latter -- putting the styling in
``tools_plotting`` would make that circular.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["ax_plot", "n_plot"]

# Setting plot aesthetics for better visibility
plt.rcParams['axes.linewidth'] = 2.0


def n_plot(xlab,
           ylab,
           xs=14,
           ys=14):
    """
    Configures the appearance of the current matplotlib plot.

    This function sets up minor ticks, major ticks, and axis labels for the
    active pyplot axes.  It adjusts the tick parameters and applies a tight
    layout to ensure proper spacing.

    Parameters
    ----------
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    xs : int, optional
        Font size for the x-axis label (default is 14).
    ys : int, optional
        Font size for the y-axis label (default is 14).

    Returns
    -------
    None
    """
    ax_plot(plt.gcf(), plt.gca(), xlab, ylab, xs=xs, ys=ys)
    return None


def ax_plot(fig,
            ax,
            xlab,
            ylab,
            xs=14,
            ys=14):
    """
    Configures the appearance of a matplotlib plot using a given figure and axes.

    This function sets up minor ticks, major ticks, and axis labels for the provided
    matplotlib axes. It adjusts the tick parameters and applies a tight layout to
    ensure proper spacing.  The layout pass is skipped when the figure already
    manages its own layout (for example ``constrained_layout=True``), which would
    otherwise trigger a matplotlib warning.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object to configure.
    xlab : str or None
        Label for the x-axis.  ``None`` leaves the existing label untouched,
        which is what stacked panels sharing an x-axis want.
    ylab : str or None
        Label for the y-axis.  ``None`` leaves the existing label untouched.
    xs : int, optional
        Font size for the x-axis label (default is 14).
    ys : int, optional
        Font size for the y-axis label (default is 14).

    Returns
    -------
    None
    """
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    ax.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    ax.tick_params(axis='both', which='both', top=True, right=True)
    if xlab is not None:
        ax.set_xlabel(xlab, fontsize=xs)
    if ylab is not None:
        ax.set_ylabel(ylab, fontsize=ys)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    return None


def _style_axes(fig, axes, x_lab=None, y_lab=None, xs=14, ys=14):
    """
    Apply :func:`ax_plot` styling to one or more axes.

    Only the left-most axes keeps the y-label so that shared-axis panels do
    not repeat it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure owning *axes*.
    axes : matplotlib.axes.Axes or sequence of matplotlib.axes.Axes
        Axes to style.
    x_lab, y_lab : str or None, optional
        Axis labels.  ``None`` leaves the existing label untouched.
    xs, ys : int, optional
        Label font sizes.

    Returns
    -------
    None
    """
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax_plot(fig, ax, x_lab, y_lab if i == 0 else None, xs=xs, ys=ys)
    return None


def _finalise(fig, filename=None, show=False, dpi=600, formats=("png", "pdf")):
    """
    Optionally save and/or display a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.  Saving goes through the figure itself rather than
        ``pyplot``, so the correct figure is written when several are open.
    filename : str or None, optional
        Output path.  ``None`` (default) writes nothing.  A name carrying an
        extension is written in that format only; a bare stem is written once
        per entry in *formats*.
    show : bool, optional
        Whether to call ``plt.show()`` afterwards (default is False).
    dpi : int, optional
        Resolution used for raster formats (default is 600).
    formats : sequence of str, optional
        Extensions used when *filename* has none (default ``("png", "pdf")``).

    Returns
    -------
    None
    """
    if filename:
        stem, ext = os.path.splitext(str(filename))
        extensions = [ext.lstrip(".")] if ext else list(formats)
        for extension in extensions:
            fig.savefig(f"{stem}.{extension}", dpi=dpi)
    if show:
        plt.show()
    return None

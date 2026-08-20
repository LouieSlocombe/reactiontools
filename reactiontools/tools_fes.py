"""Plotting helpers for free-energy surfaces (FES) and other PLUMED output.

The module is organised in three layers so that anything that can be turned
into a free-energy surface can be plotted by the same handful of functions:

1. **Readers** -- :func:`read_plumed_file` parses any PLUMED-style file
   (``COLVAR``, ``fes.dat``, ``HILLS``, ``FES_from_State.py`` output) into a
   :class:`PlumedData` container of numeric columns, field names and
   ``#! SET`` metadata.
2. **Container** -- :class:`FES` normalises 1-D and 2-D free-energy data into
   a common form (collective-variable grids, energies, labels). Anything a
   user is likely to have -- a file path, a ``(2, N)``/``(N, 2)`` array, a
   stacked ``(3, ny, nx)`` array, a ``(x, y, Z)`` tuple or scattered
   ``(N, 3)`` columns -- is accepted by :func:`as_fes`.
3. **Plotters** -- :func:`plot_fes_1d`, :func:`plot_fes_2d`,
   :func:`plot_fes_path`, :func:`plot_fes_2d_overlay` and
   :func:`plot_fes_slices` cover profiles, surfaces, paths through CV space
   and comparisons. :func:`plot_fes` dispatches on dimensionality when the
   caller does not care.

Every plotting function shares the same conventions:

* sources may be mixed and matched (paths, arrays, :class:`FES` objects),
* ``energy_unit`` converts the file/array energies on the way in,
* ``max_energy`` masks poorly sampled regions instead of letting them
  dominate the colour scale,
* ``filename=None`` means *do not write anything*; passing a name without an
  extension writes every format in ``formats``,
* the return value is always ``(fig, ax)``.

Energies are assumed to be in kJ/mol unless told otherwise, because that is
what PLUMED writes when it is driven from OpenMM. Runs driven from ASE are in
eV instead, so pass ``source_unit="eV"`` -- which is what the thin wrappers
:func:`~reactiontools.tools_plotting.plot_plumed` and
:func:`~reactiontools.tools_plotting.plot_plumed_multi` do.

The units themselves -- :data:`~reactiontools.tools_units.ENERGY_UNITS`,
:func:`~reactiontools.tools_units.convert_energy` and
:func:`~reactiontools.tools_units.unit_label` -- live in
:mod:`reactiontools.tools_units` and are re-exported here, so that the script
builders can convert energies without importing matplotlib through this
module.
"""

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.contour import ContourSet
from matplotlib.figure import Figure

from .tools_style import _finalise, _style_axes, ax_plot
from .tools_units import (
    DEFAULT_ENERGY_UNIT,
    ENERGY_UNITS,
    convert_energy,
    thermal_energy,
    unit_label,
)

__all__ = [
    "DEFAULT_ENERGY_UNIT",
    "ENERGY_UNITS",
    "FES",
    "FESSummary",
    "PlumedData",
    "as_fes",
    "convert_energy",
    "fes_convergence",
    "fes_series_files",
    "load_fes_series",
    "plot_fes",
    "plot_fes_1d",
    "plot_fes_2d",
    "plot_fes_2d_overlay",
    "plot_fes_convergence",
    "plot_fes_path",
    "plot_fes_slices",
    "plot_plumed_colvar",
    "plot_plumed_fes",
    "read_plumed_file",
    "summarise_fes",
    "unit_label",
]


# ---------------------------------------------------------------------------
# PLUMED file reading
# ---------------------------------------------------------------------------
@dataclass
class PlumedData:
    """Container for the contents of a PLUMED-style data file.

    Attributes
    ----------
    data : numpy.ndarray
        Numeric columns with shape ``(n_rows, n_fields)``.
    fields : list of str
        Column names taken from the ``#! FIELDS`` header. Empty when the
        file carries no header.
    metadata : dict
        Key/value pairs collected from ``#! SET key value`` lines.
    """

    data: np.ndarray
    fields: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def index(self, name: str | int) -> int:
        """Return the column index of a field.

        Parameters
        ----------
        name : str or int
            Field name, or an integer index which is returned unchanged
            after bounds checking.

        Returns
        -------
        int
            Index of the requested column.

        Raises
        ------
        KeyError
            If the named field is not present.
        IndexError
            If an integer index is out of range.
        """
        if isinstance(name, (int, np.integer)):
            index = int(name)
            if not -self.data.shape[1] <= index < self.data.shape[1]:
                raise IndexError(
                    f"Column {index} out of range for {self.data.shape[1]} columns"
                )
            return index % self.data.shape[1]
        if name not in self.fields:
            raise KeyError(f"Field {name!r} not found. Available fields: {self.fields}")
        return self.fields.index(name)

    def column(self, name: str | int) -> np.ndarray:
        """Return a single column by field name or index.

        Parameters
        ----------
        name : str or int
            Field name or column index.

        Returns
        -------
        numpy.ndarray
            The requested column.
        """
        return self.data[:, self.index(name)]

    def label(self, index: int, default: str = "") -> str:
        """Return the field name of a column, falling back to *default*.

        Parameters
        ----------
        index : int
            Column index.
        default : str, optional
            Label used when the file carried no ``#! FIELDS`` header.

        Returns
        -------
        str
            The field name or *default*.
        """
        return self.fields[index] if index < len(self.fields) else default

    def to_dataframe(self) -> pd.DataFrame:
        """Return the data as a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            Columns are named after the fields, or ``col0``, ``col1``, ...
            when no header was present.
        """
        names = list(self.fields) or [f"col{i}" for i in range(self.data.shape[1])]
        return pd.DataFrame(self.data, columns=names)


def read_plumed_file(path: str | Path, drop_der: bool = True) -> PlumedData:
    """Read a PLUMED-style data file.

    Handles the files produced by ``plumed sum_hills``, ``PRINT``/``COLVAR``
    output and the bundled OPES ``FES_from_*`` scripts: the ``#! FIELDS``
    header gives the column names, ``#! SET`` lines are collected as
    metadata, and blank block separators (used between rows of a 2-D grid)
    are ignored.

    Parameters
    ----------
    path : str
        Path to the PLUMED data file.
    drop_der : bool, optional
        Whether to discard derivative columns whose field name starts with
        ``der_``. Ignored when the header and the data do not agree on the
        number of columns.

    Returns
    -------
    PlumedData
        The parsed file contents.

    Raises
    ------
    ValueError
        If the file contains no numeric rows.
    """
    fields: list[str] = []
    metadata: dict[str, str] = {}
    numeric_lines: list[str] = []

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("#!"):
                parts = stripped.split()
                if len(parts) >= 3 and parts[1] == "FIELDS":
                    fields = parts[2:]
                elif len(parts) >= 4 and parts[1] == "SET":
                    metadata[parts[2]] = " ".join(parts[3:])
                continue

            if stripped.startswith(("#", "@")):
                continue

            numeric_lines.append(stripped)

    if not numeric_lines:
        raise ValueError(f"No numeric data found in {path}")

    data = np.loadtxt(numeric_lines, ndmin=2)

    # Only trust the header when it lines up with the data.
    if fields and len(fields) == data.shape[1]:
        if drop_der:
            keep = [i for i, name in enumerate(fields) if not name.startswith("der_")]
            data = data[:, keep]
            fields = [fields[i] for i in keep]
    else:
        fields = []

    return PlumedData(data=data, fields=fields, metadata=metadata)


# ---------------------------------------------------------------------------
# Free-energy surface container
# ---------------------------------------------------------------------------
@dataclass
class FES:
    """A free-energy surface in a form the plotting functions understand.

    Attributes
    ----------
    cvs : list of numpy.ndarray
        One entry per collective variable. For a 1-D surface this is a
        single 1-D array; for a 2-D surface on a regular grid the two arrays
        have the same shape as *energy*; for scattered 2-D data they are
        flat coordinate arrays.
    energy : numpy.ndarray
        Free energies, shaped like the entries of *cvs*.
    cv_labels : list of str
        Axis label for each collective variable.
    energy_unit : str or None
        Unit *energy* is expressed in, or None when unknown.
    energy_label : str or None
        Explicit colour-bar/y-axis label. Derived from *energy_unit* when
        left as None.
    regular : bool
        True when the data lies on a regular grid and can be drawn with
        ``contourf``; False when it must be triangulated.
    """

    cvs: list[np.ndarray]
    energy: np.ndarray
    cv_labels: list[str] = field(default_factory=list)
    energy_unit: str | None = None
    energy_label: str | None = None
    regular: bool = True

    def __post_init__(self) -> None:
        """Coerce and validate the arrays, then name unlabelled variables.

        Raises
        ------
        ValueError
            If there is not one or two CVs, the energies are empty, a CV
            shape does not match the energy shape, the dimensionality does
            not suit the layout, or the CV labels are miscounted.
        """
        self.cvs = [np.asarray(cv, dtype=float) for cv in self.cvs]
        self.energy = np.asarray(self.energy, dtype=float)
        if len(self.cvs) not in (1, 2):
            raise ValueError(f"FES requires one or two CVs, got {len(self.cvs)}")
        if self.energy.size == 0:
            raise ValueError("FES energy data cannot be empty")
        if any(cv.shape != self.energy.shape for cv in self.cvs):
            shapes = [cv.shape for cv in self.cvs]
            raise ValueError(
                f"CV shapes {shapes} must match energy shape {self.energy.shape}"
            )

        expected_ndim = 1 if len(self.cvs) == 1 or not self.regular else 2
        if self.energy.ndim != expected_ndim:
            layout = "a regular grid" if self.regular else "scattered points"
            raise ValueError(
                f"A {len(self.cvs)}-D FES on {layout} needs {expected_ndim}-D "
                f"arrays, got shape {self.energy.shape}"
            )

        if not self.cv_labels:
            self.cv_labels = [f"CV{i + 1}" for i in range(len(self.cvs))]
        else:
            self.cv_labels = list(self.cv_labels)
            if len(self.cv_labels) != len(self.cvs):
                raise ValueError(
                    f"Got {len(self.cv_labels)} CV labels for {len(self.cvs)} CVs"
                )

    @property
    def ndim(self) -> int:
        """int: Number of collective variables (1 or 2)."""
        return len(self.cvs)

    @property
    def label(self) -> str:
        """str: Label to use for the free-energy axis or colour bar."""
        return self.energy_label or unit_label(self.energy_unit)

    def finite_range(self) -> tuple[float, float]:
        """Return the range spanned by the finite energies.

        Returns
        -------
        tuple of float
            ``(minimum, maximum)``, or ``(nan, nan)`` when nothing is finite.
        """
        finite = np.isfinite(self.energy)
        if not finite.any():
            return float("nan"), float("nan")
        return float(np.min(self.energy[finite])), float(np.max(self.energy[finite]))

    def slice_at(
        self,
        value: float,
        axis: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Take a 1-D cut through a 2-D surface at a fixed value of one CV.

        Parameters
        ----------
        value : float
            Value of the collective variable held fixed. The nearest grid
            point is used.
        axis : int, optional
            Index of the collective variable held fixed.

        Returns
        -------
        tuple
            ``(x, energy, held_value)`` where *x* runs along the free
            collective variable and *held_value* is the grid value actually
            used.

        Raises
        ------
        ValueError
            If the surface is not a 2-D surface on a regular grid.
        """
        if self.ndim != 2 or not self.regular:
            raise ValueError(
                "Slicing requires a 2-D free-energy surface on a regular grid"
            )
        if axis not in (0, 1):
            raise ValueError(f"axis must be 0 or 1, got {axis!r}")

        # Grids are stored as (n_cv2, n_cv1); axis 0 varies along the columns.
        axis_values = self.cvs[axis][0, :] if axis == 0 else self.cvs[axis][:, 0]
        index = int(np.argmin(np.abs(axis_values - value)))
        other = 1 - axis
        if axis == 0:
            return (
                self.cvs[other][:, index],
                self.energy[:, index],
                float(axis_values[index]),
            )
        return (
            self.cvs[other][index, :],
            self.energy[index, :],
            float(axis_values[index]),
        )


def _grid_from_columns(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Reshape scattered column data onto a regular grid when possible.

    Parameters
    ----------
    x, y, z : numpy.ndarray
        Flat coordinate and value arrays of equal length.

    Returns
    -------
    tuple
        ``(X, Y, Z, regular)``. When the points form a complete rectangular
        grid the arrays are 2-D and *regular* is True, otherwise the inputs
        are returned unchanged with *regular* set to False.
    """
    n_x = np.unique(x).size
    n_y = np.unique(y).size
    if n_x * n_y != z.size:
        return x, y, z, False

    order = np.lexsort((x, y))
    return (
        x[order].reshape(n_y, n_x),
        y[order].reshape(n_y, n_x),
        z[order].reshape(n_y, n_x),
        True,
    )


def _fes_from_plumed(
    path: str | Path,
    columns: Sequence[str | int] | None = None,
    cv_labels: Sequence[str] | None = None,
    energy_label: str | None = None,
) -> FES:
    """Build an :class:`FES` from a PLUMED FES file.

    Parameters
    ----------
    path : str
        Path to the file.
    columns : sequence of (str or int) or None, optional
        Columns to use, ordered ``(cv1, energy)`` or ``(cv1, cv2, energy)``.
        By default the leading columns are used, which matches the layout
        written by ``plumed sum_hills`` and the OPES scripts once derivative
        columns have been dropped.
    cv_labels : sequence of str or None, optional
        Override the collective-variable labels taken from the header.
    energy_label : str or None, optional
        Override the free-energy label.

    Returns
    -------
    FES
        The parsed surface, in the units of the file.

    Raises
    ------
    ValueError
        If the file has fewer than two usable columns.
    """
    plumed = read_plumed_file(path)
    n_columns = plumed.data.shape[1]
    if n_columns < 2:
        raise ValueError(f"Need at least 2 columns to plot, got {n_columns} in {path}")

    if columns is None:
        # 2 columns -> 1-D FES, anything else -> use the first three columns.
        indices = list(range(min(n_columns, 3)))
    else:
        indices = [plumed.index(column) for column in columns]
        if len(indices) not in (2, 3):
            raise ValueError(f"columns must select 2 or 3 fields, got {len(indices)}")

    defaults = ["CV1", "CV2", r"$F$"] if len(indices) == 3 else ["CV1", r"$F$"]
    labels = [plumed.label(index, default) for index, default in zip(indices, defaults)]
    energies = plumed.data[:, indices[-1]]

    if len(indices) == 2:
        x = plumed.data[:, indices[0]]
        order = np.argsort(x)
        return FES(
            cvs=[x[order]],
            energy=energies[order],
            cv_labels=list(cv_labels) if cv_labels else labels[:1],
            energy_label=energy_label,
        )

    x, y, z, regular = _grid_from_columns(
        plumed.data[:, indices[0]], plumed.data[:, indices[1]], energies
    )
    return FES(
        cvs=[x, y],
        energy=z,
        cv_labels=list(cv_labels) if cv_labels else labels[:2],
        energy_label=energy_label,
        regular=regular,
    )


def _fes_from_array(
    source: np.ndarray | Sequence[Any],
    cv_labels: Sequence[str] | None = None,
    energy_label: str | None = None,
) -> FES:
    """Build an :class:`FES` from in-memory arrays.

    The following layouts are recognised:

    ==========================  ==============================================
    Input                       Interpretation
    ==========================  ==============================================
    ``(2, N)``                  1-D surface, rows ``[cv, F]``
    ``(N, 2)``                  1-D surface, columns ``[cv, F]``
    ``(3, ny, nx)``             2-D surface, stacked ``[X, Y, F]``
    ``(X, Y, Z)`` sequence      2-D surface on a grid
    ``(x, y, z)`` sequence      2-D surface, scattered points
    ``(N, 3)``                  2-D surface, columns ``[cv1, cv2, F]``
    ``(3, N)``                  2-D surface, rows ``[cv1, cv2, F]``
    ==========================  ==============================================

    Square inputs such as ``(2, 2)`` are ambiguous and are read row-wise,
    matching the ``[cv, F]`` convention used elsewhere in the package.

    Parameters
    ----------
    source : array_like or sequence of array_like
        The free-energy data.
    cv_labels : sequence of str or None, optional
        Labels for the collective variables.
    energy_label : str or None, optional
        Label for the free energy.

    Returns
    -------
    FES
        The interpreted surface.

    Raises
    ------
    ValueError
        If the layout cannot be interpreted.
    """
    # A sequence of separate arrays, e.g. (X, Y, Z) of matching shape.
    if isinstance(source, (list, tuple)) and len(source) in (2, 3):
        parts = [np.asarray(part, dtype=float) for part in source]
        if all(
            part.ndim == parts[0].ndim and part.shape == parts[0].shape
            for part in parts
        ):
            source = np.stack(parts)
        else:
            raise ValueError("Sequence FES input requires arrays of identical shape")

    array = np.asarray(source, dtype=float)

    if array.ndim == 3:
        if array.shape[0] != 3:
            raise ValueError(
                f"3-D FES input must be stacked as (3, ny, nx), got {array.shape}"
            )
        return FES(
            cvs=[array[0], array[1]],
            energy=array[2],
            cv_labels=list(cv_labels) if cv_labels else [],
            energy_label=energy_label,
        )

    if array.ndim != 2:
        raise ValueError(f"Cannot interpret FES input with shape {array.shape}")

    # Orient so that variables run along the rows.
    if array.shape[0] not in (2, 3):
        if array.shape[1] not in (2, 3):
            raise ValueError(f"Cannot interpret FES input with shape {array.shape}")
        array = array.T

    if array.shape[0] == 2:
        order = np.argsort(array[0])
        return FES(
            cvs=[array[0][order]],
            energy=array[1][order],
            cv_labels=list(cv_labels) if cv_labels else [],
            energy_label=energy_label,
        )

    x, y, z, regular = _grid_from_columns(array[0], array[1], array[2])
    return FES(
        cvs=[x, y],
        energy=z,
        cv_labels=list(cv_labels) if cv_labels else [],
        energy_label=energy_label,
        regular=regular,
    )


def as_fes(
    source: FES | str | Path | np.ndarray | Sequence[Any],
    energy_unit: str | None = None,
    source_unit: str | None = DEFAULT_ENERGY_UNIT,
    shift_min_to_zero: bool = True,
    max_energy: float | None = None,
    columns: Sequence[str | int] | None = None,
    cv_labels: Sequence[str] | None = None,
    energy_label: str | None = None,
) -> FES:
    """Coerce anything FES-shaped into a prepared :class:`FES`.

    This is the single entry point used by every plotting function, so all
    of them accept file paths, raw arrays and :class:`FES` objects
    interchangeably.

    Parameters
    ----------
    source : str, array_like or FES
        A PLUMED FES file, an array in one of the layouts documented in
        :func:`_fes_from_array`, or an already-built surface.
    energy_unit : str or None, optional
        Unit to convert the energies to, e.g. ``"eV"``. ``None``, the default,
        leaves them untouched. Surfaces that already carry this unit are
        not converted twice.
    source_unit : str, optional
        Unit the incoming energies are expressed in. Left at kJ/mol, which is
        what PLUMED writes when driven from OpenMM; an ASE-driven run needs
        ``"eV"``.
    shift_min_to_zero : bool, optional
        Whether to subtract the global minimum so the surface starts at
        zero. The operation is idempotent.
    max_energy : float or None, optional
        Energies above this value (in the *output* unit, after shifting) are
        replaced by NaN so that poorly sampled regions do not dominate the
        colour scale. ``None``, the default, keeps everything.
    columns : sequence of (str or int) or None, optional
        Which file columns to use, ordered ``(cv1, energy)`` or
        ``(cv1, cv2, energy)``. Only meaningful for file sources.
    cv_labels : sequence of str or None, optional
        Override the collective-variable labels.
    energy_label : str or None, optional
        Override the free-energy label.

    Returns
    -------
    FES
        A new, prepared surface. The input is never modified in place.

    Raises
    ------
    ValueError
        If the source cannot be read as a surface, or if *energy_unit* is
        asked for without a *source_unit* to convert from.
    KeyError
        If either unit is not one of :data:`ENERGY_UNITS`.
    """
    if isinstance(source, FES):
        fes = FES(
            cvs=list(source.cvs),
            energy=source.energy.copy(),
            cv_labels=list(source.cv_labels),
            energy_unit=source.energy_unit,
            energy_label=source.energy_label,
            regular=source.regular,
        )
        if cv_labels:
            fes.cv_labels = list(cv_labels)
        if energy_label:
            fes.energy_label = energy_label
    elif isinstance(source, (str, os.PathLike)):
        fes = _fes_from_plumed(
            source, columns=columns, cv_labels=cv_labels, energy_label=energy_label
        )
    else:
        fes = _fes_from_array(source, cv_labels=cv_labels, energy_label=energy_label)

    # Non-finite entries (unvisited bins) must not take part in the shift.
    fes.energy = np.where(np.isfinite(fes.energy), fes.energy, np.nan)

    current_unit = fes.energy_unit or source_unit
    if energy_unit is not None and current_unit is None:
        raise ValueError("Cannot convert to energy_unit without knowing source_unit")
    if energy_unit is not None:
        fes.energy = convert_energy(fes.energy, source=current_unit, target=energy_unit)
    fes.energy_unit = energy_unit or current_unit

    if shift_min_to_zero and np.isfinite(fes.energy).any():
        fes.energy = fes.energy - np.nanmin(fes.energy)

    if max_energy is not None:
        fes.energy = np.where(fes.energy > max_energy, np.nan, fes.energy)

    return fes


def fes_series_files(
    directory: str | Path = ".",
    pattern: str = r"^fes_?(\d+)\.dat$",
) -> list[Path]:
    r"""Find the numbered free-energy surfaces in a directory, in index order.

    The counterpart of :func:`~reactiontools.tools_plumed.sum_hills_files` for
    the other two ways a convergence series gets written: the bundled OPES
    ``FES_from_*`` scripts, and any run that numbered its own output. Both
    produce ``fes_1.dat``, ``fes2.dat`` and the like, which sorting by name
    puts in the wrong order the moment there are ten of them -- and for a
    convergence series the order is the entire point.

    Parameters
    ----------
    directory : str or path-like, optional
        Directory to search.
    pattern : str, optional
        Regular expression matched against each file name, case-insensitively.
        Its first capture group, if present, is the sort key. The default
        matches ``fes_<n>.dat`` and ``fes<n>.dat``.

    Returns
    -------
    list of pathlib.Path
        Matching files, ordered by the index they carry, ready to hand to
        :func:`plot_fes_1d` or :func:`fes_convergence`.

    See Also
    --------
    reactiontools.tools_plumed.sum_hills_files : For a strided ``sum_hills``
        run, which numbers its files differently again.

    Examples
    --------
    A convergence series written by the OPES scripts::

        plot_fes_1d(fes_series_files("."), source_unit="kJ/mol")
    """
    regex = re.compile(pattern, re.IGNORECASE)
    directory = Path(directory)

    matches = []
    for path in directory.iterdir():
        found = regex.match(path.name)
        if found:
            # Sort on the captured index when the pattern provides one.
            key = int(found.group(1)) if found.groups() else path.name
            matches.append((key, path))

    return [path for _, path in sorted(matches, key=lambda item: item[0])]


def load_fes_series(
    directory: str | Path = ".",
    energy_unit: str | None = "eV",
    source_unit: str | None = DEFAULT_ENERGY_UNIT,
    pattern: str = r"^fes_?(\d+)\.dat$",
    verbose: bool = True,
) -> list[FES]:
    r"""Load the numbered free-energy surfaces in a directory, in index order.

    :func:`fes_series_files` followed by :func:`as_fes` on each, so 1-D and
    2-D surfaces are both handled and the result goes straight to
    :func:`plot_fes_1d`, :func:`plot_fes_2d` or :func:`fes_convergence`.

    Parameters
    ----------
    directory : str or path-like, optional
        Directory containing the surfaces.
    energy_unit : str, optional
        Unit to convert the free energies to. Default is ``"eV"``.
    source_unit : str, optional
        Unit the files are written in. The default is kJ/mol, which is what
        PLUMED writes when driven from OpenMM; a run driven from ASE through
        :data:`~reactiontools.tools_plumed.PLUMED_ASE_UNITS` writes eV.
    pattern : str, optional
        Regular expression selecting the files; see :func:`fes_series_files`.
    verbose : bool, optional
        Report each file as it is loaded.

    Returns
    -------
    list of FES
        One surface per file, ordered by file index.
    """
    surfaces = []
    for path in fes_series_files(directory, pattern=pattern):
        fes = as_fes(path, energy_unit=energy_unit, source_unit=source_unit)
        if verbose:
            print(f"Loading {path} with {fes.ndim} CV(s)", flush=True)
        surfaces.append(fes)
    return surfaces


def _as_fes_list(sources: Any, **kwargs: Any) -> list[FES]:
    """Coerce one source or a collection of sources into a list of surfaces.

    Parameters
    ----------
    sources : object
        A single FES source, or a list/tuple of them.
    **kwargs
        Forwarded to :func:`as_fes`.

    Returns
    -------
    list of FES
        The prepared surfaces.

    Raises
    ------
    ValueError
        If *sources* is an empty collection.
    """
    if _is_single_source(sources):
        sources = [sources]
    fes_list = [as_fes(source, **kwargs) for source in sources]
    if not fes_list:
        raise ValueError("At least one free-energy surface is required")
    return fes_list


def _looks_like_fes_array(array: np.ndarray) -> bool:
    """Return True when an array can stand on its own as a free-energy surface.

    Used to tell ``[fes_a, fes_b]`` (a collection) apart from
    ``(X, Y, Z)`` (one surface), since both are sequences of arrays.

    Parameters
    ----------
    array : numpy.ndarray
        Candidate array.

    Returns
    -------
    bool
        True if the shape matches one of the layouts accepted by
        :func:`_fes_from_array`.
    """
    shape = np.shape(array)
    if len(shape) == 3:
        return shape[0] == 3
    if len(shape) == 2:
        return shape[0] in (2, 3) or shape[1] in (2, 3)
    return False


def _is_single_source(source: Any) -> bool:
    """Return True when *source* is one FES rather than a collection of them.

    A tuple of matching coordinate/value arrays is one surface -- ``(x, F)``,
    ``(x, y, z)`` or ``(X, Y, Z)``. A list whose entries are themselves
    FES-shaped arrays is a collection, so ``[fes_a, fes_b]`` behaves as
    expected. This tuple/list distinction resolves small grids whose shape
    can otherwise look exactly like a collection of curves.

    Parameters
    ----------
    source : object
        Candidate FES source.

    Returns
    -------
    bool
        True if *source* should be treated as a single surface.
    """
    if isinstance(source, (FES, str, os.PathLike, np.ndarray)):
        return True
    if isinstance(source, tuple):
        if len(source) not in (2, 3):
            return False
        return all(
            isinstance(part, np.ndarray) and part.shape == source[0].shape
            for part in source
        )
    if isinstance(source, list):
        if len(source) not in (2, 3):
            return False
        parts = [part for part in source if isinstance(part, np.ndarray)]
        if len(parts) != len(source):
            return False
        if any(part.shape != parts[0].shape for part in parts):
            return False
        return not _looks_like_fes_array(parts[0])
    return True


def _resolve_labels(
    labels: Sequence[Any] | None,
    count: int,
    template: str | None = None,
) -> list[str | None]:
    """Build one label per dataset.

    Parameters
    ----------
    labels : sequence or None
        Explicit labels, or values to be formatted with *template*.
    count : int
        Number of datasets.
    template : str or None, optional
        ``str.format`` template applied to non-string labels, e.g.
        ``r"$t={:g}$ ps"``.

    Returns
    -------
    list
        A list of *count* labels; entries are None when nothing should be
        shown in the legend.

    Raises
    ------
    ValueError
        If *labels* is given but has the wrong length.
    """
    if labels is None:
        return [None] * count if count == 1 else [f"FES {i + 1}" for i in range(count)]

    labels = list(labels)
    if len(labels) != count:
        raise ValueError(f"Got {len(labels)} labels for {count} datasets")

    resolved = []
    for label in labels:
        if label is None or isinstance(label, str):
            resolved.append(label)
        elif template is not None:
            resolved.append(template.format(label))
        else:
            resolved.append(f"{label}")
    return resolved


def _keep_last(
    fes_list: list[FES],
    label_list: list[str | None],
    max_datasets: int | None,
) -> tuple[list[FES], list[str | None]]:
    """Keep only the last *max_datasets* surfaces and their labels.

    Trimming happens after label resolution, so a convergence series
    labelled by time keeps the labels of the *latest* surfaces.

    Parameters
    ----------
    fes_list : list of FES
        The prepared surfaces.
    label_list : list
        One label per surface.
    max_datasets : int or None
        Number of surfaces to keep. ``None`` keeps everything.

    Returns
    -------
    tuple of list
        ``(fes_list, label_list)``, trimmed together.

    Raises
    ------
    ValueError
        If *max_datasets* is neither ``None`` nor a positive integer.
    """
    if max_datasets is not None:
        if isinstance(max_datasets, bool) or not isinstance(
            max_datasets, (int, np.integer)
        ):
            raise ValueError("max_datasets must be a positive integer or None")
        if max_datasets < 1:
            raise ValueError("max_datasets must be a positive integer or None")
    if max_datasets is not None and len(fes_list) > max_datasets:
        return fes_list[-max_datasets:], label_list[-max_datasets:]
    return fes_list, label_list


def _default_colors(count: int) -> list[str | tuple[float, ...]]:
    """Pick one colour per surface from the active colour cycle, repeating.

    Parameters
    ----------
    count : int
        Number of colours needed.

    Returns
    -------
    list
        The colours, cycling when *count* exceeds the cycle length.
    """
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
    return [cycle[i % len(cycle)] for i in range(count)]


def _resolve_colors(
    colors: str | Sequence[str | tuple[float, ...]] | None,
    count: int,
) -> list[str | tuple[float, ...]]:
    """Return exactly one colour per dataset, rejecting silent truncation.

    Parameters
    ----------
    colors : str or sequence of str or None
        One colour name, one per dataset, or ``None`` for the default cycle.
    count : int
        Number of datasets to be drawn.

    Returns
    -------
    list of str
        One colour per dataset, in order.

    Raises
    ------
    ValueError
        If a sequence was given whose length is not *count*. Recycling or
        truncating it would silently colour two datasets alike.
    """
    if colors is None:
        return _default_colors(count)
    if isinstance(colors, str):
        colors = [colors]
    colors = list(colors)
    if len(colors) != count:
        raise ValueError(f"Got {len(colors)} colors for {count} surfaces")
    return colors


def _figure_from_axes(
    fig: Figure | None,
    axes: Axes | Sequence[Axes] | np.ndarray,
) -> Figure:
    """Resolve and validate the figure which owns supplied axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or None
        Figure the caller supplied alongside the axes, checked against them
        when it is not ``None``.
    axes : matplotlib.axes.Axes or sequence of matplotlib.axes.Axes
        The axes to be drawn on, in any array shape.

    Returns
    -------
    matplotlib.figure.Figure
        The figure the axes belong to.

    Raises
    ------
    ValueError
        If the axes are spread over more than one figure, or *fig* is not the
        figure that owns them.
    """
    axes = np.atleast_1d(axes).ravel()
    owners = {axis.figure for axis in axes}
    if len(owners) != 1:
        raise ValueError("All supplied axes must belong to the same figure")
    owner = owners.pop()
    if fig is not None and fig is not owner:
        raise ValueError("The supplied figure does not own the supplied axes")
    return owner


def _shared_levels(
    fes_list: Sequence[FES],
    levels: int | Sequence[float] | np.ndarray,
) -> np.ndarray | int:
    """Build contour levels spanning every surface in *fes_list*.

    Sharing the levels is what makes a single colour bar meaningful across
    panels.

    Parameters
    ----------
    fes_list : sequence of FES
        The surfaces to be drawn.
    levels : int or array_like
        Number of levels, or explicit level values which are returned
        unchanged.

    Returns
    -------
    numpy.ndarray or int
        The contour levels. When every surface is flat there is no range to
        divide, and the count is passed straight back for matplotlib to
        place the levels itself.

    Raises
    ------
    ValueError
        If *levels* is neither an integer of at least 2 nor a strictly
        increasing finite sequence, or no surface has a finite energy.
    """
    if not np.isscalar(levels):
        levels = np.asarray(levels, dtype=float)
        if (
            levels.ndim != 1
            or levels.size < 2
            or not np.isfinite(levels).all()
            or np.any(np.diff(levels) <= 0)
        ):
            raise ValueError("levels must be a strictly increasing finite sequence")
        return levels

    try:
        count = int(levels)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("levels must be an integer of at least 2") from exc
    if isinstance(levels, (bool, np.bool_)) or count != levels or count < 2:
        raise ValueError("levels must be an integer of at least 2")
    levels = count

    ranges = [fes.finite_range() for fes in fes_list]
    if any(not np.isfinite(bounds).all() for bounds in ranges):
        raise ValueError("Cannot contour a surface with no finite energies")
    lows, highs = zip(*ranges)
    low = min(lows)
    high = max(highs)
    if high <= low:
        return levels
    return np.linspace(low, high, levels)


def _draw_fes_contour(axis: Axes, fes: FES, filled: bool, **kwargs: Any) -> ContourSet:
    """Draw one regular or triangulated FES with a common validation path.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axes to draw on.
    fes : FES
        The surface. Gridded surfaces go through ``contour``/``contourf``,
        scattered ones through their ``tri`` counterparts.
    filled : bool
        Fill between the levels rather than drawing lines.
    **kwargs
        Passed through to the matplotlib contouring call.

    Returns
    -------
    matplotlib.contour.QuadContourSet or matplotlib.tri.TriContourSet
        The contour set that was drawn.

    Raises
    ------
    ValueError
        If a gridded surface has a non-finite CV coordinate, or a scattered
        one has fewer than the three finite points a triangulation needs.
    """
    method = "contourf" if filled else "contour"
    if fes.regular:
        if any(not np.isfinite(cv).all() for cv in fes.cvs):
            raise ValueError("Regular FES grids must have finite CV coordinates")
        return getattr(axis, method)(fes.cvs[0], fes.cvs[1], fes.energy, **kwargs)

    finite = np.isfinite(fes.energy)
    finite &= np.isfinite(fes.cvs[0]) & np.isfinite(fes.cvs[1])
    if np.count_nonzero(finite) < 3:
        raise ValueError("A scattered FES needs at least three finite points")
    return getattr(axis, f"tri{method}")(
        fes.cvs[0][finite], fes.cvs[1][finite], fes.energy[finite], **kwargs
    )


def _default_grid_size(
    n_panels: int,
    fig_size: tuple[float, float] | None,
) -> tuple[float, float]:
    """Pick a figure size that grows with the number of panels.

    Parameters
    ----------
    n_panels : int
        Number of side-by-side panels.
    fig_size : tuple or None
        Explicit size, returned unchanged when given.

    Returns
    -------
    tuple of float
        Figure size in inches.
    """
    if fig_size is not None:
        return fig_size
    return (3.2 * n_panels + 1.8, 3.4)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _format_energy(value: float) -> str:
    """Format an energy, without a sign on a value that rounds to zero.

    A barrier that comes out a hair below zero reads as a finding rather than
    as the rounding it is.

    Parameters
    ----------
    value : float
        The energy to format.

    Returns
    -------
    str
        The value to three decimal places, never as ``"-0.000"``.
    """
    text = f"{value:.3f}"
    return "0.000" if text == "-0.000" else text


def _basin_mask(fes: FES, basin: Sequence[float], name: str) -> np.ndarray:
    """Select the sampled grid points of a 1-D surface inside a CV window.

    Parameters
    ----------
    fes : FES
        One-dimensional surface.
    basin : sequence of float
        ``(low, high)`` bounds on the collective variable, in either order.
    name : str
        Argument name, quoted in the error message.

    Returns
    -------
    numpy.ndarray
        Boolean mask over the grid.

    Raises
    ------
    ValueError
        If the window holds no sampled point.
    """
    cv = fes.cvs[0]
    low, high = sorted(float(bound) for bound in basin)
    inside = (cv >= low) & (cv <= high) & np.isfinite(fes.energy)
    if not inside.any():
        raise ValueError(
            f"{name}=({low:g}, {high:g}) holds no sampled grid point. The "
            f"collective variable runs from {np.min(cv):g} to {np.max(cv):g}, "
            f"and unsampled bins do not count."
        )
    return inside


def _basin_free_energy(
    fes: FES,
    mask: np.ndarray,
    kt: float | None,
    reference: float,
) -> float:
    """Free energy of a basin: its minimum, or the Boltzmann integral over it.

    Parameters
    ----------
    fes : FES
        One-dimensional surface.
    mask : numpy.ndarray
        Grid points making up the basin.
    kt : float or None
        Thermal energy in the surface's own units. ``None`` returns the
        minimum instead of integrating.
    reference : float
        Energy the exponent is measured from, shared by every basin so that
        it cancels in a difference.

    Returns
    -------
    float
        The basin's free energy.
    """
    energy = fes.energy[mask]
    if kt is None:
        return float(np.min(energy))
    weights = np.exp(-(energy - reference) / kt)
    return float(reference - kt * np.log(np.trapezoid(weights, fes.cvs[0][mask])))


@dataclass
class FESSummary:
    """The numbers read off a free-energy profile.

    Attributes
    ----------
    minimum_a, minimum_b : float
        Where each basin bottoms out, in collective-variable units.
    depth_a, depth_b : float
        The free energy there. The barriers are measured from these, since a
        barrier is a barrier out of the bottom of a well.
    delta_f : float
        Free energy of basin B relative to basin A. The difference of the two
        depths, or of the two Boltzmann-integrated basin free energies when
        :func:`summarise_fes` was given a temperature — which is the one that
        accounts for a wide basin being more probable than a narrow one of
        equal depth.
    barrier_position : float
        Where the profile peaks between the basins.
    forward_barrier, reverse_barrier : float
        Height of that peak above basin A and above basin B.
    energy_unit : str or None
        Unit the energies are in, for reporting.
    """

    minimum_a: float
    minimum_b: float
    depth_a: float
    depth_b: float
    delta_f: float
    barrier_position: float
    forward_barrier: float
    reverse_barrier: float
    energy_unit: str | None = None

    def __str__(self) -> str:
        """Report the barriers, the basin difference and where they sit."""
        unit = f" {self.energy_unit}" if self.energy_unit else ""
        return (
            f"Barrier A->B:  {_format_energy(self.forward_barrier)}{unit}\n"
            f"Barrier B->A:  {_format_energy(self.reverse_barrier)}{unit}\n"
            f"Delta F (B-A): {_format_energy(self.delta_f)}{unit}\n"
            f"Minima at:     {self.minimum_a:g}, {self.minimum_b:g}\n"
            f"Barrier at:    {self.barrier_position:g}"
        )


def summarise_fes(
    source: FES | str | Path | np.ndarray | Sequence[Any],
    basin_a: Sequence[float],
    basin_b: Sequence[float],
    temperature: float | None = None,
    **kwargs: Any,
) -> FESSummary:
    """Measure the barrier and basin free-energy difference off a 1-D surface.

    The two basins are given as windows on the collective variable rather
    than found automatically, which keeps the answer predictable on a noisy
    surface and, more to the point, keeps it comparable across the surfaces
    of a convergence series.

    Parameters
    ----------
    source : FES source
        A PLUMED file, an array or an :class:`FES`; see :func:`as_fes`. Must
        be one-dimensional.
    basin_a, basin_b : sequence of float
        ``(low, high)`` bounds of each basin. They may be given in either
        order but must not overlap, and there must be grid points between
        them for the barrier to sit on.
    temperature : float or None, optional
        Temperature in kelvin. Given one, ``delta_f`` comes from integrating
        the Boltzmann weight across each basin, so that a wide basin counts
        for more than a narrow one of the same depth. ``None``, the default,
        just compares the two minima. The barriers are measured from the
        minima either way.
    **kwargs
        Forwarded to :func:`as_fes`: ``energy_unit``, ``source_unit``,
        ``shift_min_to_zero``, ``max_energy``, ``columns`` and the label
        overrides.

    Returns
    -------
    FESSummary
        Barriers, basin minima and their free-energy difference.

    Raises
    ------
    ValueError
        If the surface is not one-dimensional, if either basin holds no
        sampled point, if the basins overlap, if nothing lies between them,
        or if a temperature is given for a surface whose energy unit is
        unknown.

    Examples
    --------
    >>> summary = summarise_fes("fes.dat", basin_a=(1.0, 2.0),   # doctest: +SKIP
    ...                         basin_b=(3.0, 4.5), source_unit="eV")
    >>> print(summary)                                       # doctest: +SKIP
    Barrier A->B:  0.352 eV
    Barrier B->A:  0.418 eV
    Delta F (B-A): -0.066 eV
    Minima at:     1.5, 3.75
    Barrier at:    2.6
    """
    fes = as_fes(source, **kwargs)
    if fes.ndim != 1:
        raise ValueError(
            f"summarise_fes reads a barrier off a 1-D profile, but this "
            f"surface has {fes.ndim} collective variables. Project it first "
            f"with run_sum_hills(idw=...), or cut it with FES.slice_at."
        )

    kt = None
    if temperature is not None:
        if fes.energy_unit is None:
            raise ValueError(
                "Cannot use temperature without knowing the surface's energy "
                "unit; pass source_unit or energy_unit."
            )
        kt = thermal_energy(temperature, fes.energy_unit)

    cv = fes.cvs[0]
    mask_a = _basin_mask(fes, basin_a, "basin_a")
    mask_b = _basin_mask(fes, basin_b, "basin_b")

    low_a, high_a = sorted(float(bound) for bound in basin_a)
    low_b, high_b = sorted(float(bound) for bound in basin_b)
    if low_a <= high_b and low_b <= high_a:
        raise ValueError(
            f"basin_a=({low_a:g}, {high_a:g}) and basin_b=({low_b:g}, "
            f"{high_b:g}) overlap, so there is no barrier between them."
        )

    inner_low, inner_high = (high_a, low_b) if high_a < low_b else (high_b, low_a)
    between = (cv > inner_low) & (cv < inner_high) & np.isfinite(fes.energy)
    if not between.any():
        raise ValueError(
            f"No sampled grid point lies between the basins, over "
            f"({inner_low:g}, {inner_high:g}), so there is no barrier to "
            f"measure. Widen the gap between them, or check the surface is "
            f"sampled there."
        )

    index_a = np.flatnonzero(mask_a)[np.argmin(fes.energy[mask_a])]
    index_b = np.flatnonzero(mask_b)[np.argmin(fes.energy[mask_b])]
    top = np.flatnonzero(between)[np.argmax(fes.energy[between])]

    reference = float(np.nanmin(fes.energy))
    energy_a = _basin_free_energy(fes, mask_a, kt, reference)
    energy_b = _basin_free_energy(fes, mask_b, kt, reference)

    return FESSummary(
        minimum_a=float(cv[index_a]),
        minimum_b=float(cv[index_b]),
        depth_a=float(fes.energy[index_a]),
        depth_b=float(fes.energy[index_b]),
        delta_f=energy_b - energy_a,
        barrier_position=float(cv[top]),
        forward_barrier=float(fes.energy[top] - fes.energy[index_a]),
        reverse_barrier=float(fes.energy[top] - fes.energy[index_b]),
        energy_unit=fes.energy_unit,
    )


def fes_convergence(
    sources: FES | str | Path | np.ndarray | Sequence[Any],
    basin_a: Sequence[float],
    basin_b: Sequence[float],
    temperature: float | None = None,
    **kwargs: Any,
) -> list[FESSummary]:
    """Summarise each surface of a series, to see whether the numbers settle.

    A metadynamics run is converged when the barrier and the basin difference
    stop moving, not when the surface stops looking different. Feed it the
    series :func:`~reactiontools.sum_hills_files` collects from a strided
    ``sum_hills``.

    Parameters
    ----------
    sources : sequence of FES source
        The surfaces, in order.
    basin_a, basin_b : sequence of float
        Basin windows, the same for every surface — that is the point of
        fixing them rather than finding them per surface.
    temperature : float or None, optional
        See :func:`summarise_fes`.
    **kwargs
        Forwarded to :func:`as_fes`.

    Returns
    -------
    list of FESSummary
        One per surface, in the order given.

    Raises
    ------
    ValueError
        If *sources* is empty, or for any of the reasons
        :func:`summarise_fes` raises on one of the surfaces.
    """
    # Pass the sources through untouched rather than preparing them here:
    # summarise_fes calls as_fes itself, and doing it twice would apply
    # max_energy and the unit conversion on top of themselves.
    if _is_single_source(sources):
        sources = [sources]
    summaries = [
        summarise_fes(source, basin_a, basin_b, temperature=temperature, **kwargs)
        for source in sources
    ]
    if not summaries:
        raise ValueError("At least one free-energy surface is required")
    return summaries


def plot_fes_1d(
    sources: FES | str | Path | np.ndarray | Sequence[Any],
    fig: Figure | None = None,
    ax: Axes | None = None,
    labels: Sequence[Any] | None = None,
    label_template: str | None = None,
    max_datasets: int | None = None,
    energy_unit: str | None = None,
    source_unit: str | None = DEFAULT_ENERGY_UNIT,
    shift_min_to_zero: bool = True,
    max_energy: float | None = None,
    columns: Sequence[str | int] | None = None,
    x_lab: str | None = None,
    y_lab: str | None = None,
    filename: str | Path | None = None,
    show: bool = False,
    fig_size: tuple[float, float] = (8, 3),
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot one or more 1-D free-energy profiles on a single axes.

    This covers a single profile, a convergence series over time and a
    method-to-method comparison, since they differ only in their labels.

    Parameters
    ----------
    sources : FES source or sequence of FES sources
        Paths, arrays or :class:`FES` objects; see :func:`as_fes`.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on. Inferred from *ax* when omitted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    labels : sequence, optional
        Legend entries, one per dataset. Non-string values are formatted
        with *label_template*, which makes ``labels=times`` with
        ``label_template=r"$t={:g}$ ps"`` a convenient time series.
    label_template : str, optional
        ``str.format`` template applied to non-string *labels*.
    max_datasets : int, optional
        Keep only the last *max_datasets* surfaces. ``None``, the default,
        plots all of them.
    energy_unit : str, optional
        Unit to convert energies to, e.g. ``"eV"``.
    source_unit : str, optional
        Unit the input energies are in.
    shift_min_to_zero : bool, optional
        Whether to shift each surface so its minimum is zero.
    max_energy : float, optional
        Mask energies above this value.
    columns : sequence of (str or int), optional
        Columns to use for file sources, ordered ``(cv, energy)``.
    x_lab, y_lab : str, optional
        Axis labels. Taken from the data when not given.
    filename : str, optional
        Output path; ``None``, the default, writes nothing. A bare stem writes
        both PNG and PDF.
    show : bool, optional
        Whether to display the figure.
    fig_size : tuple, optional
        Figure size in inches.
    **plot_kwargs
        Extra keyword arguments forwarded to ``Axes.plot``.

    Returns
    -------
    tuple
        ``(fig, ax)`` containing the matplotlib figure and axes.

    Raises
    ------
    ValueError
        If any source is not a 1-D surface.
    """
    fes_list = _as_fes_list(
        sources,
        energy_unit=energy_unit,
        source_unit=source_unit,
        shift_min_to_zero=shift_min_to_zero,
        max_energy=max_energy,
        columns=columns,
    )
    if any(fes.ndim != 1 for fes in fes_list):
        raise ValueError(
            "plot_fes_1d expects 1-D free-energy surfaces; use plot_fes_2d instead"
        )

    label_list = _resolve_labels(labels, len(fes_list), template=label_template)
    fes_list, label_list = _keep_last(fes_list, label_list, max_datasets)

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    else:
        fig = _figure_from_axes(fig, ax)

    for fes, label in zip(fes_list, label_list):
        ax.plot(fes.cvs[0], fes.energy, label=label, **plot_kwargs)

    if any(label is not None for label in label_list):
        ax.legend(loc="best")

    reference = fes_list[0]
    _style_axes(
        fig,
        ax,
        x_lab if x_lab is not None else reference.cv_labels[0],
        y_lab if y_lab is not None else reference.label,
    )
    _finalise(fig, filename=filename, show=show)
    return fig, ax


def plot_fes_2d(
    sources: FES | str | Path | np.ndarray | Sequence[Any],
    fig: Figure | None = None,
    ax: Axes | Sequence[Axes] | np.ndarray | None = None,
    labels: Sequence[Any] | None = None,
    label_template: str | None = None,
    max_datasets: int | None = None,
    energy_unit: str | None = None,
    source_unit: str | None = DEFAULT_ENERGY_UNIT,
    shift_min_to_zero: bool = True,
    max_energy: float | None = None,
    columns: Sequence[str | int] | None = None,
    levels: int | Sequence[float] | np.ndarray = 30,
    cmap: str | Colormap | None = None,
    x_lab: str | None = None,
    y_lab: str | None = None,
    colorbar: bool = True,
    filename: str | Path | None = None,
    show: bool = False,
    fig_size: tuple[float, float] | None = None,
    **contour_kwargs: Any,
) -> tuple[Figure, np.ndarray]:
    """Plot one or more 2-D free-energy surfaces as filled contours.

    A single surface gives one panel; several surfaces are drawn side by
    side with shared axes, shared contour levels and one colour bar, which
    covers both convergence series and method comparisons.

    Parameters
    ----------
    sources : FES source or sequence of FES sources
        Paths, arrays or :class:`FES` objects; see :func:`as_fes`.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on. Inferred from *ax* when omitted.
    ax : matplotlib.axes.Axes or sequence of Axes, optional
        Axes to draw on; must provide one per surface.
    labels : sequence, optional
        Panel titles, one per dataset. Non-string values are formatted with
        *label_template*.
    label_template : str, optional
        ``str.format`` template applied to non-string *labels*, e.g.
        ``r"$t={:g}$ ps"``.
    max_datasets : int, optional
        Keep only the last *max_datasets* surfaces. ``None``, the default,
        plots all of them.
    energy_unit : str, optional
        Unit to convert energies to, e.g. ``"eV"``.
    source_unit : str, optional
        Unit the input energies are in.
    shift_min_to_zero : bool, optional
        Whether to shift each surface so its minimum is zero.
    max_energy : float, optional
        Mask energies above this value, leaving unsampled regions blank.
    columns : sequence of (str or int), optional
        Columns to use for file sources, ordered ``(cv1, cv2, energy)``.
    levels : int or array_like, optional
        Number of contour levels, or explicit level values. An integer is
        expanded to levels spanning every panel.
    cmap : str or matplotlib.colors.Colormap, optional
        Colour map passed to ``contourf``.
    x_lab, y_lab : str, optional
        Axis labels. Taken from the data when not given.
    colorbar : bool, optional
        Whether to draw the shared colour bar.
    filename : str, optional
        Output path; ``None``, the default, writes nothing.
    show : bool, optional
        Whether to display the figure.
    fig_size : tuple, optional
        Figure size in inches. Scales with the number of panels by default.
    **contour_kwargs
        Extra keyword arguments forwarded to ``contourf``/``tricontourf``.

    Returns
    -------
    tuple
        ``(fig, axes)`` with *axes* always a 1-D array, even for a single
        panel.

    Raises
    ------
    ValueError
        If any source is not a 2-D surface, or if the supplied axes do not
        match the number of surfaces.
    """
    fes_list = _as_fes_list(
        sources,
        energy_unit=energy_unit,
        source_unit=source_unit,
        shift_min_to_zero=shift_min_to_zero,
        max_energy=max_energy,
        columns=columns,
    )
    if any(fes.ndim != 2 for fes in fes_list):
        raise ValueError(
            "plot_fes_2d expects 2-D free-energy surfaces; use plot_fes_1d instead"
        )

    label_list = _resolve_labels(labels, len(fes_list), template=label_template)
    fes_list, label_list = _keep_last(fes_list, label_list, max_datasets)

    n_panels = len(fes_list)
    if ax is None:
        fig, ax = plt.subplots(
            nrows=1,
            ncols=n_panels,
            figsize=_default_grid_size(n_panels, fig_size),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
    else:
        fig = _figure_from_axes(fig, ax)
    axes = np.atleast_1d(ax).ravel()
    if axes.size < n_panels:
        raise ValueError(f"Got {axes.size} axes for {n_panels} surfaces")

    shared_levels = _shared_levels(fes_list, levels)
    contours = []
    for axis, fes, label in zip(axes, fes_list, label_list):
        mappable = _draw_fes_contour(
            axis,
            fes,
            filled=True,
            levels=shared_levels,
            cmap=cmap,
            **contour_kwargs,
        )
        contours.append(mappable)
        if label is not None and n_panels > 1:
            axis.set_title(label)

    reference = fes_list[0]
    _style_axes(
        fig,
        axes[:n_panels],
        x_lab if x_lab is not None else reference.cv_labels[0],
        y_lab if y_lab is not None else reference.cv_labels[1],
    )

    if colorbar:
        fig.colorbar(
            contours[-1],
            ax=list(axes[:n_panels]),
            orientation="vertical",
            label=reference.label,
        )

    _finalise(fig, filename=filename, show=show)
    return fig, axes[:n_panels]


def _path_coordinates(
    source: str | Path | PlumedData | np.ndarray | Sequence[Any],
    columns: Sequence[str | int] | None = None,
    cv_labels: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the two CV coordinates of a path.

    Parameters
    ----------
    source : str, path-like, PlumedData or array_like
        A PLUMED ``COLVAR``-style file/container or coordinates shaped
        ``(n_points, 2)``, ``(2, n_points)`` or ``(x, y)``.
    columns : sequence of (str or int), optional
        The two columns to use for PLUMED data. When omitted, fields matching
        *cv_labels* are preferred, then the first two fields other than
        ``time`` are used.
    cv_labels : sequence of str, optional
        FES axis labels to match against a PLUMED header.

    Returns
    -------
    tuple of numpy.ndarray
        The x and y coordinates in path order. Non-finite coordinate pairs
        are retained as NaNs so they break the plotted line instead of
        joining unrelated path segments.

    Raises
    ------
    ValueError
        If two path coordinates cannot be identified or no finite points
        remain.
    """
    if isinstance(source, (str, os.PathLike)):
        source = read_plumed_file(source, drop_der=False)

    if isinstance(source, PlumedData):
        if columns is None:
            labels = list(cv_labels or [])
            if len(labels) == 2 and all(label in source.fields for label in labels):
                columns = labels
            else:
                candidates = [
                    i
                    for i, field_name in enumerate(source.fields)
                    if field_name.lower() != "time"
                ]
                if not source.fields:
                    candidates = list(range(source.data.shape[1]))
                if len(candidates) < 2:
                    raise ValueError("Path data must contain two CV columns")
                columns = candidates[:2]
        columns = list(columns)
        if len(columns) != 2:
            raise ValueError(f"path_columns must select 2 fields, got {len(columns)}")
        coordinates = np.column_stack([source.column(column) for column in columns])
    else:
        if columns is not None:
            raise ValueError("path_columns is only meaningful for PLUMED path data")
        if isinstance(source, (list, tuple)) and len(source) == 2:
            parts = [np.asarray(part, dtype=float) for part in source]
            if all(part.ndim == 1 and part.shape == parts[0].shape for part in parts):
                coordinates = np.column_stack(parts)
            else:
                coordinates = np.asarray(source, dtype=float)
        else:
            coordinates = np.asarray(source, dtype=float)

        if coordinates.ndim != 2 or 2 not in coordinates.shape:
            raise ValueError(
                "Path coordinates must have shape (n_points, 2) or (2, n_points)"
            )
        # For the ambiguous (2, 2) case, treat rows as path points.
        if coordinates.shape[1] != 2:
            coordinates = coordinates.T

    x, y = np.asarray(coordinates, dtype=float).T
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        raise ValueError("Path contains no finite CV coordinates")
    return np.where(finite, x, np.nan), np.where(finite, y, np.nan)


def plot_fes_path(
    source: FES | str | Path | np.ndarray | Sequence[Any],
    path: str | Path | PlumedData | np.ndarray | Sequence[Any],
    fig: Figure | None = None,
    ax: Axes | None = None,
    energy_unit: str | None = None,
    source_unit: str | None = DEFAULT_ENERGY_UNIT,
    shift_min_to_zero: bool = True,
    max_energy: float | None = None,
    columns: Sequence[str | int] | None = None,
    path_columns: Sequence[str | int] | None = None,
    levels: int | Sequence[float] | np.ndarray = 30,
    cmap: str | Colormap | None = None,
    x_lab: str | None = None,
    y_lab: str | None = None,
    colorbar: bool = True,
    path_label: str | None = "Path",
    path_kwargs: Mapping[str, Any] | None = None,
    filename: str | Path | None = None,
    show: bool = False,
    fig_size: tuple[float, float] = (6, 5),
    **contour_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a path through collective-variable space over a 2-D FES.

    The path can come straight from a PLUMED ``COLVAR`` file. If its header
    contains fields with the same names as the FES axes, those fields are
    selected automatically, so a leading ``time`` column and trailing bias
    columns are ignored. Raw ``(n_points, 2)`` and ``(2, n_points)`` arrays
    are also accepted.

    Parameters
    ----------
    source : FES source
        One 2-D surface accepted by :func:`as_fes`.
    path : str, path-like, PlumedData or array_like
        Ordered CV coordinates, or a PLUMED file containing them.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on. Inferred from *ax* when omitted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    energy_unit, source_unit, shift_min_to_zero, max_energy, columns
        FES preparation options; see :func:`plot_fes_2d`.
    path_columns : sequence of (str or int), optional
        Two path columns, ordered ``(cv1, cv2)``. Field names matching the FES
        axes are used automatically when this is omitted.
    levels, cmap, x_lab, y_lab, colorbar
        Surface appearance options; see :func:`plot_fes_2d`.
    path_label : str or None, optional
        Legend label for the path. ``None`` suppresses the legend.
    path_kwargs : mapping, optional
        Appearance options forwarded to ``Axes.plot``. Defaults to a
        semi-transparent white line like the conventional trajectory-on-FES
        plot.
    filename : str, optional
        Output path; ``None``, the default, writes nothing.
    show : bool, optional
        Whether to display the figure.
    fig_size : tuple, optional
        Figure size in inches.
    **contour_kwargs
        Extra keyword arguments forwarded to ``contourf``/``tricontourf``.

    Returns
    -------
    tuple
        ``(fig, ax)`` containing the matplotlib figure and axes.

    Raises
    ------
    ValueError
        If *source* is not one 2-D FES or *path* does not contain two CVs.
    """
    fes = as_fes(
        source,
        energy_unit=energy_unit,
        source_unit=source_unit,
        shift_min_to_zero=shift_min_to_zero,
        max_energy=max_energy,
        columns=columns,
    )
    if fes.ndim != 2:
        raise ValueError("plot_fes_path expects a 2-D free-energy surface")

    x_path, y_path = _path_coordinates(
        path, columns=path_columns, cv_labels=fes.cv_labels
    )
    fig, axes = plot_fes_2d(
        fes,
        fig=fig,
        ax=ax,
        shift_min_to_zero=False,
        levels=levels,
        cmap=cmap,
        x_lab=x_lab,
        y_lab=y_lab,
        colorbar=colorbar,
        filename=None,
        show=False,
        fig_size=fig_size,
        **contour_kwargs,
    )
    ax = axes[0]

    style = {"color": "white", "alpha": 0.7, "linewidth": 1.5, "zorder": 3}
    if path_kwargs is not None:
        style.update(path_kwargs)
    ax.plot(x_path, y_path, label=path_label, **style)
    if path_label is not None:
        ax.legend(loc="best")

    _finalise(fig, filename=filename, show=show)
    return fig, ax


def plot_fes_2d_overlay(
    sources: FES | str | Path | np.ndarray | Sequence[Any],
    fig: Figure | None = None,
    ax: Axes | None = None,
    labels: Sequence[Any] | None = None,
    label_template: str | None = None,
    energy_unit: str | None = None,
    source_unit: str | None = DEFAULT_ENERGY_UNIT,
    shift_min_to_zero: bool = True,
    max_energy: float | None = None,
    columns: Sequence[str | int] | None = None,
    levels: int | Sequence[float] | np.ndarray = 6,
    colors: str | Sequence[str | tuple[float, ...]] | None = None,
    x_lab: str | None = None,
    y_lab: str | None = None,
    filename: str | Path | None = None,
    show: bool = False,
    fig_size: tuple[float, float] = (5, 4),
    **contour_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Overlay the contour lines of several 2-D free-energy surfaces.

    Drawing the surfaces on the same axes in different colours makes small
    shifts between them, such as the effect of nuclear quantum effects on a
    barrier, easy to see.

    Parameters
    ----------
    sources : FES source or sequence of FES sources
        Paths, arrays or :class:`FES` objects; see :func:`as_fes`.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on. Inferred from *ax* when omitted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    labels : sequence, optional
        Legend entries, one per dataset.
    label_template : str, optional
        ``str.format`` template applied to non-string *labels*.
    energy_unit : str, optional
        Unit to convert energies to, e.g. ``"eV"``.
    source_unit : str, optional
        Unit the input energies are in.
    shift_min_to_zero : bool, optional
        Whether to shift each surface so its minimum is zero.
    max_energy : float, optional
        Mask energies above this value.
    columns : sequence of (str or int), optional
        Columns to use for file sources, ordered ``(cv1, cv2, energy)``.
    levels : int or array_like, optional
        Number of contour lines, or explicit level values. The levels are
        shared by every surface so the comparison is fair.
    colors : sequence, optional
        One colour per surface. Defaults to the active colour cycle.
    x_lab, y_lab : str, optional
        Axis labels. Taken from the data when not given.
    filename : str, optional
        Output path; ``None``, the default, writes nothing.
    show : bool, optional
        Whether to display the figure.
    fig_size : tuple, optional
        Figure size in inches.
    **contour_kwargs
        Extra keyword arguments forwarded to ``contour``/``tricontour``.

    Returns
    -------
    tuple
        ``(fig, ax)`` containing the matplotlib figure and axes.

    Raises
    ------
    ValueError
        If any source is not a 2-D surface.
    """
    fes_list = _as_fes_list(
        sources,
        energy_unit=energy_unit,
        source_unit=source_unit,
        shift_min_to_zero=shift_min_to_zero,
        max_energy=max_energy,
        columns=columns,
    )
    if any(fes.ndim != 2 for fes in fes_list):
        raise ValueError("plot_fes_2d_overlay expects 2-D free-energy surfaces")

    label_list = _resolve_labels(labels, len(fes_list), template=label_template)
    colors = _resolve_colors(colors, len(fes_list))

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    else:
        fig = _figure_from_axes(fig, ax)

    shared_levels = _shared_levels(fes_list, levels)
    for fes, color in zip(fes_list, colors):
        _draw_fes_contour(
            ax,
            fes,
            filled=False,
            levels=shared_levels,
            colors=color,
            **contour_kwargs,
        )

    handles = [
        plt.Line2D([0], [0], color=color, label=label)
        for color, label in zip(colors, label_list)
        if label is not None
    ]
    if handles:
        ax.legend(handles=handles, loc="best")

    reference = fes_list[0]
    _style_axes(
        fig,
        ax,
        x_lab if x_lab is not None else reference.cv_labels[0],
        y_lab if y_lab is not None else reference.cv_labels[1],
    )
    _finalise(fig, filename=filename, show=show)
    return fig, ax


def plot_fes_slices(
    sources: FES | str | Path | np.ndarray | Sequence[Any],
    at: float | Sequence[float] | np.ndarray,
    axis: int = 0,
    fig: Figure | None = None,
    ax: Axes | None = None,
    labels: Sequence[Any] | None = None,
    energy_unit: str | None = None,
    source_unit: str | None = DEFAULT_ENERGY_UNIT,
    shift_min_to_zero: bool = True,
    max_energy: float | None = None,
    columns: Sequence[str | int] | None = None,
    slice_format: str = "{label}, {cv}$={value:.2f}$",
    colors: str | Sequence[str | tuple[float, ...]] | None = None,
    linestyles: Sequence[str] = ("-", "--", ":", "-."),
    x_lab: str | None = None,
    y_lab: str | None = None,
    filename: str | Path | None = None,
    show: bool = False,
    fig_size: tuple[float, float] = (8, 3),
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot 1-D cuts through 2-D free-energy surfaces at fixed CV values.

    Each surface gets its own colour and each requested cut its own line
    style, so several surfaces can be compared at several slices at once.
    Slices are requested by collective-variable *value*; the nearest grid
    point is used and reported in the legend.

    Parameters
    ----------
    sources : FES source or sequence of FES sources
        2-D surfaces to slice; see :func:`as_fes`.
    at : float or sequence of float
        Value(s) of the held collective variable at which to cut.
    axis : int, optional
        Index of the collective variable held fixed, so the default cuts
        along CV2.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on. Inferred from *ax* when omitted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    labels : sequence, optional
        Name for each surface, used in the legend.
    energy_unit : str, optional
        Unit to convert energies to, e.g. ``"eV"``.
    source_unit : str, optional
        Unit the input energies are in.
    shift_min_to_zero : bool, optional
        Whether to shift each surface so its minimum is zero.
    max_energy : float, optional
        Mask energies above this value.
    columns : sequence of (str or int), optional
        Columns to use for file sources, ordered ``(cv1, cv2, energy)``.
    slice_format : str, optional
        ``str.format`` template for the legend entries, receiving ``label``,
        ``cv`` and ``value``.
    colors : sequence, optional
        One colour per surface. Defaults to the active colour cycle.
    linestyles : sequence, optional
        One line style per slice value.
    x_lab, y_lab : str, optional
        Axis labels. Taken from the data when not given.
    filename : str, optional
        Output path; ``None``, the default, writes nothing.
    show : bool, optional
        Whether to display the figure.
    fig_size : tuple, optional
        Figure size in inches.
    **plot_kwargs
        Extra keyword arguments forwarded to ``Axes.plot``.

    Returns
    -------
    tuple
        ``(fig, ax)`` containing the matplotlib figure and axes.

    Raises
    ------
    ValueError
        If any source is not a 2-D surface on a regular grid.
    """
    fes_list = _as_fes_list(
        sources,
        energy_unit=energy_unit,
        source_unit=source_unit,
        shift_min_to_zero=shift_min_to_zero,
        max_energy=max_energy,
        columns=columns,
    )
    if any(fes.ndim != 2 for fes in fes_list):
        raise ValueError("plot_fes_slices expects 2-D free-energy surfaces")

    values = np.atleast_1d(np.asarray(at, dtype=float))
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("at must contain at least one finite CV value")
    label_list = _resolve_labels(labels, len(fes_list))
    colors = _resolve_colors(colors, len(fes_list))
    if not linestyles:
        raise ValueError("linestyles must contain at least one style")

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    else:
        fig = _figure_from_axes(fig, ax)

    reference = fes_list[0]
    for fes, color, label in zip(fes_list, colors, label_list):
        for j, value in enumerate(values):
            x, energy, used = fes.slice_at(value, axis=axis)
            ax.plot(
                x,
                energy,
                color=color,
                linestyle=linestyles[j % len(linestyles)],
                label=slice_format.format(
                    label=label if label is not None else "",
                    cv=fes.cv_labels[axis],
                    value=used,
                ).strip(", "),
                **plot_kwargs,
            )

    ax.legend(loc="best", fontsize=9, ncols=max(1, len(fes_list)))
    _style_axes(
        fig,
        ax,
        x_lab if x_lab is not None else reference.cv_labels[1 - axis],
        y_lab if y_lab is not None else reference.label,
    )
    _finalise(fig, filename=filename, show=show)
    return fig, ax


#: Options understood only by the 2-D plotters, dropped by :func:`plot_fes`
#: when the data turns out to be one dimensional.
_2D_ONLY_KWARGS = ("levels", "cmap", "colorbar")

#: Options consumed while turning a source into an :class:`FES`.
_PREPARE_KWARGS = (
    "energy_unit",
    "source_unit",
    "shift_min_to_zero",
    "max_energy",
    "columns",
    "cv_labels",
    "energy_label",
)


def plot_fes(
    sources: FES | str | Path | np.ndarray | Sequence[Any],
    **kwargs: Any,
) -> tuple[Figure, Axes | np.ndarray]:
    """Plot a free-energy surface, dispatching on its dimensionality.

    Sends 1-D data to :func:`plot_fes_1d` and 2-D data to
    :func:`plot_fes_2d`, which is convenient when the dimensionality is
    decided by the PLUMED input rather than by the caller. Sources are read
    once here and handed on as :class:`FES` objects. Options that only make
    sense for contour plots (``levels``, ``cmap``, ``colorbar``) are ignored
    for 1-D data instead of raising.

    Parameters
    ----------
    sources : FES source or sequence of FES sources
        Paths, arrays or :class:`FES` objects; see :func:`as_fes`.
    **kwargs
        Forwarded to the selected plotting function.

    Returns
    -------
    tuple
        ``(fig, ax)`` containing the matplotlib figure and axes.
    """
    prepare = {key: kwargs.pop(key) for key in _PREPARE_KWARGS if key in kwargs}
    fes_list = _as_fes_list(sources, **prepare)

    if fes_list[0].ndim == 1:
        for key in _2D_ONLY_KWARGS:
            kwargs.pop(key, None)
        return plot_fes_1d(fes_list, **kwargs)
    return plot_fes_2d(fes_list, **kwargs)


def plot_plumed_fes(
    path: str | Path,
    ax: Axes | None = None,
    shift_min_to_zero: bool = True,
    levels: int = 30,
    **kwargs: Any,
) -> tuple[Figure, Axes | np.ndarray]:
    """Plot a PLUMED free-energy surface from a data file.

    Thin wrapper around :func:`plot_fes` kept for the example and test
    workflows. Whether the surface is 1-D or 2-D is determined from the
    file; a 1-D FES is drawn as a line, a 2-D FES as filled contours with a
    colour bar.

    Parameters
    ----------
    path : str
        Path to the PLUMED FES data file.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when ``None``.
    shift_min_to_zero : bool, optional
        Whether to shift the surface so its minimum is zero.
    levels : int, optional
        Number of contour levels for 2-D plots.
    **kwargs
        Further options forwarded to :func:`plot_fes_1d` /
        :func:`plot_fes_2d`, such as ``energy_unit``, ``max_energy``,
        ``filename`` or ``show``.

    Returns
    -------
    tuple
        ``(fig, ax)``, with *ax* an array of axes when several were drawn.
    """
    fig, axes = plot_fes(
        path,
        fig=ax.figure if ax is not None else None,
        ax=ax,
        shift_min_to_zero=shift_min_to_zero,
        levels=levels,
        **kwargs,
    )
    axes = np.atleast_1d(axes)
    return fig, axes[0] if axes.size == 1 else axes


def plot_plumed_colvar(
    path: str | Path,
    x_axis: str = "time",
    columns: Sequence[str] | None = None,
    fig: Figure | None = None,
    axes: Axes | Sequence[Axes] | np.ndarray | None = None,
    filename: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[Figure, np.ndarray]:
    """Plot collective variables from a PLUMED COLVAR file.

    Reads the ``#! FIELDS`` header to determine the column names and creates
    a vertically stacked subplot for each variable.

    Parameters
    ----------
    path : str
        Path to the PLUMED COLVAR file.
    x_axis : str, optional
        Column name to use as the x-axis. If the column is not found, the
        row index is used instead.
    columns : sequence of str, optional
        Restrict the plot to these columns. By default every column other
        than *x_axis* is plotted.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on. Inferred from *axes* when omitted.
    axes : sequence of matplotlib.axes.Axes, optional
        Axes to draw on; must provide one per plotted column.
    filename : str, optional
        Output path; ``None``, the default, writes nothing. A bare stem writes
        both PNG and PDF.
    show : bool, optional
        Whether to display the figure.
    figsize : tuple of float, optional
        Figure size in inches ``(width, height)``.

    Returns
    -------
    tuple
        ``(fig, axes)``, with one axes per plotted variable.

    Raises
    ------
    ValueError
        If no ``#! FIELDS`` header is found, or if there is nothing to plot.
    """
    plumed = read_plumed_file(path, drop_der=False)
    if not plumed.fields:
        raise ValueError(
            f"Could not find a usable '#! FIELDS' header in {path}. "
            "Ensure it is a valid PLUMED file."
        )

    data = plumed.to_dataframe()
    if x_axis in data.columns:
        x_data = data[x_axis]
        x_label = x_axis
    else:
        print(f"Warning: '{x_axis}' column not found. Using index as X-axis.")
        x_data = data.index
        x_label = "Step (index)"

    plot_cols = (
        list(columns) if columns else [col for col in data.columns if col != x_axis]
    )
    missing = [col for col in plot_cols if col not in data.columns]
    if missing:
        raise ValueError(
            f"Columns {missing} not found in {path}. Available: {list(data.columns)}"
        )
    if not plot_cols:
        raise ValueError(f"No variables to plot in {path}")

    if axes is None:
        fig, axes = plt.subplots(
            len(plot_cols), 1, figsize=figsize, sharex=True, constrained_layout=True
        )
    else:
        fig = _figure_from_axes(fig, axes)
    axes = np.atleast_1d(axes).ravel()
    if axes.size < len(plot_cols):
        raise ValueError(f"Got {axes.size} axes for {len(plot_cols)} columns")

    for axis, col in zip(axes, plot_cols):
        axis.plot(x_data, data[col], label=col, linewidth=1.5)
        axis.legend(loc="upper right")
        ax_plot(fig, axis, None, col)

    axes[len(plot_cols) - 1].set_xlabel(x_label)
    _finalise(fig, filename=filename, show=show)
    return fig, axes[: len(plot_cols)]


def plot_fes_convergence(
    sources: FES | str | Path | np.ndarray | Sequence[Any],
    basin_a: Sequence[float],
    basin_b: Sequence[float],
    times: Sequence[float] | np.ndarray | None = None,
    temperature: float | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
    x_lab: str | None = None,
    y_lab: str | None = None,
    filename: str | Path | None = None,
    show: bool = False,
    fig_size: tuple[float, float] = (8, 3),
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot how the barrier and the basin difference settle over a series.

    The convergence test that matters: a run is done when these two numbers
    stop moving, which a series of surfaces plotted on top of each other
    shows only loosely.

    Parameters
    ----------
    sources : sequence of FES source
        The surfaces, in order; see :func:`fes_convergence`.
    basin_a, basin_b : sequence of float
        Basin windows, held fixed across the series.
    times : sequence of float, optional
        X values — simulated time, or hills deposited. ``None``, the default,
        numbers the surfaces from one.
    temperature : float or None, optional
        See :func:`summarise_fes`.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on. Inferred from *ax* when omitted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    x_lab, y_lab : str, optional
        Axis labels. Taken from the data when not given: ``"Surface"``, or
        ``"Time"`` when *times* is given, and the energy unit of the first
        surface.
    filename : str, optional
        Output path; ``None``, the default, writes nothing.
    show : bool, optional
        Whether to display the figure.
    fig_size : tuple, optional
        Figure size in inches.
    **kwargs
        Forwarded to :func:`as_fes`.

    Returns
    -------
    tuple
        ``(fig, ax)`` containing the matplotlib figure and axes.

    Raises
    ------
    ValueError
        If *times* does not have one value per surface.
    """
    summaries = fes_convergence(
        sources, basin_a, basin_b, temperature=temperature, **kwargs
    )
    default_x_lab = "Surface" if times is None else "Time"
    if times is None:
        times = np.arange(1, len(summaries) + 1)
    elif len(times) != len(summaries):
        raise ValueError(f"Got {len(times)} times for {len(summaries)} surfaces.")

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    else:
        fig = _figure_from_axes(fig, ax)

    ax.plot(times, [s.forward_barrier for s in summaries], "o-", label="Barrier A→B")
    ax.plot(times, [s.delta_f for s in summaries], "s-", label="ΔF (B−A)")
    ax.legend(loc="best")

    unit = summaries[0].energy_unit
    _style_axes(
        fig,
        ax,
        x_lab if x_lab is not None else default_x_lab,
        y_lab if y_lab is not None else unit_label(unit),
    )
    _finalise(fig, filename=filename, show=show)
    return fig, ax

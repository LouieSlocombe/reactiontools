"""
Plotting helpers for free-energy surfaces (FES) and other PLUMED output.

The module is organised in three layers so that anything that can be turned
into a free-energy surface can be plotted by the same handful of functions:

1. **Readers** -- :func:`read_plumed_file` parses any PLUMED-style file
   (``COLVAR``, ``fes.dat``, ``HILLS``, ``FES_from_State.py`` output) into a
   :class:`PlumedData` container of numeric columns, field names and
   ``#! SET`` metadata.
2. **Container** -- :class:`FES` normalises 1-D and 2-D free-energy data into
   a common form (collective-variable grids, energies, labels).  Anything a
   user is likely to have -- a file path, a ``(2, N)``/``(N, 2)`` array, a
   stacked ``(3, ny, nx)`` array, a ``(x, y, Z)`` tuple or scattered
   ``(N, 3)`` columns -- is accepted by :func:`as_fes`.
3. **Plotters** -- :func:`plot_fes_1d`, :func:`plot_fes_2d`,
   :func:`plot_fes_2d_overlay` and :func:`plot_fes_slices` each take *one or
   many* FES sources, so a single surface, a convergence series and a
   MD/PIMD comparison are all the same call.  :func:`plot_fes` dispatches on
   dimensionality when the caller does not care.

Every plotting function shares the same conventions:

* sources may be mixed and matched (paths, arrays, :class:`FES` objects),
* ``energy_unit`` converts the file/array energies on the way in,
* ``max_energy`` masks poorly sampled regions instead of letting them
  dominate the colour scale,
* ``filename=None`` means *do not write anything*; passing a name without an
  extension writes every format in ``formats``,
* the return value is always ``(fig, ax)``.

Energies are assumed to be in kJ/mol unless told otherwise, because that is
what PLUMED writes when it is driven from OpenMM.  Runs driven from ASE are in
eV instead, so pass ``source_unit="eV"`` -- which is what the thin wrappers
:func:`~reactiontools.tools_plotting.plot_plumed` and
:func:`~reactiontools.tools_plotting.plot_plumed_multi` do.
"""

import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .tools_style import _finalise, _style_axes, ax_plot

__all__ = [
    "DEFAULT_ENERGY_UNIT",
    "ENERGY_UNITS",
    "FES",
    "PlumedData",
    "as_fes",
    "convert_energy",
    "plot_fes",
    "plot_fes_1d",
    "plot_fes_2d",
    "plot_fes_2d_overlay",
    "plot_fes_slices",
    "plot_plumed_colvar",
    "plot_plumed_fes",
    "read_plumed_file",
    "unit_label",
]

#: Size of one energy unit expressed in kJ/mol.  PLUMED writes kJ/mol by
#: default when driven from OpenMM, which is why it is the reference.
ENERGY_UNITS = {
    "kj/mol": 1.0,
    "kcal/mol": 4.184,
    "ev": 96.48533212331,
    "mev": 0.09648533212331,
    "hartree": 2625.4996394799,
    "kt300": 2.494339,
}

#: Unit the FES files are assumed to be written in when nothing else is said.
DEFAULT_ENERGY_UNIT = "kJ/mol"

#: Pretty names used in axis labels, keyed by the normalised unit name.
_UNIT_LABELS = {
    "kj/mol": "kJ mol$^{-1}$",
    "kcal/mol": "kcal mol$^{-1}$",
    "ev": "eV",
    "mev": "meV",
    "hartree": "$E_\\mathrm{h}$",
    "kt300": "$k_\\mathrm{B}T$",
}


# ---------------------------------------------------------------------------
# Energy units
# ---------------------------------------------------------------------------
def _normalise_unit(unit):
    """
    Normalise an energy-unit name and validate it.

    Parameters
    ----------
    unit : str or None
        Unit name, matched case- and whitespace-insensitively against
        :data:`ENERGY_UNITS`.

    Returns
    -------
    str or None
        The normalised key, or None if *unit* is None.

    Raises
    ------
    KeyError
        If the unit is not known.
    """
    if unit is None:
        return None
    key = str(unit).strip().lower().replace(" ", "")
    if key not in ENERGY_UNITS:
        raise KeyError(f"Unknown energy unit {unit!r}. Known units: {sorted(ENERGY_UNITS)}")
    return key


def unit_label(unit):
    """
    Return the axis label for an energy unit.

    Parameters
    ----------
    unit : str or None
        Energy unit name.

    Returns
    -------
    str
        A LaTeX-ready label such as ``"$F$ (eV)"``, or ``"$F$"`` when the
        unit is unknown.
    """
    key = _normalise_unit(unit)
    if key is None:
        return r"$F$"
    return rf"$F$ ({_UNIT_LABELS[key]})"


def convert_energy(values, source=DEFAULT_ENERGY_UNIT, target=None):
    """
    Convert energies between the units listed in :data:`ENERGY_UNITS`.

    Parameters
    ----------
    values : array_like
        Energies expressed in *source* units.
    source : str, optional
        Unit of *values* (default is ``"kJ/mol"``).
    target : str or None, optional
        Unit to convert to.  ``None`` (default) returns *values* unchanged.

    Returns
    -------
    numpy.ndarray
        The converted energies.
    """
    values = np.asarray(values, dtype=float)
    source_key = _normalise_unit(source)
    target_key = _normalise_unit(target)
    if target_key is None or target_key == source_key:
        return values
    return values * (ENERGY_UNITS[source_key] / ENERGY_UNITS[target_key])


# ---------------------------------------------------------------------------
# PLUMED file reading
# ---------------------------------------------------------------------------
@dataclass
class PlumedData:
    """
    Container for the contents of a PLUMED-style data file.

    Attributes
    ----------
    data : numpy.ndarray
        Numeric columns with shape ``(n_rows, n_fields)``.
    fields : list of str
        Column names taken from the ``#! FIELDS`` header.  Empty when the
        file carries no header.
    metadata : dict
        Key/value pairs collected from ``#! SET key value`` lines.
    """

    data: np.ndarray
    fields: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def index(self, name):
        """
        Return the column index of a field.

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
                raise IndexError(f"Column {index} out of range for {self.data.shape[1]} columns")
            return index % self.data.shape[1]
        if name not in self.fields:
            raise KeyError(f"Field {name!r} not found. Available fields: {self.fields}")
        return self.fields.index(name)

    def column(self, name):
        """
        Return a single column by field name or index.

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

    def label(self, index, default=""):
        """
        Return the field name of a column, falling back to *default*.

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

    def to_dataframe(self):
        """
        Return the data as a :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame
            Columns are named after the fields, or ``col0``, ``col1``, ...
            when no header was present.
        """
        names = list(self.fields) or [f"col{i}" for i in range(self.data.shape[1])]
        return pd.DataFrame(self.data, columns=names)


def read_plumed_file(path, drop_der=True):
    """
    Read a PLUMED-style data file.

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
        ``der_`` (default is True).  Ignored when the header and the data do
        not agree on the number of columns.

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

    with open(path, "r", encoding="utf-8") as handle:
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
    """
    A free-energy surface in a form the plotting functions understand.

    Attributes
    ----------
    cvs : list of numpy.ndarray
        One entry per collective variable.  For a 1-D surface this is a
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
        Explicit colour-bar/y-axis label.  Derived from *energy_unit* when
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

    def __post_init__(self):
        self.cvs = [np.asarray(cv, dtype=float) for cv in self.cvs]
        self.energy = np.asarray(self.energy, dtype=float)
        if not self.cv_labels:
            self.cv_labels = [f"CV{i + 1}" for i in range(len(self.cvs))]

    @property
    def ndim(self):
        """int: Number of collective variables (1 or 2)."""
        return len(self.cvs)

    @property
    def label(self):
        """str: Label to use for the free-energy axis or colour bar."""
        return self.energy_label or unit_label(self.energy_unit)

    def finite_range(self):
        """
        Return the range spanned by the finite energies.

        Returns
        -------
        tuple of float
            ``(minimum, maximum)``, or ``(nan, nan)`` when nothing is finite.
        """
        finite = np.isfinite(self.energy)
        if not finite.any():
            return float("nan"), float("nan")
        return float(np.min(self.energy[finite])), float(np.max(self.energy[finite]))

    def slice_at(self, value, axis=0):
        """
        Take a 1-D cut through a 2-D surface at a fixed value of one CV.

        Parameters
        ----------
        value : float
            Value of the collective variable held fixed.  The nearest grid
            point is used.
        axis : int, optional
            Index of the collective variable held fixed (default is 0).

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
            raise ValueError("Slicing requires a 2-D free-energy surface on a regular grid")

        # Grids are stored as (n_cv2, n_cv1); axis 0 varies along the columns.
        axis_values = self.cvs[axis][0, :] if axis == 0 else self.cvs[axis][:, 0]
        index = int(np.argmin(np.abs(axis_values - value)))
        other = 1 - axis
        if axis == 0:
            return self.cvs[other][:, index], self.energy[:, index], float(axis_values[index])
        return self.cvs[other][index, :], self.energy[index, :], float(axis_values[index])


def _grid_from_columns(x, y, z):
    """
    Reshape scattered column data onto a regular grid when possible.

    Parameters
    ----------
    x, y, z : numpy.ndarray
        Flat coordinate and value arrays of equal length.

    Returns
    -------
    tuple
        ``(X, Y, Z, regular)``.  When the points form a complete rectangular
        grid the arrays are 2-D and *regular* is True, otherwise the inputs
        are returned unchanged with *regular* set to False.
    """
    n_x = np.unique(x).size
    n_y = np.unique(y).size
    if n_x * n_y != z.size:
        return x, y, z, False

    order = np.lexsort((x, y))
    return (x[order].reshape(n_y, n_x),
            y[order].reshape(n_y, n_x),
            z[order].reshape(n_y, n_x),
            True)


def _fes_from_plumed(path, columns=None, cv_labels=None, energy_label=None):
    """
    Build an :class:`FES` from a PLUMED FES file.

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
        return FES(cvs=[x[order]],
                   energy=energies[order],
                   cv_labels=list(cv_labels) if cv_labels else labels[:1],
                   energy_label=energy_label)

    x, y, z, regular = _grid_from_columns(plumed.data[:, indices[0]],
                                          plumed.data[:, indices[1]],
                                          energies)
    return FES(cvs=[x, y],
               energy=z,
               cv_labels=list(cv_labels) if cv_labels else labels[:2],
               energy_label=energy_label,
               regular=regular)


def _fes_from_array(source, cv_labels=None, energy_label=None):
    """
    Build an :class:`FES` from in-memory arrays.

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
        if all(part.ndim == parts[0].ndim and part.shape == parts[0].shape for part in parts):
            source = np.stack(parts)
        else:
            raise ValueError("Sequence FES input requires arrays of identical shape")

    array = np.asarray(source, dtype=float)

    if array.ndim == 3:
        if array.shape[0] != 3:
            raise ValueError(f"3-D FES input must be stacked as (3, ny, nx), got {array.shape}")
        return FES(cvs=[array[0], array[1]],
                   energy=array[2],
                   cv_labels=list(cv_labels) if cv_labels else [],
                   energy_label=energy_label)

    if array.ndim != 2:
        raise ValueError(f"Cannot interpret FES input with shape {array.shape}")

    # Orient so that variables run along the rows.
    if array.shape[0] not in (2, 3):
        if array.shape[1] not in (2, 3):
            raise ValueError(f"Cannot interpret FES input with shape {array.shape}")
        array = array.T

    if array.shape[0] == 2:
        order = np.argsort(array[0])
        return FES(cvs=[array[0][order]],
                   energy=array[1][order],
                   cv_labels=list(cv_labels) if cv_labels else [],
                   energy_label=energy_label)

    x, y, z, regular = _grid_from_columns(array[0], array[1], array[2])
    return FES(cvs=[x, y],
               energy=z,
               cv_labels=list(cv_labels) if cv_labels else [],
               energy_label=energy_label,
               regular=regular)


def as_fes(source,
           energy_unit=None,
           source_unit=DEFAULT_ENERGY_UNIT,
           shift_min_to_zero=True,
           max_energy=None,
           columns=None,
           cv_labels=None,
           energy_label=None):
    """
    Coerce anything FES-shaped into a prepared :class:`FES`.

    This is the single entry point used by every plotting function, so all
    of them accept file paths, raw arrays and :class:`FES` objects
    interchangeably.

    Parameters
    ----------
    source : str, array_like or FES
        A PLUMED FES file, an array in one of the layouts documented in
        :func:`_fes_from_array`, or an already-built surface.
    energy_unit : str or None, optional
        Unit to convert the energies to, e.g. ``"eV"``.  ``None`` (default)
        leaves them untouched.  Surfaces that already carry this unit are
        not converted twice.
    source_unit : str, optional
        Unit the incoming energies are expressed in (default ``"kJ/mol"``,
        which is what PLUMED writes when driven from OpenMM).
    shift_min_to_zero : bool, optional
        Whether to subtract the global minimum so the surface starts at zero
        (default is True).  The operation is idempotent.
    max_energy : float or None, optional
        Energies above this value (in the *output* unit, after shifting) are
        replaced by NaN so that poorly sampled regions do not dominate the
        colour scale.  ``None`` (default) keeps everything.
    columns : sequence of (str or int) or None, optional
        Which file columns to use, ordered ``(cv1, energy)`` or
        ``(cv1, cv2, energy)``.  Only meaningful for file sources.
    cv_labels : sequence of str or None, optional
        Override the collective-variable labels.
    energy_label : str or None, optional
        Override the free-energy label.

    Returns
    -------
    FES
        A new, prepared surface.  The input is never modified in place.
    """
    if isinstance(source, FES):
        fes = FES(cvs=list(source.cvs),
                  energy=source.energy.copy(),
                  cv_labels=list(source.cv_labels),
                  energy_unit=source.energy_unit,
                  energy_label=source.energy_label,
                  regular=source.regular)
        if cv_labels:
            fes.cv_labels = list(cv_labels)
        if energy_label:
            fes.energy_label = energy_label
    elif isinstance(source, (str, os.PathLike)):
        fes = _fes_from_plumed(source, columns=columns, cv_labels=cv_labels, energy_label=energy_label)
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


def _as_fes_list(sources, **kwargs):
    """
    Coerce one source or a collection of sources into a list of surfaces.

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
    """
    if _is_single_source(sources):
        sources = [sources]
    return [as_fes(source, **kwargs) for source in sources]


def _looks_like_fes_array(array):
    """
    Return True when an array can stand on its own as a free-energy surface.

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


def _is_single_source(source):
    """
    Return True when *source* is one FES rather than a collection of them.

    A list or tuple is read as a single surface only when its elements are
    plain coordinate/value arrays -- ``(x, F)``, ``(x, y, z)`` columns or
    ``(X, Y, Z)`` grids.  As soon as an element could itself be a complete
    free-energy surface the sequence is treated as a collection, so
    ``[fes_a, fes_b]`` behaves as expected.  Stack genuinely ambiguous
    grids with :func:`numpy.stack` to force the single-surface reading.

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
    if isinstance(source, (list, tuple)):
        if len(source) not in (2, 3):
            return False
        parts = [part for part in source if isinstance(part, np.ndarray)]
        if len(parts) != len(source):
            return False
        if any(part.shape != parts[0].shape for part in parts):
            return False
        return not _looks_like_fes_array(parts[0])
    return True


def _resolve_labels(labels, count, template=None):
    """
    Build one label per dataset.

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


def _shared_levels(fes_list, levels):
    """
    Build contour levels spanning every surface in *fes_list*.

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
    numpy.ndarray
        The contour levels.
    """
    if not np.isscalar(levels):
        return np.asarray(levels, dtype=float)

    lows, highs = zip(*(fes.finite_range() for fes in fes_list))
    low = np.nanmin(lows)
    high = np.nanmax(highs)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return int(levels)
    return np.linspace(low, high, int(levels))


def _default_grid_size(n_panels, fig_size):
    """
    Pick a figure size that grows with the number of panels.

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
def plot_fes_1d(sources,
                fig=None,
                ax=None,
                labels=None,
                label_template=None,
                max_datasets=None,
                energy_unit=None,
                source_unit=DEFAULT_ENERGY_UNIT,
                shift_min_to_zero=True,
                max_energy=None,
                columns=None,
                x_lab=None,
                y_lab=None,
                filename=None,
                show=False,
                fig_size=(8, 3),
                **plot_kwargs):
    """
    Plot one or more 1-D free-energy profiles on a single axes.

    This covers a single profile, a convergence series over time and a
    method-to-method comparison, since they differ only in their labels.

    Parameters
    ----------
    sources : FES source or sequence of FES sources
        Paths, arrays or :class:`FES` objects; see :func:`as_fes`.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on.  A new one is created when either *fig* or *ax*
        is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    labels : sequence, optional
        Legend entries, one per dataset.  Non-string values are formatted
        with *label_template*, which makes ``labels=times`` with
        ``label_template=r"$t={:g}$ ps"`` a convenient time series.
    label_template : str, optional
        ``str.format`` template applied to non-string *labels*.
    max_datasets : int, optional
        Keep only the last *max_datasets* surfaces.  ``None`` (default)
        plots all of them.
    energy_unit : str, optional
        Unit to convert energies to, e.g. ``"eV"``.
    source_unit : str, optional
        Unit the input energies are in (default ``"kJ/mol"``).
    shift_min_to_zero : bool, optional
        Whether to shift each surface so its minimum is zero (default True).
    max_energy : float, optional
        Mask energies above this value.
    columns : sequence, optional
        Columns to use for file sources, ordered ``(cv, energy)``.
    x_lab, y_lab : str, optional
        Axis labels.  Taken from the data when not given.
    filename : str, optional
        Output path; ``None`` (default) writes nothing.  A bare stem writes
        both PNG and PDF.
    show : bool, optional
        Whether to display the figure (default is False).
    fig_size : tuple, optional
        Figure size in inches (default ``(8, 3)``).
    **plot_kwargs
        Extra keyword arguments forwarded to ``Axes.plot``.

    Returns
    -------
    tuple
        The matplotlib figure and axes.

    Raises
    ------
    ValueError
        If any source is not a 1-D surface.
    """
    fes_list = _as_fes_list(sources,
                            energy_unit=energy_unit,
                            source_unit=source_unit,
                            shift_min_to_zero=shift_min_to_zero,
                            max_energy=max_energy,
                            columns=columns)
    if any(fes.ndim != 1 for fes in fes_list):
        raise ValueError("plot_fes_1d expects 1-D free-energy surfaces; use plot_fes_2d instead")

    label_list = _resolve_labels(labels, len(fes_list), template=label_template)
    if max_datasets is not None and len(fes_list) > max_datasets:
        fes_list = fes_list[-max_datasets:]
        label_list = label_list[-max_datasets:]

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    for fes, label in zip(fes_list, label_list):
        ax.plot(fes.cvs[0], fes.energy, label=label, **plot_kwargs)

    if any(label is not None for label in label_list):
        ax.legend(loc="best")

    reference = fes_list[0]
    _style_axes(fig, ax,
                x_lab if x_lab is not None else reference.cv_labels[0],
                y_lab if y_lab is not None else reference.label)
    _finalise(fig, filename=filename, show=show)
    return fig, ax


def plot_fes_2d(sources,
                fig=None,
                ax=None,
                labels=None,
                label_template=None,
                max_datasets=None,
                energy_unit=None,
                source_unit=DEFAULT_ENERGY_UNIT,
                shift_min_to_zero=True,
                max_energy=None,
                columns=None,
                levels=30,
                cmap=None,
                x_lab=None,
                y_lab=None,
                colorbar=True,
                filename=None,
                show=False,
                fig_size=None,
                **contour_kwargs):
    """
    Plot one or more 2-D free-energy surfaces as filled contours.

    A single surface gives one panel; several surfaces are drawn side by
    side with shared axes, shared contour levels and one colour bar, which
    covers both convergence series and method comparisons.

    Parameters
    ----------
    sources : FES source or sequence of FES sources
        Paths, arrays or :class:`FES` objects; see :func:`as_fes`.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on.  A new one is created when either *fig* or *ax*
        is None.
    ax : matplotlib.axes.Axes or sequence of Axes, optional
        Axes to draw on; must provide one per surface.
    labels : sequence, optional
        Panel titles, one per dataset.  Non-string values are formatted with
        *label_template*.
    label_template : str, optional
        ``str.format`` template applied to non-string *labels*, e.g.
        ``r"$t={:g}$ ps"``.
    max_datasets : int, optional
        Keep only the last *max_datasets* surfaces.  ``None`` (default)
        plots all of them.
    energy_unit : str, optional
        Unit to convert energies to, e.g. ``"eV"``.
    source_unit : str, optional
        Unit the input energies are in (default ``"kJ/mol"``).
    shift_min_to_zero : bool, optional
        Whether to shift each surface so its minimum is zero (default True).
    max_energy : float, optional
        Mask energies above this value, leaving unsampled regions blank.
    columns : sequence, optional
        Columns to use for file sources, ordered ``(cv1, cv2, energy)``.
    levels : int or array_like, optional
        Number of contour levels, or explicit level values (default 30).
        An integer is expanded to levels spanning every panel.
    cmap : str or matplotlib.colors.Colormap, optional
        Colour map passed to ``contourf``.
    x_lab, y_lab : str, optional
        Axis labels.  Taken from the data when not given.
    colorbar : bool, optional
        Whether to draw the shared colour bar (default is True).
    filename : str, optional
        Output path; ``None`` (default) writes nothing.
    show : bool, optional
        Whether to display the figure (default is False).
    fig_size : tuple, optional
        Figure size in inches.  Scales with the number of panels by default.
    **contour_kwargs
        Extra keyword arguments forwarded to ``contourf``/``tricontourf``.

    Returns
    -------
    tuple
        The matplotlib figure and the array of axes (always 1-D, even for a
        single panel).

    Raises
    ------
    ValueError
        If any source is not a 2-D surface, or if the supplied axes do not
        match the number of surfaces.
    """
    fes_list = _as_fes_list(sources,
                            energy_unit=energy_unit,
                            source_unit=source_unit,
                            shift_min_to_zero=shift_min_to_zero,
                            max_energy=max_energy,
                            columns=columns)
    if any(fes.ndim != 2 for fes in fes_list):
        raise ValueError("plot_fes_2d expects 2-D free-energy surfaces; use plot_fes_1d instead")

    label_list = _resolve_labels(labels, len(fes_list), template=label_template)
    if max_datasets is not None and len(fes_list) > max_datasets:
        fes_list = fes_list[-max_datasets:]
        label_list = label_list[-max_datasets:]

    n_panels = len(fes_list)
    if fig is None or ax is None:
        fig, ax = plt.subplots(nrows=1,
                               ncols=n_panels,
                               figsize=_default_grid_size(n_panels, fig_size),
                               sharex=True,
                               sharey=True,
                               constrained_layout=True)
    axes = np.atleast_1d(ax).ravel()
    if axes.size < n_panels:
        raise ValueError(f"Got {axes.size} axes for {n_panels} surfaces")

    shared_levels = _shared_levels(fes_list, levels)
    contours = []
    for axis, fes, label in zip(axes, fes_list, label_list):
        if fes.regular:
            mappable = axis.contourf(fes.cvs[0], fes.cvs[1], fes.energy,
                                     levels=shared_levels, cmap=cmap, **contour_kwargs)
        else:
            finite = np.isfinite(fes.energy)
            mappable = axis.tricontourf(fes.cvs[0][finite], fes.cvs[1][finite], fes.energy[finite],
                                        levels=shared_levels, cmap=cmap, **contour_kwargs)
        contours.append(mappable)
        if label is not None and n_panels > 1:
            axis.set_title(label)

    reference = fes_list[0]
    _style_axes(fig, axes[:n_panels],
                x_lab if x_lab is not None else reference.cv_labels[0],
                y_lab if y_lab is not None else reference.cv_labels[1])

    if colorbar:
        fig.colorbar(contours[-1],
                     ax=list(axes[:n_panels]),
                     orientation="vertical",
                     label=reference.label)

    _finalise(fig, filename=filename, show=show)
    return fig, axes[:n_panels]


def plot_fes_2d_overlay(sources,
                        fig=None,
                        ax=None,
                        labels=None,
                        label_template=None,
                        energy_unit=None,
                        source_unit=DEFAULT_ENERGY_UNIT,
                        shift_min_to_zero=True,
                        max_energy=None,
                        columns=None,
                        levels=6,
                        colors=None,
                        x_lab=None,
                        y_lab=None,
                        filename=None,
                        show=False,
                        fig_size=(5, 4),
                        **contour_kwargs):
    """
    Overlay the contour lines of several 2-D free-energy surfaces.

    Drawing the surfaces on the same axes in different colours makes small
    shifts between them, such as the effect of nuclear quantum effects on a
    barrier, easy to see.

    Parameters
    ----------
    sources : sequence of FES sources
        Paths, arrays or :class:`FES` objects; see :func:`as_fes`.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    labels : sequence, optional
        Legend entries, one per dataset.
    label_template : str, optional
        ``str.format`` template applied to non-string *labels*.
    energy_unit : str, optional
        Unit to convert energies to, e.g. ``"eV"``.
    source_unit : str, optional
        Unit the input energies are in (default ``"kJ/mol"``).
    shift_min_to_zero : bool, optional
        Whether to shift each surface so its minimum is zero (default True).
    max_energy : float, optional
        Mask energies above this value.
    columns : sequence, optional
        Columns to use for file sources, ordered ``(cv1, cv2, energy)``.
    levels : int or array_like, optional
        Number of contour lines, or explicit level values (default 6).  The
        levels are shared by every surface so the comparison is fair.
    colors : sequence, optional
        One colour per surface.  Defaults to the active colour cycle.
    x_lab, y_lab : str, optional
        Axis labels.  Taken from the data when not given.
    filename : str, optional
        Output path; ``None`` (default) writes nothing.
    show : bool, optional
        Whether to display the figure (default is False).
    fig_size : tuple, optional
        Figure size in inches (default ``(5, 4)``).
    **contour_kwargs
        Extra keyword arguments forwarded to ``contour``/``tricontour``.

    Returns
    -------
    tuple
        The matplotlib figure and axes.

    Raises
    ------
    ValueError
        If any source is not a 2-D surface.
    """
    fes_list = _as_fes_list(sources,
                            energy_unit=energy_unit,
                            source_unit=source_unit,
                            shift_min_to_zero=shift_min_to_zero,
                            max_energy=max_energy,
                            columns=columns)
    if any(fes.ndim != 2 for fes in fes_list):
        raise ValueError("plot_fes_2d_overlay expects 2-D free-energy surfaces")

    label_list = _resolve_labels(labels, len(fes_list), template=label_template)
    if colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
        colors = [cycle[i % len(cycle)] for i in range(len(fes_list))]

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    shared_levels = _shared_levels(fes_list, levels)
    for fes, color in zip(fes_list, colors):
        if fes.regular:
            ax.contour(fes.cvs[0], fes.cvs[1], fes.energy,
                       levels=shared_levels, colors=color, **contour_kwargs)
        else:
            finite = np.isfinite(fes.energy)
            ax.tricontour(fes.cvs[0][finite], fes.cvs[1][finite], fes.energy[finite],
                          levels=shared_levels, colors=color, **contour_kwargs)

    handles = [plt.Line2D([0], [0], color=color, label=label)
               for color, label in zip(colors, label_list) if label is not None]
    if handles:
        ax.legend(handles=handles, loc="best")

    reference = fes_list[0]
    _style_axes(fig, ax,
                x_lab if x_lab is not None else reference.cv_labels[0],
                y_lab if y_lab is not None else reference.cv_labels[1])
    _finalise(fig, filename=filename, show=show)
    return fig, ax


def plot_fes_slices(sources,
                    at,
                    axis=0,
                    fig=None,
                    ax=None,
                    labels=None,
                    energy_unit=None,
                    source_unit=DEFAULT_ENERGY_UNIT,
                    shift_min_to_zero=True,
                    max_energy=None,
                    columns=None,
                    slice_format="{label}, {cv}$={value:.2f}$",
                    colors=None,
                    linestyles=("-", "--", ":", "-."),
                    x_lab=None,
                    y_lab=None,
                    filename=None,
                    show=False,
                    fig_size=(8, 3),
                    **plot_kwargs):
    """
    Plot 1-D cuts through 2-D free-energy surfaces at fixed CV values.

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
        Index of the collective variable held fixed (default is 0, i.e.
        slices run along CV2).
    fig : matplotlib.figure.Figure, optional
        Figure to draw on.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    labels : sequence, optional
        Name for each surface, used in the legend.
    energy_unit : str, optional
        Unit to convert energies to, e.g. ``"eV"``.
    source_unit : str, optional
        Unit the input energies are in (default ``"kJ/mol"``).
    shift_min_to_zero : bool, optional
        Whether to shift each surface so its minimum is zero (default True).
    max_energy : float, optional
        Mask energies above this value.
    columns : sequence, optional
        Columns to use for file sources, ordered ``(cv1, cv2, energy)``.
    slice_format : str, optional
        ``str.format`` template for the legend entries, receiving ``label``,
        ``cv`` and ``value``.
    colors : sequence, optional
        One colour per surface.  Defaults to the active colour cycle.
    linestyles : sequence, optional
        One line style per slice value.
    x_lab, y_lab : str, optional
        Axis labels.  Taken from the data when not given.
    filename : str, optional
        Output path; ``None`` (default) writes nothing.
    show : bool, optional
        Whether to display the figure (default is False).
    fig_size : tuple, optional
        Figure size in inches (default ``(8, 3)``).
    **plot_kwargs
        Extra keyword arguments forwarded to ``Axes.plot``.

    Returns
    -------
    tuple
        The matplotlib figure and axes.

    Raises
    ------
    ValueError
        If any source is not a 2-D surface on a regular grid.
    """
    fes_list = _as_fes_list(sources,
                            energy_unit=energy_unit,
                            source_unit=source_unit,
                            shift_min_to_zero=shift_min_to_zero,
                            max_energy=max_energy,
                            columns=columns)
    if any(fes.ndim != 2 for fes in fes_list):
        raise ValueError("plot_fes_slices expects 2-D free-energy surfaces")

    values = np.atleast_1d(np.asarray(at, dtype=float))
    label_list = _resolve_labels(labels, len(fes_list))
    if colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
        colors = [cycle[i % len(cycle)] for i in range(len(fes_list))]

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    reference = fes_list[0]
    for fes, color, label in zip(fes_list, colors, label_list):
        for j, value in enumerate(values):
            x, energy, used = fes.slice_at(value, axis=axis)
            ax.plot(x, energy,
                    color=color,
                    linestyle=linestyles[j % len(linestyles)],
                    label=slice_format.format(label=label if label is not None else "",
                                              cv=fes.cv_labels[axis],
                                              value=used).strip(", "),
                    **plot_kwargs)

    ax.legend(loc="best", fontsize=9, ncols=max(1, len(fes_list)))
    _style_axes(fig, ax,
                x_lab if x_lab is not None else reference.cv_labels[1 - axis],
                y_lab if y_lab is not None else reference.label)
    _finalise(fig, filename=filename, show=show)
    return fig, ax


#: Options understood only by the 2-D plotters, dropped by :func:`plot_fes`
#: when the data turns out to be one dimensional.
_2D_ONLY_KWARGS = ("levels", "cmap", "colorbar")

#: Options consumed while turning a source into an :class:`FES`.
_PREPARE_KWARGS = ("energy_unit", "source_unit", "shift_min_to_zero", "max_energy",
                   "columns", "cv_labels", "energy_label")


def plot_fes(sources, **kwargs):
    """
    Plot a free-energy surface, dispatching on its dimensionality.

    Sends 1-D data to :func:`plot_fes_1d` and 2-D data to
    :func:`plot_fes_2d`, which is convenient when the dimensionality is
    decided by the PLUMED input rather than by the caller.  Sources are read
    once here and handed on as :class:`FES` objects.  Options that only make
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
        The matplotlib figure and axes.
    """
    prepare = {key: kwargs.pop(key) for key in _PREPARE_KWARGS if key in kwargs}
    fes_list = _as_fes_list(sources, **prepare)

    if fes_list[0].ndim == 1:
        for key in _2D_ONLY_KWARGS:
            kwargs.pop(key, None)
        return plot_fes_1d(fes_list, **kwargs)
    return plot_fes_2d(fes_list, **kwargs)


def plot_plumed_fes(path,
                    ax=None,
                    shift_min_to_zero=True,
                    levels=30,
                    **kwargs):
    """
    Plot a PLUMED free-energy surface from a data file.

    Thin wrapper around :func:`plot_fes` kept for the example and test
    workflows.  Whether the surface is 1-D or 2-D is determined from the
    file; a 1-D FES is drawn as a line, a 2-D FES as filled contours with a
    colour bar.

    Parameters
    ----------
    path : str
        Path to the PLUMED FES data file.
    ax : matplotlib.axes.Axes or None, optional
        Axes on which to draw.  A new figure is created when None.
    shift_min_to_zero : bool, optional
        Whether to shift the surface so its minimum is zero (default True).
    levels : int, optional
        Number of contour levels for 2-D plots (default is 30).
    **kwargs
        Further options forwarded to :func:`plot_fes_1d` /
        :func:`plot_fes_2d`, such as ``energy_unit``, ``max_energy``,
        ``filename`` or ``show``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure.
    ax : matplotlib.axes.Axes
        The matplotlib axes, or an array of axes if several were drawn.
    """
    fig, axes = plot_fes(path,
                         fig=ax.figure if ax is not None else None,
                         ax=ax,
                         shift_min_to_zero=shift_min_to_zero,
                         levels=levels,
                         **kwargs)
    axes = np.atleast_1d(axes)
    return fig, axes[0] if axes.size == 1 else axes


def plot_plumed_colvar(path,
                       x_axis="time",
                       columns=None,
                       fig=None,
                       axes=None,
                       filename=None,
                       show=False,
                       figsize=(10, 8)):
    """
    Plot collective variables from a PLUMED COLVAR file.

    Reads the ``#! FIELDS`` header to determine the column names and creates
    a vertically stacked subplot for each variable.

    Parameters
    ----------
    path : str
        Path to the PLUMED COLVAR file.
    x_axis : str, optional
        Column name to use as the x-axis.  If the column is not found, the
        row index is used instead.  Default is ``'time'``.
    columns : sequence of str, optional
        Restrict the plot to these columns.  By default every column other
        than *x_axis* is plotted.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on.  A new one is created when either *fig* or
        *axes* is None.
    axes : sequence of matplotlib.axes.Axes, optional
        Axes to draw on; must provide one per plotted column.
    filename : str, optional
        Output path; ``None`` (default) writes nothing.  A bare stem writes
        both PNG and PDF.
    show : bool, optional
        Whether to display the figure (default is False).
    figsize : tuple of float, optional
        Figure size in inches ``(width, height)``.  Default is ``(10, 8)``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure.
    axes : numpy.ndarray of matplotlib.axes.Axes
        One axes per plotted variable.

    Raises
    ------
    ValueError
        If no ``#! FIELDS`` header is found, or if there is nothing to plot.
    """
    plumed = read_plumed_file(path, drop_der=False)
    if not plumed.fields:
        raise ValueError(f"Could not find a usable '#! FIELDS' header in {path}. "
                         "Ensure it is a valid PLUMED file.")

    data = plumed.to_dataframe()
    if x_axis in data.columns:
        x_data = data[x_axis]
        x_label = x_axis
    else:
        print(f"Warning: '{x_axis}' column not found. Using index as X-axis.")
        x_data = data.index
        x_label = "Step (index)"

    plot_cols = list(columns) if columns else [col for col in data.columns if col != x_axis]
    missing = [col for col in plot_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found in {path}. Available: {list(data.columns)}")
    if not plot_cols:
        raise ValueError(f"No variables to plot in {path}")

    if fig is None or axes is None:
        fig, axes = plt.subplots(len(plot_cols), 1, figsize=figsize, sharex=True,
                                 constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    if axes.size < len(plot_cols):
        raise ValueError(f"Got {axes.size} axes for {len(plot_cols)} columns")

    for axis, col in zip(axes, plot_cols):
        axis.plot(x_data, data[col], label=col, linewidth=1.5)
        axis.legend(loc="upper right")
        ax_plot(fig, axis, None, col)

    axes[len(plot_cols) - 1].set_xlabel(x_label)
    _finalise(fig, filename=filename, show=show)
    return fig, axes[:len(plot_cols)]

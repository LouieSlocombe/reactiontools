"""Energy units, and the conversions between them.

Every part of this package that reads a number off a file or writes one into
an input has to agree about what the number means, and the two simulation
worlds this package spans do not agree by default: PLUMED driven from OpenMM
writes kJ/mol and nanometres, PLUMED driven from ASE writes eV and angstrom.
:func:`convert_energy` is the single place that reconciles them, and
:data:`DEFAULT_ENERGY_UNIT` records which one the rest of the package assumes
when nothing is said.

This lives in its own module rather than in :mod:`reactiontools.tools_fes`,
where it grew up, because the script builders in
:mod:`reactiontools.tools_cv` and :mod:`reactiontools.tools_plumed` need the
conversions but must not drag in matplotlib -- and importing anything from
``tools_fes`` does exactly that. It is the same argument
:mod:`reactiontools.tools_style` makes for existing at all.
"""

import numpy as np
from ase.units import kB

__all__ = [
    "DEFAULT_ENERGY_UNIT",
    "ENERGY_UNITS",
    "as_kelvin",
    "convert_energy",
    "thermal_energy",
    "unit_label",
]

#: Size of one energy unit expressed in kJ/mol. PLUMED writes kJ/mol by
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


def _normalise_unit(unit):
    """Normalise an energy-unit name and validate it.

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
    """Return the axis label for an energy unit.

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
    """Convert energies between the units listed in :data:`ENERGY_UNITS`.

    Parameters
    ----------
    values : array_like
        Energies expressed in *source* units.
    source : str, optional
        Unit of *values*.
    target : str or None, optional
        Unit to convert to. ``None``, the default, returns *values* unchanged.

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


def as_kelvin(temperature):
    """Return a temperature in kelvin, from a bare number or an OpenMM quantity.

    Everything in this package works in plain kelvin, but the callers driving
    OpenMM have an ``openmm.unit.Quantity`` to hand and would otherwise have to
    unwrap it at every call. The OpenMM branch is chosen by looking at which
    package built the object, and imports lazily, so nothing here depends on
    OpenMM being installed.

    Parameters
    ----------
    temperature : float or openmm.unit.Quantity
        Temperature. A bare number is taken to be in kelvin already.

    Returns
    -------
    float
        The temperature in kelvin.
    """
    if type(temperature).__module__.split(".")[0] == "openmm":
        from openmm import unit as openmm_unit
        return float(temperature.value_in_unit(openmm_unit.kelvin))
    return float(temperature)


def thermal_energy(temperature, energy_unit=DEFAULT_ENERGY_UNIT):
    """Return kBT at *temperature*, in *energy_unit*.

    This is the ``--kt`` that ``plumed sum_hills`` and the bundled OPES
    scripts are given when they integrate a variable out of a surface, and it
    has to be in the same units as the file they are reading: kJ/mol for a run
    driven from OpenMM, eV for one driven from ASE via
    :data:`~reactiontools.tools_plumed.PLUMED_ASE_UNITS`. Getting it wrong
    does not fail, it reweights by the wrong temperature.

    Parameters
    ----------
    temperature : float or openmm.unit.Quantity
        Temperature in kelvin; see :func:`as_kelvin` for what is accepted.
    energy_unit : str, optional
        Unit to return the result in, from :data:`ENERGY_UNITS`.

    Returns
    -------
    float
        kBT at that temperature.

    Examples
    --------
    >>> round(float(thermal_energy(300.0)), 4)
    2.4943
    """
    return float(convert_energy(kB * as_kelvin(temperature),
                                source="eV", target=energy_unit))

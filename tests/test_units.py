"""Tests for the energy units and the thermal energy built on them."""

import numpy as np
import pytest

from reactiontools import (
    DEFAULT_ENERGY_UNIT,
    ENERGY_UNITS,
    as_kelvin,
    convert_energy,
    thermal_energy,
    unit_label,
)


class TestConvertEnergy:
    def test_converting_to_the_same_unit_changes_nothing(self) -> None:
        values = [1.0, 2.5, -3.0]

        assert convert_energy(values, "eV", "eV").tolist() == values

    def test_no_target_returns_the_values_unchanged(self) -> None:
        values = [1.0, 2.5, -3.0]

        assert convert_energy(values, "eV").tolist() == values

    @pytest.mark.parametrize("unit", sorted(ENERGY_UNITS))
    def test_every_unit_round_trips_through_the_default(self, unit: str) -> None:
        there = convert_energy(1.0, DEFAULT_ENERGY_UNIT, unit)
        back = convert_energy(there, unit, DEFAULT_ENERGY_UNIT)

        assert back == pytest.approx(1.0)

    def test_an_unknown_unit_names_the_ones_that_are_known(self) -> None:
        with pytest.raises(KeyError, match="kcal/mol"):
            convert_energy(1.0, "furlongs")


class TestThermalEnergy:
    def test_kbt_at_300_kelvin_in_kilojoules_per_mole(self) -> None:
        # 300 K is the temperature nearly every run here uses, and 2.4943
        # kJ/mol is the number that ends up in the `--kt` of sum_hills.
        assert thermal_energy(300.0) == pytest.approx(2.4943, abs=1e-4)

    def test_kbt_at_300_kelvin_in_electronvolts(self) -> None:
        # What a run driven from ASE needs instead, PLUMED_ASE_UNITS being eV.
        assert thermal_energy(300.0, "eV") == pytest.approx(0.025852, abs=1e-6)

    def test_the_default_unit_is_the_package_default(self) -> None:
        assert thermal_energy(300.0) == thermal_energy(300.0, DEFAULT_ENERGY_UNIT)

    def test_it_scales_with_temperature(self) -> None:
        assert thermal_energy(600.0) == pytest.approx(2 * thermal_energy(300.0))

    def test_it_returns_a_float_not_an_array(self) -> None:
        # It is interpolated straight into a shell command, where a numpy
        # scalar's repr would be wrong.
        assert isinstance(thermal_energy(300.0), float)

    def test_it_accepts_anything_float_accepts(self) -> None:
        assert thermal_energy(np.float64(300.0)) == pytest.approx(thermal_energy(300.0))

    def test_an_unknown_unit_is_rejected(self) -> None:
        with pytest.raises(KeyError):
            thermal_energy(300.0, "furlongs")

    def test_an_openmm_quantity_is_accepted(self) -> None:
        # Callers driving OpenMM have a Quantity to hand; making them unwrap
        # it at every call would be the only reason they ever touch
        # openmm.unit here.
        openmm_unit = pytest.importorskip("openmm.unit")

        assert thermal_energy(300.0 * openmm_unit.kelvin) == pytest.approx(
            thermal_energy(300.0)
        )


class TestAsKelvin:
    def test_a_bare_number_is_already_kelvin(self) -> None:
        assert as_kelvin(300.0) == 300.0

    def test_it_returns_a_float(self) -> None:
        assert isinstance(as_kelvin(300), float)

    def test_an_openmm_quantity_is_unwrapped(self) -> None:
        openmm_unit = pytest.importorskip("openmm.unit")

        assert as_kelvin(300.0 * openmm_unit.kelvin) == pytest.approx(300.0)

    def test_other_temperature_units_are_converted_not_stripped(self) -> None:
        openmm_unit = pytest.importorskip("openmm.unit")

        # OpenMM knows this is 300 K expressed differently; taking the bare
        # number would give 26.85.
        quantity = (300.0 * openmm_unit.kelvin).in_units_of(openmm_unit.kelvin)
        assert as_kelvin(quantity) == pytest.approx(300.0)


class TestUnitLabel:
    def test_a_known_unit_is_wrapped_for_an_axis(self) -> None:
        assert unit_label("eV") == r"$F$ (eV)"

    def test_no_unit_gives_a_bare_symbol(self) -> None:
        assert unit_label(None) == r"$F$"

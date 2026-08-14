"""Tests for the energy units and the thermal energy built on them."""

import numpy as np
import pytest

from reactiontools import (DEFAULT_ENERGY_UNIT,
                           ENERGY_UNITS,
                           convert_energy,
                           thermal_energy,
                           unit_label)


class TestConvertEnergy:
    def test_converting_to_the_same_unit_changes_nothing(self):
        values = [1.0, 2.5, -3.0]

        assert convert_energy(values, "eV", "eV").tolist() == values

    def test_no_target_returns_the_values_unchanged(self):
        values = [1.0, 2.5, -3.0]

        assert convert_energy(values, "eV").tolist() == values

    @pytest.mark.parametrize("unit", sorted(ENERGY_UNITS))
    def test_every_unit_round_trips_through_the_default(self, unit):
        there = convert_energy(1.0, DEFAULT_ENERGY_UNIT, unit)
        back = convert_energy(there, unit, DEFAULT_ENERGY_UNIT)

        assert back == pytest.approx(1.0)

    def test_an_unknown_unit_names_the_ones_that_are_known(self):
        with pytest.raises(KeyError, match="kcal/mol"):
            convert_energy(1.0, "furlongs")


class TestThermalEnergy:
    def test_kbt_at_300_kelvin_in_kilojoules_per_mole(self):
        # 300 K is the temperature nearly every run here uses, and 2.4943
        # kJ/mol is the number that ends up in the `--kt` of sum_hills.
        assert thermal_energy(300.0) == pytest.approx(2.4943, abs=1e-4)

    def test_kbt_at_300_kelvin_in_electronvolts(self):
        # What a run driven from ASE needs instead, PLUMED_ASE_UNITS being eV.
        assert thermal_energy(300.0, "eV") == pytest.approx(0.025852, abs=1e-6)

    def test_the_default_unit_is_the_package_default(self):
        assert thermal_energy(300.0) == thermal_energy(300.0, DEFAULT_ENERGY_UNIT)

    def test_it_scales_with_temperature(self):
        assert thermal_energy(600.0) == pytest.approx(2 * thermal_energy(300.0))

    def test_it_returns_a_float_not_an_array(self):
        # It is interpolated straight into a shell command, where a numpy
        # scalar's repr would be wrong.
        assert isinstance(thermal_energy(300.0), float)

    def test_it_accepts_anything_float_accepts(self):
        assert thermal_energy(np.float64(300.0)) == pytest.approx(thermal_energy(300.0))

    def test_an_unknown_unit_is_rejected(self):
        with pytest.raises(KeyError):
            thermal_energy(300.0, "furlongs")


class TestUnitLabel:
    def test_a_known_unit_is_wrapped_for_an_axis(self):
        assert unit_label("eV") == r"$F$ (eV)"

    def test_no_unit_gives_a_bare_symbol(self):
        assert unit_label(None) == r"$F$"

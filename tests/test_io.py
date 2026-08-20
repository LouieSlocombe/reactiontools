"""Tests for the structure-file readers and writers."""

from pathlib import Path

import pytest
from ase.build import molecule

from reactiontools import (
    convert_pdb_to_xyz,
    convert_xyz_to_pdb,
    convert_xyz_to_plumed_ref,
    element_from_pdb_line,
    format_pdb_atom_name,
    pdb_remove_ter_index,
    strip_hydrogens_keep_indices,
    write_xyz_frame,
)


def _pdb_line(
    serial: int = 1,
    name: str = " C1 ",
    resname: str = "AAA",
    chain: str = "A",
    resid: int = 1,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    element: str = "C",
) -> str:
    """One ATOM record, built by column so the tests can vary one field."""
    return (
        f"ATOM  {serial:>5} {name} {resname:>3} {chain}{resid:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2}\n"
    )


class TestElementFromPdbLine:
    def test_the_element_column_wins_when_it_is_there(self) -> None:
        line = _pdb_line(name=" CA ", element="CA")

        assert element_from_pdb_line(line) == "Ca"

    def test_a_missing_element_column_falls_back_to_the_name(self) -> None:
        line = _pdb_line(name=" CA ", element="")

        assert element_from_pdb_line(line) == "C"

    def test_alignment_separates_calcium_from_an_alpha_carbon(self) -> None:
        alpha_carbon = _pdb_line(name=" CA ", element="")
        calcium = _pdb_line(name="CA  ", element="")

        assert element_from_pdb_line(alpha_carbon) == "C"
        assert element_from_pdb_line(calcium) == "Ca"

    @pytest.mark.parametrize("name", ["HG  ", "HE  ", "HF  ", "HO  ", "HS  "])
    def test_hydrogen_lookalikes_resolve_to_hydrogen(self, name: str) -> None:
        # A PDB with no element column is far likelier to hold a gamma
        # hydrogen than mercury.
        assert element_from_pdb_line(_pdb_line(name=name, element="")) == "H"

    def test_a_digit_prefixed_name_carries_a_one_letter_symbol(self) -> None:
        assert element_from_pdb_line(_pdb_line(name="1HB ", element="")) == "H"

    def test_a_nameless_record_gives_up_rather_than_guessing(self) -> None:
        assert element_from_pdb_line(_pdb_line(name="    ", element="")) == "X"


class TestFormatPdbAtomName:
    def test_one_letter_symbols_are_indented_by_a_column(self) -> None:
        assert format_pdb_atom_name("C", 1) == " C1 "

    def test_two_letter_symbols_start_in_the_first_column(self) -> None:
        assert format_pdb_atom_name("Cl", 1) == "Cl1 "

    def test_a_long_name_is_truncated_to_keep_the_columns_aligned(self) -> None:
        assert len(format_pdb_atom_name("Cl", 12345)) == 4


class TestRoundTrip:
    def test_xyz_survives_a_trip_through_pdb(self, tmp_path: Path) -> None:
        water = molecule("H2O")
        xyz = tmp_path / "in.xyz"
        with open(xyz, "w") as handle:
            write_xyz_frame(handle, water.get_chemical_symbols(), water.positions)

        pdb = tmp_path / "out.pdb"
        n_clusters = convert_xyz_to_pdb(xyz, pdb)

        back = tmp_path / "back.xyz"
        n_frames = convert_pdb_to_xyz(pdb, back)

        assert n_clusters == 1
        assert n_frames == 1
        assert back.read_text().splitlines()[0].strip() == "3"

    def test_a_multi_model_pdb_gives_one_frame_per_model(self, tmp_path: Path) -> None:
        pdb = tmp_path / "path.pdb"
        pdb.write_text(
            "".join(
                f"MODEL     {model}\n{_pdb_line(x=float(model))}ENDMDL\n"
                for model in (1, 2, 3)
            )
        )

        assert convert_pdb_to_xyz(pdb, tmp_path / "path.xyz") == 3

    def test_a_pdb_with_no_atoms_is_an_error_not_an_empty_file(
        self, tmp_path: Path
    ) -> None:
        pdb = tmp_path / "empty.pdb"
        pdb.write_text("REMARK nothing here\nEND\n")

        with pytest.raises(ValueError, match="No ATOM or HETATM"):
            convert_pdb_to_xyz(pdb, tmp_path / "empty.xyz")


class TestPdbRemoveTerIndex:
    def test_serials_are_renumbered_from_one(self, tmp_path: Path) -> None:
        pdb = tmp_path / "in.pdb"
        pdb.write_text(_pdb_line(serial=17) + _pdb_line(serial=42) + "END\n")

        pdb_remove_ter_index(pdb, pdb)

        serials = [
            line[6:11].strip()
            for line in pdb.read_text().splitlines()
            if line.startswith("ATOM")
        ]
        assert serials == ["1", "2"]

    def test_numbering_restarts_at_each_model(self, tmp_path: Path) -> None:
        pdb = tmp_path / "in.pdb"
        pdb.write_text(
            "MODEL     1\n" + _pdb_line(serial=5) + _pdb_line(serial=6) + "ENDMDL\n"
            "MODEL     2\n" + _pdb_line(serial=7) + _pdb_line(serial=8) + "ENDMDL\n"
        )

        pdb_remove_ter_index(pdb, pdb)

        serials = [
            line[6:11].strip()
            for line in pdb.read_text().splitlines()
            if line.startswith("ATOM")
        ]
        assert serials == ["1", "2", "1", "2"]

    def test_conect_records_follow_the_atoms_they_point_at(
        self, tmp_path: Path
    ) -> None:
        pdb = tmp_path / "in.pdb"
        pdb.write_text(
            _pdb_line(serial=17) + _pdb_line(serial=42) + "CONECT   17   42\n"
        )

        pdb_remove_ter_index(pdb, pdb)

        conect = next(
            line for line in pdb.read_text().splitlines() if line.startswith("CONECT")
        )
        assert conect.split() == ["CONECT", "1", "2"]

    def test_a_ter_shares_the_serial_of_the_atom_after_it(self, tmp_path: Path) -> None:
        pdb = tmp_path / "in.pdb"
        pdb.write_text(
            _pdb_line(serial=1) + "TER       2      AAA A   1\n" + _pdb_line(serial=3)
        )

        pdb_remove_ter_index(pdb, pdb)

        lines = pdb.read_text().splitlines()
        ter = next(line for line in lines if line.startswith("TER"))
        # The TER does not consume a serial, so the atom after it gets 2 and
        # the TER carries the same number.
        assert ter[6:11].strip() == "2"
        assert lines[-1][6:11].strip() == "2"


class TestStripHydrogensKeepIndices:
    def test_every_hydrogen_goes_when_nothing_is_kept(self, tmp_path: Path) -> None:
        pdb = tmp_path / "in.pdb"
        pdb.write_text(
            _pdb_line(serial=1, element="O")
            + _pdb_line(serial=2, element="H")
            + _pdb_line(serial=3, element="H")
        )

        out = tmp_path / "out.pdb"
        strip_hydrogens_keep_indices(pdb, out)

        assert len(out.read_text().splitlines()) == 1

    def test_the_named_hydrogens_stay(self, tmp_path: Path) -> None:
        pdb = tmp_path / "in.pdb"
        pdb.write_text(
            _pdb_line(serial=1, element="O")
            + _pdb_line(serial=2, element="H")
            + _pdb_line(serial=3, element="H")
        )

        out = tmp_path / "out.pdb"
        # keep is 0-based, so index 1 is the atom with serial 2.
        strip_hydrogens_keep_indices(pdb, out, keep=[1])

        serials = [line[6:11].strip() for line in out.read_text().splitlines()]
        assert serials == ["1", "2"]

    def test_non_atom_records_are_passed_through(self, tmp_path: Path) -> None:
        pdb = tmp_path / "in.pdb"
        pdb.write_text("MODEL     1\n" + _pdb_line(element="H") + "ENDMDL\nEND\n")

        out = tmp_path / "out.pdb"
        strip_hydrogens_keep_indices(pdb, out)

        assert out.read_text() == "MODEL     1\nENDMDL\nEND\n"


class TestConvertXyzToPlumedRef:
    def test_each_frame_becomes_a_model_with_the_template_records(
        self, tmp_path: Path
    ) -> None:
        template = tmp_path / "index_atoms.pdb"
        template.write_text(
            _pdb_line(serial=1, name=" O1 ", element="O")
            + _pdb_line(serial=2, name=" H1 ", element="H")
        )

        xyz = tmp_path / "path.xyz"
        with open(xyz, "w") as handle:
            for step in range(3):
                write_xyz_frame(
                    handle,
                    ["O", "H"],
                    [(0.0, 0.0, 0.0), (float(step), 0.0, 0.0)],
                    comment=f"image {step}",
                )

        out = tmp_path / "neb_path.pdb"
        convert_xyz_to_plumed_ref(xyz, template, out, atom_line="ATOM")

        text = out.read_text()
        assert "REMARK TYPE=MULTI-ST-PDB" in text
        assert "REMARK ARG=path.s,path.z" in text
        assert text.count("ENDMDL") == 3
        # The template's names survive; only the coordinates are substituted.
        assert text.count(" O1 ") == 3
        assert " 2.000   0.000   0.000" in text

    def test_the_reference_and_its_template_end_up_numbered_alike(
        self, tmp_path: Path
    ) -> None:
        template = tmp_path / "index_atoms.pdb"
        template.write_text(
            _pdb_line(serial=90, name=" O1 ", element="O")
            + _pdb_line(serial=91, name=" H1 ", element="H")
        )

        xyz = tmp_path / "path.xyz"
        with open(xyz, "w") as handle:
            write_xyz_frame(handle, ["O", "H"], [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])

        out = tmp_path / "neb_path.pdb"
        convert_xyz_to_plumed_ref(xyz, template, out, atom_line="ATOM")

        # PLUMED matches reference to template by serial, so both are
        # renumbered from 1 on the way out. This is the whole point.
        template_serials = [
            line[6:11].strip()
            for line in template.read_text().splitlines()
            if line.startswith("ATOM")
        ]
        out_serials = [
            line[6:11].strip()
            for line in out.read_text().splitlines()
            if line.startswith("ATOM")
        ]
        assert template_serials == ["1", "2"]
        assert out_serials == ["1", "2"]

    def test_accepts_multiple_template_record_types(self, tmp_path: Path) -> None:
        template = tmp_path / "index_atoms.pdb"
        template.write_text(
            _pdb_line(serial=1, name=" O1 ", element="O")
            + _pdb_line(serial=2, name=" H1 ", element="H").replace(
                "ATOM  ", "HETATM", 1
            )
        )
        xyz = tmp_path / "path.xyz"
        with open(xyz, "w") as handle:
            write_xyz_frame(handle, ["O", "H"], [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])

        out = tmp_path / "neb_path.pdb"
        convert_xyz_to_plumed_ref(xyz, template, out, atom_line=("ATOM", "HETATM"))

        atom_lines = [
            line
            for line in out.read_text().splitlines()
            if line.startswith(("ATOM", "HETATM"))
        ]
        assert len(atom_lines) == 2

    def test_an_empty_xyz_writes_nothing_rather_than_failing(
        self, tmp_path: Path
    ) -> None:
        template = tmp_path / "index_atoms.pdb"
        template.write_text(_pdb_line())
        xyz = tmp_path / "empty.xyz"
        xyz.write_text("")

        out = tmp_path / "out.pdb"
        convert_xyz_to_plumed_ref(xyz, template, out, atom_line="ATOM")

        assert not out.exists()

"""Structure files: XYZ, PDB, and the reference a path collective variable reads.

The formats a reaction path passes through on its way from a band of images to
something PLUMED will bias. A band comes out of :mod:`reactiontools.tools_reaction`
as ASE images and is written as XYZ; ``PATHMSD`` wants a multi-model PDB whose
atom serials match, to the number, the template the run is aligned against with
``FIT_TO_TEMPLATE``. :func:`convert_xyz_to_plumed_ref` is what bridges the two,
and :func:`pdb_remove_ter_index` is why it can promise the numbering agrees.

The PDB reading here is deliberately by column rather than through ASE. PLUMED
identifies atoms by serial, so the atom records have to survive the round trip
with their names, residues and numbering exactly as written -- and a parser that
understands a PDB well enough to rebuild it is a parser that will quietly
normalise something. Reading columns 31-54 and putting them back is the whole
job.

Coordinates are in angstrom throughout, which is what both formats use.
"""

import os
import string
from collections import defaultdict

import numpy as np
from ase.data import chemical_symbols
from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

__all__ = [
    "convert_pdb_to_xyz",
    "convert_xyz_to_pdb",
    "convert_xyz_to_plumed_ref",
    "element_from_pdb_line",
    "format_pdb_atom_name",
    "pdb_remove_ter_index",
    "strip_hydrogens_keep_indices",
    "write_xyz_frame",
]

# Two-letter element symbols that are also common prefixes of hydrogen atom
# names, e.g. "HG21" is a gamma hydrogen rather than mercury. These are only
# consulted when a PDB line has no element column to fall back on.
_HYDROGEN_LOOKALIKES = {"He", "Hf", "Hg", "Ho", "Hs"}


def element_from_pdb_line(line):
    """
    Determine the element symbol for a PDB ``ATOM``/``HETATM`` line.

    The element column (77-78) is used when present. Otherwise the symbol is
    inferred from the atom name in columns 13-16, which by convention starts in
    column 13 for two-character symbols and column 14 for one-character ones.

    Parameters
    ----------
    line : str
        A single ``ATOM`` or ``HETATM`` record.

    Returns
    -------
    str
        The element symbol, or ``'X'`` if none could be determined.

    Notes
    -----
    The atom-name fallback relies on the name being correctly aligned, which
    is what distinguishes calcium (``CA  ``) from an alpha carbon (`` CA ``).
    Hydrogen names that collide with two-letter symbols -- ``HE``, ``HF``,
    ``HG``, ``HO``, ``HS`` -- are resolved as hydrogen, since a PDB with no
    element column is far likelier to hold a gamma hydrogen than mercury.
    """
    symbol = line[76:78].strip()
    if symbol:
        return symbol.capitalize()

    # No element column, so fall back to the atom name
    name = line[12:16]
    letters = "".join(c for c in name if c.isalpha())
    if not letters:
        return "X"

    # A name indented or prefixed by a digit carries a one-character symbol
    if not name[:1].isalpha():
        return letters[0] if letters[0] in chemical_symbols else "X"

    symbol = letters[:2].capitalize()
    if symbol in _HYDROGEN_LOOKALIKES:
        return "H"
    if symbol in chemical_symbols:
        return symbol
    # Two-letter guess is not an element, so treat it as a single-letter symbol
    return symbol[0] if symbol[0] in chemical_symbols else "X"


def write_xyz_frame(fh, symbols, positions, comment=""):
    """
    Write a single frame to an open XYZ file handle.

    Parameters
    ----------
    fh : file-like object
        An open, writable text handle.
    symbols : sequence of str
        Element symbol for each atom.
    positions : sequence of sequence of float
        Cartesian coordinates in angstrom, one ``(x, y, z)`` triple per atom.
    comment : str, optional
        Text for the frame's comment line. Default is an empty string.
    """
    fh.write(f"{len(symbols)}\n")
    fh.write(f"{comment}\n")
    for symbol, (x, y, z) in zip(symbols, positions):
        fh.write(f"{symbol:<2}   {x:>12.6f} {y:>12.6f} {z:>12.6f}\n")


def format_pdb_atom_name(symbol, count):
    """
    Format a unique atom name for the PDB atom-name field (columns 13-16).

    Single-character elements are indented by one column, following the PDB
    convention, and names longer than four characters are truncated so that
    column alignment is preserved.

    Parameters
    ----------
    symbol : str
        The element symbol.
    count : int
        Occurrence index of this element within its residue.

    Returns
    -------
    str
        A four-character atom name.
    """
    name = f"{symbol}{count}"
    if len(symbol) == 1 and len(name) < 4:
        return f" {name:<3}"
    return f"{name:<4}"[:4]


def convert_xyz_to_pdb(input_file, output_file, cutoff_multiplier=1.1, index=-1):
    """
    Convert an XYZ file to a PDB file with connectivity and residue assignment.

    Molecules (clusters) are identified using distance-based connectivity and
    assigned unique chain IDs, residue IDs, and three-letter residue names.
    Atoms within each cluster are reordered following Hill-system convention
    (C, H, then remaining elements alphabetically). CONECT records are written
    for all bonds.

    Parameters
    ----------
    input_file : str
        Path to the input XYZ file, or any other structure file ASE can read.
    output_file : str
        Path to the output PDB file.
    cutoff_multiplier : float, optional
        Multiplier applied to natural covalent-radius cutoffs when
        determining bonded neighbours. Default is 1.1.
    index : int, optional
        Which frame to convert when the input holds a trajectory. Default is
        -1 (the last frame). Only a single frame is written.

    Returns
    -------
    int
        The number of molecular clusters (connected components) found.

    See Also
    --------
    convert_pdb_to_xyz : The inverse conversion.
    """
    original_atoms = read(input_file, index=index)
    n_atoms = len(original_atoms)

    cutoffs = [c * cutoff_multiplier for c in natural_cutoffs(original_atoms)]
    i, j = neighbor_list("ij", original_atoms, cutoffs)

    adjacency_matrix = csr_matrix((np.ones_like(i), (i, j)), shape=(n_atoms, n_atoms))
    n_clusters, labels = connected_components(csgraph=adjacency_matrix, directed=False)

    # Group atoms by cluster, then order each cluster by the Hill system (C, H, others).
    hill_priority = {"C": 0, "H": 1}
    symbols = original_atoms.get_chemical_symbols()
    order = sorted(
        range(n_atoms),
        key=lambda idx: (labels[idx], hill_priority.get(symbols[idx], 2), symbols[idx]),
    )

    atoms = original_atoms[order]
    sorted_labels = [labels[idx] for idx in order]

    # Re-index the connectivity found above onto the new ordering, rather than
    # paying for a second neighbour-list pass over the sorted atoms.
    old_to_new = np.empty(n_atoms, dtype=int)
    old_to_new[order] = np.arange(n_atoms)
    i, j = old_to_new[i], old_to_new[j]

    unique_labels = list(dict.fromkeys(sorted_labels))
    available_chains = string.ascii_uppercase + string.ascii_lowercase + string.digits
    num_chains = len(available_chains)
    cluster_ids = {}

    for cluster_idx, lbl in enumerate(unique_labels):
        # Chain ID wraps around every 62 clusters, and the residue ID
        # increments only when the chain ID wraps
        chain = available_chains[cluster_idx % num_chains]
        resid = (cluster_idx // num_chains) + 1

        # Unique three-letter residue name (AAA, AAB, ..., AAZ, ABA, ..., ZZZ)
        resname = "".join(chr(65 + (cluster_idx // 26**p) % 26) for p in (2, 1, 0))

        cluster_ids[lbl] = (chain, resid, resname)

    element_counts_per_cluster = defaultdict(int)

    with open(output_file, "w") as f:
        for idx, atom in enumerate(atoms):
            chain, resid, resname = cluster_ids[sorted_labels[idx]]
            sym = atom.symbol
            x, y, z = atom.position

            # Atom names are made unique within each chain/residue combination
            element_counts_per_cluster[(chain, resid, sym)] += 1
            name = format_pdb_atom_name(
                sym, element_counts_per_cluster[(chain, resid, sym)]
            )

            f.write(
                f"ATOM  {idx + 1:>5} {name} {resname:>3} {chain}{resid:>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {sym:>2}\n"
            )

        conect_dict = defaultdict(list)
        for a1, a2 in zip(i, j):
            conect_dict[a1].append(a2)

        for atom_idx in sorted(conect_dict):
            bonded_atoms = sorted(b + 1 for b in conect_dict[atom_idx])

            # CONECT records hold at most four bonded partners each
            for chunk_start in range(0, len(bonded_atoms), 4):
                chunk = "".join(
                    f"{b:>5}" for b in bonded_atoms[chunk_start : chunk_start + 4]
                )
                f.write(f"CONECT{atom_idx + 1:>5}{chunk}\n")

        f.write("END\n")

    return n_clusters


def convert_pdb_to_xyz(input_file, output_file, comment=None):
    """
    Convert a PDB file to an XYZ file.

    Every ``ATOM`` and ``HETATM`` record contributes one atom, in file order.
    Multi-model PDB files (``MODEL``/``ENDMDL``) produce one XYZ frame per
    model; files without model records produce a single frame. Coordinates are
    passed through unchanged, both formats using angstrom.

    Parameters
    ----------
    input_file : str
        Path to the input PDB file.
    output_file : str
        Path to the output XYZ file.
    comment : str, optional
        Text for the comment line of every frame. If None, a comment naming
        the source file (and frame number, when there is more than one) is
        generated. Default is None.

    Returns
    -------
    int
        The number of frames written.

    Raises
    ------
    ValueError
        If the input file contains no ``ATOM`` or ``HETATM`` records.

    See Also
    --------
    convert_xyz_to_pdb : The inverse conversion.
    """
    frames = []
    symbols = []
    positions = []

    with open(input_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                symbols.append(element_from_pdb_line(line))
                positions.append(
                    (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                )
            elif line.startswith("ENDMDL") and symbols:
                frames.append((symbols, positions))
                symbols, positions = [], []

    # Catch the trailing model, and the single-frame case with no model records
    if symbols:
        frames.append((symbols, positions))

    if not frames:
        raise ValueError(f"No ATOM or HETATM records found in {input_file!r}.")

    source = os.path.basename(input_file)
    with open(output_file, "w") as f:
        for frame_idx, (frame_symbols, frame_positions) in enumerate(frames, start=1):
            if comment is not None:
                text = comment
            elif len(frames) > 1:
                text = f"Frame: {frame_idx} of {len(frames)}, Source: {source}"
            else:
                text = f"Source: {source}"
            write_xyz_frame(f, frame_symbols, frame_positions, text)

    print(f"Wrote {len(frames)} frame(s) to {output_file}", flush=True)
    return len(frames)


def convert_xyz_to_plumed_ref(xyz_file, template_pdb, output_file, atom_line="HETATM"):
    """
    Convert a reaction path from XYZ into the reference PLUMED's PATHMSD reads.

    Each XYZ frame becomes one model of a multi-model PDB, written by taking
    the template's atom records and substituting the frame's coordinates into
    the columns that hold them.  Going through a template rather than writing
    fresh records is what keeps the atom names, residues and numbering
    identical to the structure the simulation will be aligned against; PLUMED
    matches the two by atom serial, and any disagreement is fatal.  Both files
    are renumbered on the way out to guarantee that.

    Parameters
    ----------
    xyz_file : str
        Path to the input XYZ file, holding one frame per path image.
    template_pdb : str
        Path to the template PDB, holding only the atoms the collective
        variable is built from. Renumbered in place.
    output_file : str
        Path to write the multi-model PDB to.
    atom_line : str or tuple of str, optional
        Record type the template's atoms are written under; only these lines
        are carried into the output. Default is ``'HETATM'``, which is what
        OpenMM writes for ligand-like residues. ASE writes ``'ATOM'``.

    See Also
    --------
    reactiontools.tools_path.path_from_steered_md : Builds the path from steered MD.
    estimate_path_lambda : Sizes the LAMBDA the reference should be given.
    """
    atom_records = (atom_line,) if isinstance(atom_line, str) else tuple(atom_line)
    with open(template_pdb, "r") as f:
        template_lines = [line for line in f if line.startswith((*atom_records, "TER"))]

    with open(xyz_file, "r") as f:
        lines = f.readlines()

    if not lines:
        return

    num_atoms = int(lines[0].strip())
    frames = []
    for i in range(0, len(lines), num_atoms + 2):
        frame_coords = lines[i + 2 : i + num_atoms + 2]
        frames.append([line.split()[1:] for line in frame_coords])

    with open(output_file, "w") as f:
        f.write("REMARK TYPE=MULTI-ST-PDB\n")
        f.write("REMARK ARG=path.s,path.z\n")

        for i, frame in enumerate(frames):
            f.write(f"REMARK NUMBER={i + 1}\n")
            f.write(f"REMARK STEP={i}\n")

            # Tracks position within `frame` separately from `template_lines`,
            # since TER lines consume the latter but not the former.
            coord_idx = 0

            for t_line in template_lines:
                if t_line.startswith("TER"):
                    f.write(t_line)
                else:
                    coords = frame[coord_idx]
                    new_line = (
                        t_line[:30]
                        + f"{float(coords[0]):8.3f}{float(coords[1]):8.3f}{float(coords[2]):8.3f}"
                        + t_line[54:]
                    )
                    f.write(new_line)
                    coord_idx += 1

            f.write("ENDMDL\n")
    pdb_remove_ter_index(template_pdb, template_pdb)
    pdb_remove_ter_index(output_file, output_file)


def pdb_remove_ter_index(input_path, output_path):
    """
    Renumber the atom serials of a PDB file, keeping TER and CONECT in step.

    Atoms are renumbered sequentially from 1, restarting at each model, and
    ``CONECT`` records are rewritten to point at the new serials.  PLUMED
    insists that a reference and its template agree on numbering, which an
    edited PDB usually does not, so both are put through this before being
    handed over.  Records other than these are copied out unchanged.

    Parameters
    ----------
    input_path : str
        Path to the input PDB file.
    output_path : str
        Path to write the renumbered PDB to. May be the input path.
    """
    with open(input_path, "r") as f:
        lines = f.readlines()

    clean_lines = []
    atom_serial = 1
    # Old serial (as written) -> new right-aligned 5-character serial field
    index_map = {}

    for line in lines:
        if line.startswith(("MODEL", "REMARK NUMBER=")):
            atom_serial = 1
            clean_lines.append(line)

        elif line.startswith(("ATOM  ", "HETATM")):
            old_serial = line[6:11].strip()
            index_map.setdefault(old_serial, f"{atom_serial:>5}")
            new_line = line[:6] + f"{atom_serial:5d}" + line[11:]
            clean_lines.append(new_line)
            atom_serial += 1

        elif line.startswith("TER"):
            # A TER shares the serial of the atom that follows it, so this
            # deliberately does not advance the counter.
            if len(line) >= 11:
                new_line = line[:6] + f"{atom_serial:5d}" + line[11:]
            else:
                new_line = f"{line.strip():<6}{atom_serial:5d}\n"
            clean_lines.append(new_line)

        elif line.startswith("CONECT"):
            new_conect = line[:6]
            for i in range(6, len(line.strip()), 5):
                old_idx = line[i : i + 5].strip()
                new_conect += index_map.get(old_idx, line[i : i + 5])
            clean_lines.append(new_conect.rstrip() + "\n")

        else:
            clean_lines.append(line)

    with open(output_path, "w") as f:
        f.writelines(clean_lines)


def strip_hydrogens_keep_indices(input_pdb, output_pdb, keep=None):
    """
    Remove hydrogen atoms from a PDB file, except for a chosen subset.

    A path collective variable built over every atom of a reacting group
    spends most of its resolution on hydrogens that only rattle. Stripping all
    but the ones actually being transferred leaves ``PATHMSD`` measuring the
    reaction rather than the thermal noise around it.

    Parameters
    ----------
    input_pdb : str
        Path to the input PDB file.
    output_pdb : str
        Path to write the filtered PDB file. Must not be *input_pdb*: the two
        files are open at once, so writing in place would truncate the input
        before it had been read.
    keep : iterable of int, optional
        0-based atom indices (matching PDB serial number minus one) of
        hydrogens to retain even though they would otherwise be stripped.
        If None, all hydrogens are removed. Default is None.
    """
    keep = set() if keep is None else {i + 1 for i in keep}

    with open(input_pdb, "r") as fin, open(output_pdb, "w") as fout:
        for line in fin:
            if not line.startswith(("ATOM", "HETATM")):
                fout.write(line)
                continue

            # Short-circuits, so a record that is not a hydrogen never has its
            # serial parsed and a malformed one cannot raise here.
            if element_from_pdb_line(line) != "H" or int(line[6:11].strip()) in keep:
                fout.write(line)

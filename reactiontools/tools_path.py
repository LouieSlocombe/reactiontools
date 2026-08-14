"""Reference paths for path collective variables, estimated from steered MD.

A ``PATHMSD`` simulation needs a reference path: an ordered set of frames
walking the system from reactant to product. :mod:`reactiontools.tools_reaction`
builds that path with a nudged elastic band, which is accurate but needs both
endpoints optimised up front. The functions here take the cheaper route of
dragging a collective variable across with a moving harmonic restraint (see
:func:`~reactiontools.tools_cv.plumed_input_steered`) and picking frames out of
the trajectory that leaves behind. The result is a path in the full simulation
environment -- solvent included -- rather than one interpolated in vacuum.

The typical sequence is

1. :func:`~reactiontools.tools_cv.plumed_input_steered_pt` writes the pulling
   script,
2. an MD engine runs it -- ``openmmnqe.run_openmm_steered`` for an OpenMM run,
3. :func:`path_from_steered_md` selects the frames and writes ``neb_path.pdb``,
4. :func:`~reactiontools.tools_cv.plumed_input_neb_path` biases along it, with
   the ``LAMBDA`` :func:`estimate_path_lambda` recommends.

Note that the path atoms should belong to a single molecule. A trajectory is
written with molecules wrapped into the periodic box, so a path spanning two
molecules -- the two bases of a pair, say -- can have them wrapped to opposite
sides of the box, which no amount of alignment repairs.

Trajectory readers use MDTraj, which is installed with the package and imported
only when needed. The frame-selection functions (:func:`select_frames_by_cv`,
:func:`select_frames_by_msd`, :func:`cv_from_colvar`) remain NumPy-only.
"""

import os

import numpy as np

from .tools_fes import read_plumed_file
from .tools_io import convert_xyz_to_plumed_ref, write_xyz_frame

__all__ = [
    "cv_from_colvar",
    "estimate_path_lambda",
    "path_from_steered_md",
    "select_frames_by_cv",
    "select_frames_by_msd",
]

_MDTRAJ_HINT = (
    "{name} needs mdtraj to read the trajectory, which is not "
    "installed. Reinstall reactiontools with its dependencies."
)

#: How many of mdtraj's nanometres go into one unit of each supported length.
#: mdtraj always hands back nanometres, whatever the file said, so this is the
#: only place the choice has to be honoured.
_LENGTH_PER_NM = {"nm": 1.0, "a": 10.0}


def _import_mdtraj(name):
    """Import mdtraj, or explain how to get it.

    Imported here rather than at module scope so that the frame-selection
    functions, which are numpy only, stay usable without it.

    Parameters
    ----------
    name : str
        Name of the calling function, for the error message.

    Returns
    -------
    module
        The mdtraj module.

    Raises
    ------
    ImportError
        If mdtraj is not installed.
    """
    try:
        import mdtraj
    except ImportError as exc:
        raise ImportError(_MDTRAJ_HINT.format(name=name)) from exc
    return mdtraj


def _length_scale(length_unit):
    """Nanometres per unit of *length_unit*, validated.

    Parameters
    ----------
    length_unit : str
        ``'nm'`` or ``'A'``, case-insensitively.

    Returns
    -------
    float
        How many of that unit make up one nanometre.

    Raises
    ------
    ValueError
        If the unit is not one of the two.
    """
    key = str(length_unit).strip().lower()
    if key not in _LENGTH_PER_NM:
        raise ValueError(
            f"Unknown length unit {length_unit!r}. PLUMED works in nanometres "
            f"when driven from OpenMM and angstrom when driven from ASE, so "
            f"this is 'nm' or 'A'."
        )
    return _LENGTH_PER_NM[key]


def estimate_path_lambda(pdb_path, length_unit="nm"):
    """
    Estimate the LAMBDA a reference path should be given in ``PATHMSD``.

    ``LAMBDA`` sets how sharply ``path.sss`` tells neighbouring frames apart.
    Too small and the whole path reads as one blurred position; too large and
    the CV jumps between frames instead of moving smoothly along them.  The
    recommendation is the usual rule of thumb, ``2.3 / avg_msd``, which puts
    the crossover between consecutive frames at roughly one unit of the
    exponent.  A summary of the path and any warnings are printed as it goes.

    Parameters
    ----------
    pdb_path : str
        Multi-frame PDB holding the path, as written by
        :func:`~reactiontools.tools_io.convert_xyz_to_plumed_ref` or
        :func:`path_from_steered_md`.
    length_unit : str, optional
        Length unit the PLUMED script this LAMBDA is going into works in:
        ``'nm'`` (the default, and what PLUMED uses when driven from OpenMM)
        or ``'A'`` (what it uses when driven from ASE, via
        :data:`~reactiontools.tools_plumed.PLUMED_ASE_UNITS`). LAMBDA has
        units of inverse squared length, so the two answers differ by a factor
        of a hundred and using the wrong one is not a small error.

    Returns
    -------
    float
        Recommended ``LAMBDA``, to pass to the ``plumed_input_neb_path*``
        builders in :mod:`reactiontools.tools_cv`.

    Raises
    ------
    ImportError
        If mdtraj is not installed.
    ValueError
        If *length_unit* is not ``'nm'`` or ``'A'``, or the path holds fewer
        than two frames.

    Notes
    -----
    A path of 15 to 30 frames behaves best. Fewer and consecutive frames are
    too far apart for the CV to interpolate between; more and ``PATHMSD``
    spends its time on frames that add nothing.
    """
    md = _import_mdtraj("estimate_path_lambda")
    per_nm = _length_scale(length_unit)

    traj = md.load(pdb_path)
    if traj.n_frames < 2:
        raise ValueError(
            f"{pdb_path} holds {traj.n_frames} frame(s); LAMBDA is set from "
            f"the spacing between frames, so there must be at least two."
        )
    traj.superpose(traj[0])

    # mdtraj works in nanometres whatever the file said, so scale the
    # coordinates before squaring rather than the MSD after.
    xyz = traj.xyz * per_nm
    msds = [
        np.mean(np.sum((xyz[i] - xyz[i + 1]) ** 2, axis=1))
        for i in range(len(traj) - 1)
    ]
    avg_msd = np.mean(msds)
    max_msd = np.max(msds)
    ideal_lambda = 2.3 / avg_msd

    unit = "nm" if per_nm == 1.0 else "A"
    print("--- Path Analysis ---", flush=True)
    print(f"Number of frames: {len(traj)} (aim for 15 to 30)", flush=True)
    print(f"Average MSD between frames: {avg_msd:.6f} {unit}^2", flush=True)
    print(f"Maximum MSD between frames: {max_msd:.6f} {unit}^2", flush=True)
    print(f"Recommended LAMBDA for PLUMED: {ideal_lambda:.2f} {unit}^-2", flush=True)

    if max_msd > 2 * avg_msd:
        print("WARNING: Your path frames are unevenly spaced.", flush=True)
        print("Consider interpolating your path for better stability.", flush=True)

    if ideal_lambda > 500.0:
        print("WARNING: The recommended LAMBDA is very high", flush=True)

    return ideal_lambda


def cv_from_colvar(colvar_file, n_frames, cv_name=None):
    """
    Read a CV from a PLUMED COLVAR file, one value per trajectory frame.

    PLUMED and OpenMM disagree about when to write: ``PRINT`` fires at step 0
    and every stride thereafter, while a reporter first fires one interval in.
    So a COLVAR written with the same stride as the reporter holds exactly one
    row more than the trajectory has frames, and its first row is dropped
    here.  Any other length is resampled onto the frames by assuming both
    cover the same span of time at a constant rate.

    Parameters
    ----------
    colvar_file : str
        Path to the COLVAR file written by the steered run.
    n_frames : int
        Number of frames in the trajectory the values are wanted for.
    cv_name : str or None, optional
        Field name of the CV, e.g. ``'pt_cv'``. If None, the first column
        after ``time`` is used. Default is None.

    Returns
    -------
    numpy.ndarray
        CV value for each trajectory frame, of length *n_frames*.
    """
    colvar = read_plumed_file(colvar_file)
    cv = colvar.column(cv_name if cv_name is not None else 1)

    if cv.size == n_frames:
        return cv
    if cv.size == n_frames + 1:
        # The extra row is the one PLUMED wrote at step 0, before the
        # trajectory had a frame to go with it.
        return cv[1:]

    print(f"COLVAR has {cv.size} rows for {n_frames} frames, resampling.", flush=True)
    # Frame i sits at (i + 1) / n_frames of the way through the run.
    frame_fraction = (np.arange(n_frames) + 1.0) / n_frames
    colvar_fraction = np.arange(cv.size) / (cv.size - 1.0)
    return np.interp(frame_fraction, colvar_fraction, cv)


def _nearest_monotone(series, targets):
    """
    Pick the entry of *series* closest to each target, never going backwards.

    Searching the whole series for each target would let a noisy trajectory
    hand back frames out of order, which is no use as a path.  Each search
    therefore starts after the previously chosen entry, and stops early enough
    to leave one entry for every target still to come.

    Parameters
    ----------
    series : array_like
        Values to search, one per frame.
    targets : array_like
        Values to match, in the order they should appear along the path.

    Returns
    -------
    numpy.ndarray
        Strictly increasing indices into *series*, one per target.

    Raises
    ------
    ValueError
        If *series* is shorter than *targets*.
    """
    series = np.asarray(series, dtype=float)
    n_targets = len(targets)
    if series.size < n_targets:
        raise ValueError(
            f"Cannot pick {n_targets} frames from a series of {series.size}."
        )

    picks = []
    low = 0
    for i, target in enumerate(targets):
        # Leave one frame behind for each target that has not been placed yet
        high = series.size - (n_targets - i - 1)
        window = series[low:high]
        picks.append(low + int(np.argmin(np.abs(window - target))))
        low = picks[-1] + 1
    return np.asarray(picks)


def select_frames_by_cv(cv, n_images, cv_start=None, cv_stop=None):
    """
    Choose the frames that are evenly spaced along a collective variable.

    Parameters
    ----------
    cv : array_like
        CV value for each frame of the trajectory.
    n_images : int
        Number of frames to select.
    cv_start, cv_stop : float or None, optional
        Ends of the CV range to cover. If None, the first and last values in
        *cv* are used. Default is None.

    Returns
    -------
    numpy.ndarray
        Indices of the selected frames, in path order.
    """
    cv = np.asarray(cv, dtype=float)
    start = cv[0] if cv_start is None else cv_start
    stop = cv[-1] if cv_stop is None else cv_stop
    return _nearest_monotone(cv, np.linspace(start, stop, n_images))


def select_frames_by_msd(xyz, n_images):
    """
    Choose the frames that are evenly spaced along the trajectory itself.

    Distance is measured as the RMSD between consecutive frames accumulated
    along the trajectory, which is the spacing ``PATHMSD`` cares about.  Use
    this when there is no COLVAR to select on, and bear in mind that thermal
    jitter inflates the arc length of a noisy trajectory.

    Parameters
    ----------
    xyz : numpy.ndarray
        Coordinates with shape ``(n_frames, n_atoms, 3)``, already aligned.
    n_images : int
        Number of frames to select.

    Returns
    -------
    numpy.ndarray
        Indices of the selected frames, in path order.
    """
    step = np.sqrt(np.mean(np.sum(np.diff(xyz, axis=0) ** 2, axis=-1), axis=-1))
    arc = np.concatenate(([0.0], np.cumsum(step)))
    return _nearest_monotone(arc, np.linspace(0.0, arc[-1], n_images))


def _smooth_frames(xyz, picks, window):
    """
    Average each selected frame with its neighbours to damp thermal noise.

    Parameters
    ----------
    xyz : numpy.ndarray
        Aligned coordinates with shape ``(n_frames, n_atoms, 3)``.
    picks : array_like of int
        Indices of the selected frames.
    window : int
        Number of frames either side to average over. Zero returns the
        selected frames untouched.

    Returns
    -------
    numpy.ndarray
        Coordinates of the path, with shape ``(len(picks), n_atoms, 3)``.
    """
    if window <= 0:
        return xyz[picks]

    smoothed = np.empty((len(picks),) + xyz.shape[1:])
    for i, frame in enumerate(picks):
        low = max(0, frame - window)
        high = min(xyz.shape[0], frame + window + 1)
        smoothed[i] = xyz[low:high].mean(axis=0)
    return smoothed


def path_from_steered_md(
    traj_file,
    template_pdb="index_atoms.pdb",
    output_file="neb_path.pdb",
    colvar_file="COLVAR_SMD",
    cv_name=None,
    n_images=15,
    atom_indices=None,
    top=None,
    cv_start=None,
    cv_stop=None,
    smooth=0,
    align=True,
    atom_line="HETATM",
    length_unit="nm",
):
    """
    Estimate a path collective variable from a steered MD trajectory.

    Frames evenly spaced along the CV are pulled out of the trajectory,
    aligned, and written as the multi-model PDB that ``PATHMSD`` reads, in the
    same format :func:`~reactiontools.tools_io.convert_xyz_to_plumed_ref`
    produces for a NEB path.  An XYZ copy is written alongside it for viewing.

    Parameters
    ----------
    traj_file : str
        Trajectory from the steered run, e.g. the ``smd_steps.pdb`` written by
        ``openmmnqe.run_openmm_steered``.
    template_pdb : str, optional
        PDB holding the path atoms, as written by
        ``openmmnqe.save_only_index_atoms``. Its atom records are the template
        for the output. Default is ``'index_atoms.pdb'``. Note that this file
        is renumbered in place so that it and the path agree on atom
        numbering, which is what PLUMED expects of them.
    output_file : str, optional
        Multi-model PDB to write the path to. Default is ``'neb_path.pdb'``,
        which is what the ``PATHMSD`` inputs in
        :mod:`reactiontools.tools_cv` reference.
    colvar_file : str or None, optional
        COLVAR file from the steered run. If None, frames are spaced by RMSD
        along the trajectory instead of by CV. Default is ``'COLVAR_SMD'``.
    cv_name : str or None, optional
        Field name of the CV in *colvar_file*. If None, the first column after
        ``time`` is used. Default is None.
    n_images : int, optional
        Number of frames in the path. Default is 15; ``PATHMSD`` behaves best
        with 15 to 30.
    atom_indices : list of int or None, optional
        Atoms of the full system that make up the path, normally the same
        indices the template was written from. If None, every atom in the
        trajectory is used. Default is None.
    top : str or None, optional
        Topology file for trajectory formats that carry none, such as DCD.
        Default is None.
    cv_start, cv_stop : float or None, optional
        Ends of the CV range the path should span. If None, the first and last
        values in the COLVAR are used, i.e. the whole pull. Default is None.
    smooth : int, optional
        Number of neighbouring frames either side to average each path frame
        with. Default is 0, which keeps the frames as they were sampled. Try 2
        or 3 if the path comes out jagged.
    align : bool, optional
        Whether to superpose every frame on the first before selecting.
        Default is True.
    atom_line : str or tuple of str, optional
        Record type the template's atoms are written under. Default is
        ``'HETATM'``, which is what OpenMM writes for the ligand-like
        residues these paths normally cover. ASE writes ``'ATOM'``.
    length_unit : str, optional
        Length unit of the PLUMED run this path is for, passed on to
        :func:`estimate_path_lambda` to scale the recommended ``LAMBDA``.
        Default is ``'nm'``. The written files are always in angstrom, which
        is what PDB and XYZ use; this affects only the returned number.

    Returns
    -------
    float
        The LAMBDA value recommended for this path, as reported by
        :func:`estimate_path_lambda`.

    Raises
    ------
    ImportError
        If mdtraj is not installed.
    ValueError
        If the path atoms and the template PDB do not match, or the trajectory
        holds fewer frames than the path needs.
    """
    md = _import_mdtraj("path_from_steered_md")

    if atom_indices is not None:
        # Sorting keeps the frames in topology order, which is the order the
        # template was written in.
        atom_indices = np.asarray(sorted(atom_indices), dtype=int)

    # Only formats that carry no topology of their own accept `top`
    load_kwargs = {"top": top} if top is not None else {}
    traj = md.load(traj_file, atom_indices=atom_indices, **load_kwargs)
    n_template = md.load(template_pdb).n_atoms
    if traj.n_atoms != n_template:
        raise ValueError(
            f"Path has {traj.n_atoms} atoms but {template_pdb} has {n_template}. "
            "Pass the atom indices the template was written from."
        )
    with open(template_pdb, "r") as handle:
        n_records = sum(1 for line in handle if line.startswith(atom_line))
    if n_records != n_template:
        # Only lines of this record type make it into the output, so a
        # mismatch here would quietly write a path with atoms missing.
        raise ValueError(
            f"{template_pdb} holds {n_template} atoms but {n_records} {atom_line} "
            "records. Set atom_line to the record type it uses."
        )
    if traj.n_frames < n_images:
        raise ValueError(
            f"Trajectory has {traj.n_frames} frames, too few for {n_images} images. "
            "Report the steered run more often, or ask for fewer images."
        )

    if align:
        traj.superpose(traj, 0)

    if colvar_file is not None:
        cv = cv_from_colvar(colvar_file, traj.n_frames, cv_name=cv_name)
        picks = select_frames_by_cv(cv, n_images, cv_start=cv_start, cv_stop=cv_stop)
        print(f"Path spans CV {cv[picks[0]]:.3f} to {cv[picks[-1]]:.3f}", flush=True)
    else:
        picks = select_frames_by_msd(traj.xyz, n_images)

    print(f"Selected frames {picks.tolist()} of {traj.n_frames}", flush=True)
    positions = _smooth_frames(traj.xyz, picks, smooth) * 10.0  # nm to angstrom

    symbols = [
        atom.element.symbol if atom.element is not None else atom.name[:2]
        for atom in traj.topology.atoms
    ]
    xyz_file = f"{os.path.splitext(output_file)[0]}.xyz"
    with open(xyz_file, "w") as handle:
        for i, frame in enumerate(positions):
            write_xyz_frame(
                handle, symbols, frame, comment=f"steered MD path image {i + 1}"
            )

    convert_xyz_to_plumed_ref(xyz_file, template_pdb, output_file, atom_line=atom_line)
    print(f"Wrote {n_images} path images to {output_file}", flush=True)
    return estimate_path_lambda(output_file, length_unit=length_unit)

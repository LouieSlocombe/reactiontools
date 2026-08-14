# Installation guide

This guide will help you install and set up the project on your local machine.

## 1. Create the conda environment

```bash
conda env create -f environment.yml -y
```

```bash
conda activate reactiontools
```

## 2. Install reactiontools

From the repository root, install the package in editable mode so changes to the
source are picked up without reinstalling:

```bash
pip install -e .
```

This installs the complete runtime and development toolchain, including Sella,
MDTraj, pytest and Ruff.

## 3. Check the install

```bash
pytest
```

Check formatting and lint before committing:

```bash
ruff format --check .
ruff check .
```

# Install PLUMED and OPES module

`run_sum_hills` calls the `plumed` executable, so it needs to be on your `PATH`.
The conda environment provides one; build from source if you need the OPES
module or a specific version.

```bash
wget https://github.com/plumed/plumed2/releases/download/v2.10.1/plumed-2.10.1.tgz
mkdir plumed && tar xzf plumed-*.tgz -C plumed --strip-components=1 && cd plumed
./configure --enable-modules=opes
make -j$(nproc) && make install
```

Update your .bashrc file to include the following lines:

```bash
export PLUMED_KERNEL="$HOME/plumed/src/lib/libplumedKernel.so"
```

## ORCA

The helpers in `reactiontools.tools_orca` shell out to ORCA, which is licensed
separately and must be installed by hand:

1. Download it from the [ORCA website](https://www.faccts.de/orca/).
2. Extract it: `tar -xf orca-x.y.z.tar.gz`
3. Point `ORCA_PATH` at the `orca` binary, adding this to your `~/.bashrc`:

   ```bash
   export ORCA_PATH="/path/to/orca_6_1_1/orca"
   ```

`orca_calc_preset()`, `orca_optimise_atoms()` and `orca_calculate_goat()` read
`ORCA_PATH` when no explicit path is passed, as do the ORCA tests — which skip
rather than fail when it is unset.

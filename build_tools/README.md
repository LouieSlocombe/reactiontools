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
pip install -e ".[dev]"
```

The `environment.yml` already pulls in `geodesic_interpolate` from git. If you
installed the dependencies some other way, add it with:

```bash
pip install -e ".[geodesic]"
```

## 3. Check the install

```bash
pytest
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

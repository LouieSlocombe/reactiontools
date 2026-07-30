# Installation guide

This guide will help you install and set up the project on your local machine.

```bash
conda env create -f environment.yml -y
```

```bash
conda activate reactiontools
```

# Install PLUMED and OPES module

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
#!/bin/bash
# Shared PLUMED build steps and installation checks, sourced by both installers.
# Both installers need an identical PLUMED, so the version is pinned here in one place.

PLUMED_VERSION="v2.10.1"

# build_plumed <work_dir>
# Compiles PLUMED (with the opes module) into $CONDA_PREFIX, cloning the sources into
# <work_dir>. Leaves the shell in <work_dir>.
build_plumed() {
    local work_dir="$1"

    echo "=== Compiling PLUMED ${PLUMED_VERSION} ==="
    cd "${work_dir}"
    git clone --branch "${PLUMED_VERSION}" --depth 1 --filter=blob:none https://github.com/plumed/plumed2.git
    cd plumed2
    # libplumedKernel.so links conda's BLAS and libgomp from ${CONDA_PREFIX}/lib, but
    # ld does not search there when resolving that library's own dependencies at the
    # final `plumed` link (-rpath-link, the same fix conda-forge's recipe uses) or at
    # run time (-rpath, which also lets py-plumed load the kernel without
    # LD_LIBRARY_PATH). --disable-python stops the top-level make from also
    # building the Python interface in-tree: it would bake the source-tree kernel
    # path into build artifacts that build_py_plumed's pip install then reuses,
    # instead of the ${CONDA_PREFIX} kernel it bakes itself.
    ./configure --prefix="${CONDA_PREFIX}" --enable-modules=opes --disable-python \
        LDFLAGS="-L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib" \
        STATIC_LIBS="-Wl,-rpath-link,${CONDA_PREFIX}/lib"
    make -j"$(nproc)"
    make install

    cd "${work_dir}"
}

# build_py_plumed <work_dir>
# Builds the PLUMED Python bindings (the `plumed` module that plumed_calculator
# imports) from the plumed2 sources that build_plumed left in <work_dir>, against
# the PLUMED installed in $CONDA_PREFIX. The kernel path is baked in as the
# default, so `plumed.Plumed()` can load it without PLUMED_KERNEL being set.
# Requires Cython. Leaves the shell in <work_dir>.
build_py_plumed() {
    local work_dir="$1"

    echo "=== Building py-plumed ${PLUMED_VERSION} ==="
    cd "${work_dir}/plumed2/python"
    # Stages ./PLUMED_VERSION and include/Plumed.h for setup.py.
    make pip
    plumed_default_kernel="${CONDA_PREFIX}/lib/libplumedKernel.so" \
        pip install . --no-build-isolation

    cd "${work_dir}"
}

# verify_installation <python_executable>
# Checks the OPES module, Python kernel loading, and the installed package using
# the caller's active interpreter. The installers' set -e stops at a failed check.
verify_installation() {
    local python_executable="$1"

    echo "=== Verifying Installation ==="
    plumed --no-mpi config -q module opes
    echo "PLUMED opes module: OK"
    "${python_executable}" -c "import plumed; plumed.Plumed()"
    echo "py-plumed kernel load: OK"
    "${python_executable}" -c "import reactiontools"
    echo "reactiontools: OK"
}

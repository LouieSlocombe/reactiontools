#!/bin/bash
# One-command install of the reactiontools conda environment: creates the environment
# from environment.yml, compiles PLUMED (with the opes module) and the PLUMED Python
# bindings (py-plumed) into it, then installs reactiontools and its two git
# dependencies in editable mode and verifies the result.
#
#   bash conda_install.sh
#
# WARNING: the target environment (default: reactiontools) is REMOVED and recreated
# from scratch on every run, as are the sources cloned into build_tools/sources/.
# Set ENV_NAME to install into a differently named environment instead:
#
#   ENV_NAME=reactiontools2 bash conda_install.sh
#
# geodesic_interpolate and sella are cloned next to this repository and installed
# editable. Existing checkouts are used as they are, never wiped. Set SRC_DIR to
# keep them somewhere else:
#
#   SRC_DIR="${HOME}/src" bash conda_install.sh

# Exit immediately on error and fail pipelines cleanly, so a broken build does not
# fall through to the later steps and report success.
set -eo pipefail

ENV_NAME="${ENV_NAME:-reactiontools}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${SCRIPT_DIR}/sources"
# Alongside the repository, so the checkouts survive the wipe WORK_DIR gets.
SRC_DIR="${SRC_DIR:-$(dirname "${REPO_DIR}")}"

# Pulls in build_plumed() and build_py_plumed(), with the PLUMED version they pin.
source "${SCRIPT_DIR}/build_plumed.sh"
# Pulls in install_editable_repos() and check_editable_repos(), with the git
# dependencies they clone.
source "${SCRIPT_DIR}/editable_repos.sh"

echo "=== Initializing Conda Environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda refuses to remove the active environment, so drop back to base first
# (covers running this script from inside an activated ${ENV_NAME}).
conda activate base
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
# -n overrides the name pinned inside environment.yml, so ENV_NAME works.
conda env create -n "${ENV_NAME}" -f "${SCRIPT_DIR}/environment.yml"
conda activate "${ENV_NAME}"

echo "=== Preparing Build Directory ==="
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

build_plumed "${WORK_DIR}"
build_py_plumed "${WORK_DIR}"

echo "=== Installing reactiontools (editable) ==="
pip install -e "${REPO_DIR}"

# After reactiontools, which drags its own copies of these in from git.
install_editable_repos "${SRC_DIR}"

echo "=== Verifying Installation ==="
cd "${REPO_DIR}"
plumed --no-mpi config -q module opes
echo "PLUMED opes module: OK"
python -c "import plumed; plumed.Plumed()"
echo "py-plumed kernel load: OK"
check_editable_repos "${SRC_DIR}"
echo "editable dependencies: OK"
python -c "import reactiontools"
echo "reactiontools: OK"

echo "=== Build Complete! ==="
echo "Activate with: conda activate ${ENV_NAME}"

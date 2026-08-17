#!/bin/bash
# Shared handling of the dependencies that come from git rather than conda-forge,
# sourced by conda_install.sh and custom_install_sol.sh. Both are forks that get
# edited alongside reactiontools, so they are cloned once and installed in
# editable mode instead of being pulled fresh from GitHub on every install.

# name=url pairs. The name is both the directory the repo is cloned into and the
# module the install is checked against.
EDITABLE_REPOS=(
    "geodesic_interpolate=https://github.com/LouieSlocombe/geodesic_interpolate.git"
    "sella=https://github.com/LouieSlocombe/sella.git"
)

# clone_repo <url> <path>
# Clones <url> into <path> unless a checkout is already there, which is left
# exactly as it is -- these hold work in progress, so nothing here pulls,
# resets or removes them.
clone_repo() {
    local url="$1"
    local path="$2"

    if [ -d "${path}/.git" ]; then
        echo "=== Using existing checkout: ${path} ==="
    elif [ -e "${path}" ]; then
        echo "${path} exists but is not a git checkout; move it aside and re-run." >&2
        return 1
    else
        echo "=== Cloning $(basename "${path}") into ${path} ==="
        git clone "${url}" "${path}"
    fi
}

# install_editable_repos <src_dir>
# Clones each git dependency into <src_dir> and installs it editable. Run this
# *after* reactiontools itself is installed: pip re-clones anything declared as a
# `name @ git+...` dependency in pyproject.toml even when it is already present,
# so an editable install done earlier would be overwritten by that copy.
install_editable_repos() {
    local src_dir="$1"
    local entry name url

    mkdir -p "${src_dir}"
    for entry in "${EDITABLE_REPOS[@]}"; do
        name="${entry%%=*}"
        url="${entry#*=}"
        clone_repo "${url}" "${src_dir}/${name}"
        echo "=== Installing ${name} (editable) ==="
        pip install -e "${src_dir}/${name}"
    done
}

# check_editable_repos <src_dir>
# Fails if any of them import from site-packages rather than the checkout.
check_editable_repos() {
    local src_dir="$1"

    python -c "
import importlib, pathlib, sys

src = pathlib.Path('${src_dir}').resolve()
for name in '${EDITABLE_REPOS[*]%%=*}'.split():
    path = pathlib.Path(importlib.import_module(name).__file__).resolve()
    if src not in path.parents:
        sys.exit(f'{name} is not editable: imported from {path.parent}')
    print(f'{name}: {path.parent}')
"
}

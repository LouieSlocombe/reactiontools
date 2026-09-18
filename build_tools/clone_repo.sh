#!/bin/bash
# Shared git handling, sourced by custom_install_sol.sh.
#
# This file used to install the dependencies that came from git rather than
# conda-forge, cloned once and installed editable so they could be edited
# alongside reactiontools. There are none left: geodesic_interpolate and then
# sella were both brought into the package, as reactiontools/tools_geodesic.py
# and reactiontools/tools_sella.py, so cloning either would install a copy that
# nothing imports. To change that code, edit the module.
#
# What remains is the clone helper, which custom_install_sol.sh uses to fetch
# reactiontools itself.

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

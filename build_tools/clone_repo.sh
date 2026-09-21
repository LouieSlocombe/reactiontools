#!/bin/bash
# Clone helper used by custom_install_sol.sh to fetch reactiontools.

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

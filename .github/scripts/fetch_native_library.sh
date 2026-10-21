#!/usr/bin/env bash
# Replaces native/<target>/libapbase_native.a with the libapbase_native-<target>.a
# asset of a release of ap-base/apbase-fortran-kernels.
#
#   fetch_native_library.sh <kernels_version> <target>...
#
# Needs KERNELS_TOKEN (read access to that repository), curl and jq.
set -euo pipefail

version="$1"
shift
api=https://api.github.com/repos/ap-base/apbase-fortran-kernels
auth=(-H "Authorization: Bearer ${KERNELS_TOKEN}" -H "X-GitHub-Api-Version: 2022-11-28")

release="$(curl -fsSL "${auth[@]}" "${api}/releases/tags/${version}")"
for target in "$@"; do
  asset="libapbase_native-${target}.a"
  # -j: no trailing newline (jq on Windows would end the URL with \r).
  url="$(jq -j --arg name "${asset}" '.assets[] | select(.name == $name) | .url' <<<"${release}")"
  if [ -z "${url}" ]; then
    echo "::error::kernels release ${version} has no ${asset}" >&2
    exit 1
  fi
  mkdir -p "native/${target}"
  curl -fsSL "${auth[@]}" -H "Accept: application/octet-stream" \
    -o "native/${target}/libapbase_native.a" "${url}"
  echo "native/${target}/libapbase_native.a <- ${version}/${asset}"
done

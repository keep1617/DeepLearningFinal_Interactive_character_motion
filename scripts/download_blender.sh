#!/usr/bin/env bash
set -euo pipefail

BLENDER_VERSION="${BLENDER_VERSION:-5.0.1}"
BLENDER_MAJOR="${BLENDER_VERSION%%.*}"
INSTALL_ROOT="${BLENDER_INSTALL_ROOT:-$HOME/Downloads}"
ARCHIVE_NAME="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
TARGET_DIR="${INSTALL_ROOT}/blender-${BLENDER_VERSION}-linux-x64"
DOWNLOAD_URL="https://download.blender.org/release/Blender${BLENDER_MAJOR}.0/${ARCHIVE_NAME}"

mkdir -p "${INSTALL_ROOT}"

if [ -x "${TARGET_DIR}/blender" ]; then
  echo "Blender already installed:"
  echo "${TARGET_DIR}/blender"
  exit 0
fi

echo "Downloading Blender ${BLENDER_VERSION}..."
echo "URL: ${DOWNLOAD_URL}"
curl -L -o "/tmp/${ARCHIVE_NAME}" "${DOWNLOAD_URL}"

echo "Extracting to ${INSTALL_ROOT}..."
tar -xf "/tmp/${ARCHIVE_NAME}" -C "${INSTALL_ROOT}"
rm -f "/tmp/${ARCHIVE_NAME}"

echo "Blender installed:"
echo "${TARGET_DIR}/blender"

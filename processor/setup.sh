#!/usr/bin/env bash

set -eE
cd "$(dirname "$0")"

hrir_zip_url=ftp://ftp.ircam.fr/pub/IRCAM/equipes/salles/listen/archive/SUBJECTS/IRC_1013.zip
hrir_zip_fn=data/$(basename "$hrir_zip_url")
if [ ! -f "$hrir_zip_fn" ]; then
    echo "Downloading default HRIR data..."
    tmp=$(mktemp)
    curl "$hrir_zip_url" -o "$tmp" && mv "$tmp" "$hrir_zip_fn"
fi

if [ ! -x .venv/bin/pip ]; then
    echo "Setting up Python venv in $(pwd)/.venv"
    python3 -m venv .venv/
fi

source .venv/bin/activate
pip install -e .

echo
echo "=== Setup Complete ==="
echo "Run with:"
echo "    source .venv/bin/activate"
echo "    pulse3d-processor -h"

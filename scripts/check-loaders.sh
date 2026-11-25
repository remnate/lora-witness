#!/bin/sh
# Run in a clean Python 3.12 oracle environment with the oracle extra installed.
set -eu
repo=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
output=${1:-oracle-work}
if [ -e "$output" ]; then
    echo "Choose a new output directory for loader evidence." >&2
    exit 2
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
rev=88ff6b5391e89822d0463df8769591273d0d28b1
curl -fsSL --retry 3 "https://codeload.github.com/kohya-ss/sd-scripts/tar.gz/$rev" -o "$work/source.tar.gz"
printf '%s  %s\n' 944873ac4a8cb58b46d01bbe94bc120524249b4716e98cc8c867992a34007715 "$work/source.tar.gz" | sha256sum -c -
tar -xzf "$work/source.tar.gz" -C "$work"
curl -fsSL --retry 3 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/462165984030d82259a11f4367a4eed129e94a7b/unet/config.json' -o "$work/config.json"
HF_HUB_OFFLINE=1 PYTHONPATH="$work/sd-scripts-$rev" python "$repo/scripts/loader_check.py" "$work/config.json" "$repo/examples/synthetic" "$output"

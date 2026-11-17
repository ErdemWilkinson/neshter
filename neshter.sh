#!/bin/sh
# Neshter launcher for Linux and macOS: runs THIS checkout with the locked dependency set (uv.lock).
# The Windows counterpart is neshter.bat.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if ! command -v uv >/dev/null 2>&1; then
  for dir in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    if [ -x "$dir/uv" ]; then PATH="$dir:$PATH"; export PATH; fi
  done
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv bulunamadi / uv not found. Neshter needs uv (https://docs.astral.sh/uv/)."
  printf "Install it now with Astral's official script (uv 0.13.6)? [y/N] "
  read -r answer || answer=n
  case "$answer" in
    y | Y | e | E) ;;
    *) echo "Cancelled. Install uv yourself and run this again."; exit 1 ;;
  esac
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/0.13.6/install.sh | sh
  else
    wget -qO- https://astral.sh/uv/0.13.6/install.sh | sh
  fi
  PATH="$HOME/.local/bin:$PATH"
  export PATH
  command -v uv >/dev/null 2>&1 || { echo "uv could not be installed: https://docs.astral.sh/uv/"; exit 1; }
fi

# An NVIDIA card on Linux uses torch's CUDA build; NESHTER_TORCH=cpu turns it off.
# macOS and machines without NVIDIA use the CPU build.
groups=""
if [ "${NESHTER_TORCH:-}" != "cpu" ] && [ "$(uname -s)" = "Linux" ] \
  && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  groups="--no-group cpu --group gpu"
fi

# shellcheck disable=SC2086 # $groups is deliberately split into separate arguments
exec uv run --project "$here" --frozen $groups neshter "$@"

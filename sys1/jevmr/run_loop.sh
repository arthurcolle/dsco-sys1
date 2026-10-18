#!/bin/zsh
# Scheduled entry point (launchd). Loads TYPESAFE_API_KEY from the user's shell
# profile at run time so the key is never stored in the plist.
source "$HOME/.zshrc" >/dev/null 2>&1
cd "$(dirname "$0")/../.." || exit 1
mkdir -p runs/jev
exec .venv/bin/python -m sys1.jevmr.jevmr loop sys1/jevmr/jobs >> runs/jev/loop.log 2>&1

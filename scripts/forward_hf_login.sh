#!/usr/bin/env bash
set -euo pipefail

# Forward THIS machine's Hugging Face login to a remote training node, so pulls
# of private model / dataset repos work there without an interactive
# `huggingface-cli login` and without the download-locally-then-copy detour.
#
# Usage:
#   scripts/forward_hf_login.sh <ssh-target>
#     e.g. scripts/forward_hf_login.sh ubuntu@211.21.0.5
#          scripts/forward_hf_login.sh gpu
#
# The token is read from your local login (respecting HF_HOME) and streamed to
# the remote's token file over SSH stdin: it never appears in argv, `ps`, or
# shell history on either side. huggingface_hub reads that file in every process
# afterwards, so `retrain` authenticates on the node with nothing else to set.
#
# On an ephemeral / untrusted node, forward a READ-scoped token rather than your
# default (which may carry write access):
#   HF_TOKEN=hf_readonly scripts/forward_hf_login.sh <ssh-target>

target="${1:-}"
if [ -z "$target" ]; then
  echo "usage: $0 <ssh-target>   (e.g. ubuntu@1.2.3.4 or gpu)" >&2
  exit 2
fi

hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
token_file="$hf_home/token"

# Prefer an explicit env token, else the local login's token file.
token="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [ -z "$token" ] && [ -f "$token_file" ]; then
  token="$(cat "$token_file")"
fi
if [ -z "$token" ]; then
  echo "no local Hugging Face token found." >&2
  echo "log in here first:  hf auth login   (or: huggingface-cli login)" >&2
  echo "or export HF_TOKEN=hf_... before rerunning." >&2
  exit 1
fi

# Push it to the remote's token file over stdin (never on the command line).
printf '%s' "$token" | ssh "$target" '
  set -e
  remote_home="${HF_HOME:-$HOME/.cache/huggingface}"
  mkdir -p "$remote_home"
  umask 077
  cat > "$remote_home/token"
  chmod 600 "$remote_home/token"
  echo "[forward-hf-login] token installed at $remote_home/token"
'
echo "[forward-hf-login] forwarded local HF login to ${target}"
echo "[forward-hf-login] verify on the node with:  hf auth whoami"

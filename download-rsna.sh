#!/bin/zsh
set -euo pipefail

: "${KAGGLE_API_TOKEN:?Set KAGGLE_API_TOKEN before running this script}"
task_dir="$(cd "$(dirname "$0")" && pwd)"
target_dir="$task_dir/data/rsna-knee-abnormality-detection"
mkdir -p "$target_dir"
umask 077
auth_config="$(mktemp)"
trap 'rm -f "$auth_config"' EXIT
printf 'header = "Authorization: Bearer %s"\n' "$KAGGLE_API_TOKEN" > "$auth_config"

# Resume an interrupted transfer. Do not run while another download is active.
curl --config "$auth_config" --fail --location \
  --retry 10 --retry-delay 5 --connect-timeout 30 --continue-at - \
  --output "$target_dir/rsna-knee-abnormality-detection.zip" \
  'https://www.kaggle.com/api/v1/competitions/data/download-all/rsna-knee-abnormality-detection'
printf '\nDownload complete: %s\n' "$target_dir/rsna-knee-abnormality-detection.zip"

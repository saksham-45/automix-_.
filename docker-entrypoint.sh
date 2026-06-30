#!/usr/bin/env bash
# Container entrypoint: optionally materialize YouTube cookies from a secret, then
# serve with gunicorn (single worker so the in-process session/chunk cache is shared;
# threads handle the cheap status/chunk requests while renders run in background threads).
set -euo pipefail

# If a YT_COOKIES secret (Netscape cookies.txt contents) is provided, write it to a
# file and point yt-dlp at it. This is the standard mitigation for datacenter-IP
# bot-throttling by YouTube. Without it, only the bundled sample is guaranteed to work.
if [ -n "${YT_COOKIES:-}" ]; then
    printf '%s' "$YT_COOKIES" > /tmp/yt_cookies.txt
    export YT_COOKIES_FILE=/tmp/yt_cookies.txt
fi

exec gunicorn \
    --workers 1 \
    --threads "${GUNICORN_THREADS:-8}" \
    --bind "0.0.0.0:${PORT:-7860}" \
    --timeout "${GUNICORN_TIMEOUT:-120}" \
    club_server:app

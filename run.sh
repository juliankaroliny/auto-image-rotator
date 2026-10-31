#!/bin/sh
# Run autorotate on the photos/ folder next to this script.
# Extra arguments are passed through, e.g. ./run.sh --apply
# or ./run.sh --rotate ccw photos/img.jpg
set -e
cd "$(dirname "$0")"
IMAGE=auto-image-rotator

# Full build on first use (a few minutes). After that a cached rebuild takes
# a second and makes sure code changes are picked up.
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -q -t "$IMAGE" . >/dev/null
else
    docker build -t "$IMAGE" .
fi

# Inside the container photos/ sits in the working folder /app, just like here,
# so paths such as photos/img.jpg work the same in both places
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/photos:/app/photos" "$IMAGE" "$@"

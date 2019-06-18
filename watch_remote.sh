#!/usr/bin/env bash
set -e
set -x

# needs https://github.com/kimmobrunfeldt/chokidar-cli

rsync -vrazh --exclude=".*" . sn:/home/filter/code/hyperhyper/ &&
chokidar "**/*.*" -c "rsync -vrazh --exclude=".*" . sn:/home/filter/code/hyperhyper/"

#!/bin/bash

cd ../renders

for frame in $(ls '../renders'); do
    mitsuba $frame
done

# ffmpeg -framerate 24 -pattern_type glob -i '*.png' \
#   -c:v libx264 -r 30 -pix_fmt yuv420p animation.mp4

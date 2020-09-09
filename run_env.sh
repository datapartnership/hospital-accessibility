#!/bin/bash
echo "Running Container"

source ./.env


docker run -p 8888:8888 \
    -v "$PWD":/home/jovyan/work \
    --env GRANT_SUDO=yes \
    --env JUPYTER_ENABLE_LAB=yes \
    --env RESTARTABLE=yes \
    --env MAPBOX_TOKEN=$MAPBOX_TOKEN \
    --env PYTHONPATH=/home/jovyan/work/src \
    --rm \
    mrmaksimize/hospital-access-env:latest start-notebook.sh 


#!/bin/bash

INSTANCE=flax-tpu-v4-64
ZONE=us-central2-b
PROJECT=flax-mixtral

# run script.bash through run_script.bash
gcloud compute tpus tpu-vm create $INSTANCE --project=$PROJECT --zone=$ZONE \
    --version=tpu-ubuntu2204-base

pip install git+https://github.com/Additrien/FlaxMixtral.git@mixtral
pip install --upgrade jax[tpu] jaxlib flax -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
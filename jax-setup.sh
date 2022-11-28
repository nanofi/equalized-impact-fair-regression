#!/bin/bash

pip3 install --user --upgrade pip
pip3 install --user --upgrade jax jaxlib==0.1.59+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip3 install --user --upgrade flax pandas fire matplotlib seaborn scikit-learn


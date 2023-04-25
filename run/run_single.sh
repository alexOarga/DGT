#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
python3.9 main.py --cfg configs/example.yaml --repeat 3

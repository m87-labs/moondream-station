#!/bin/bash
set -e

docker build -t moondream-station-inference:latest -f dockerfile.inference .
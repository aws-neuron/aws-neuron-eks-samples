#!/bin/bash

myloc="$(dirname "$0")"

python "$myloc/inference.py" -p "A cat holding a sign that says hello world" -hh 512 -w 512 -m 32 -n 50

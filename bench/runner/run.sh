#!/bin/bash
# The runner the bench worker invokes. One argument: the job spec.
exec /home/ubuntu/gantry/.venv/bin/python /home/ubuntu/gantry_runner/runner.py "$1"

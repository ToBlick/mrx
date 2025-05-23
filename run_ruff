#!/bin/bash
# https://pycodestyle.pycqa.org/en/latest/intro.html#error-codes
set -ex

ruff check . --ignore F403,F405 || (exit 0)
ruff check --fix . --ignore F403,F405
ruff check . --ignore F403,F405

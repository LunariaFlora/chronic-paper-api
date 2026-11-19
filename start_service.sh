#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec "${CHRONIC_PYTHON:-.venv/bin/python}" -m service.server

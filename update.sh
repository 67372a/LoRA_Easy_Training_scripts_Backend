#!/bin/bash

export PATH="$HOME/.local/bin:$PATH"
git pull
python updater.py

#!/bin/bash

export PATH="$HOME/.local/bin:$PATH"
git pull
python3 updater.py

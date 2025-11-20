from pathlib import Path
from subprocess import check_call
import os
from installer import PLATFORM, setup_venv

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    check_call("git submodule update --init --recursive", shell=PLATFORM == "linux")
    os.chdir("sd_scripts")

    if PLATFORM == "windows":
        pip = Path("venv/Scripts/pip.exe")
    else:
        pip = Path("venv/bin/pip")
    setup_venv(pip)


if __name__ == "__main__":
    main()

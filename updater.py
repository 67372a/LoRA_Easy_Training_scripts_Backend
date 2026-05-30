from subprocess import check_call
import os

from install_uv import ensure_uv, find_existing_venv
from installer import PLATFORM, setup_venv

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    check_call("git submodule update --init --recursive", shell=PLATFORM == "linux")
    os.chdir("sd_scripts")

    uv = ensure_uv()
    venv_path = find_existing_venv("venv") or "venv"
    setup_venv(uv, venv_path)


if __name__ == "__main__":
    main()

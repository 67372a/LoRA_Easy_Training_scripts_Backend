@echo off

set PATH=%USERPROFILE%\.local\bin;%PATH%
git pull
python updater.py
pause

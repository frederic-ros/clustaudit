@echo off
REM Check the distributed results (tests, completeness and integrity of the result files, every
REM number of the paper, every table regenerated identically). No experiment is re-run.
cd /d "%~dp0.."
python scripts\reproduce.py verify-release %*
exit /b %ERRORLEVEL%

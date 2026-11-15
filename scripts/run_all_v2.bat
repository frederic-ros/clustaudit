@echo off
REM Full reproduction of the paper from scratch, in a NEW directory (default ..\clustaudit_reproduction).
REM Same steps on every platform: they are declared once, in scripts\reproduce.py.
REM   scripts\run_all_v2.bat [DIR] [options of "reproduce.py from-scratch", e.g. --skip deep]
REM The run stops at the first failed step; DIR\REPRODUCTION_REPORT.md says what was reproduced.
REM To check the distributed results without re-running anything: scripts\verify_release.bat
cd /d "%~dp0.."
set "DIR=%~1"
if "%DIR%"=="" set "DIR=..\clustaudit_reproduction"
shift
set "REST="
:collect
if "%~1"=="" goto run
set "REST=%REST% %1"
shift
goto collect
:run
python scripts\reproduce.py from-scratch --out "%DIR%" %REST%
exit /b %ERRORLEVEL%

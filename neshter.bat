@echo off
rem Neshter launcher: runs THIS checkout with the locked dependency set (uv.lock).
where uv >nul 2>nul
if not errorlevel 1 goto run
echo uv was not found. Neshter needs uv (https://docs.astral.sh/uv/).
echo It will be installed with Astral's official script (uv 0.13.6). Alternative: winget install astral-sh.uv
choice /M "Install uv now"
if errorlevel 2 goto declined
if errorlevel 1 goto install
goto declined
:install
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.13.6/install.ps1 | iex"
if errorlevel 1 goto fail
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 goto fail
:run
rem With an NVIDIA card the CUDA build of torch is used; NESHTER_TORCH=cpu turns that off.
set "TORCH_GROUPS="
if /I "%NESHTER_TORCH%"=="cpu" goto launch
where nvidia-smi >nul 2>nul
if errorlevel 1 goto launch
nvidia-smi -L >nul 2>nul
if errorlevel 1 goto launch
set "TORCH_GROUPS=--no-group cpu --group gpu"
:launch
uv run --project "%~dp0." --frozen %TORCH_GROUPS% neshter %*
set "rc=%errorlevel%"
if not "%rc%"=="0" pause
exit /b %rc%
:declined
echo Cancelled. Install uv yourself and run this again.
pause
exit /b 1
:fail
echo uv could not be installed. Install it yourself: https://docs.astral.sh/uv/
pause
exit /b 1

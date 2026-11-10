@echo off
rem Neser launcher: installs uv if missing, then runs "uvx neshter".
where uv >nul 2>nul
if not errorlevel 1 goto run
echo uv bulunamadi, resmi kurulum betigiyle kuruluyor...
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
if errorlevel 1 goto fail
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
:run
uvx neshter %*
exit /b %errorlevel%
:fail
echo uv kurulamadi. Elle kurun: https://docs.astral.sh/uv/
exit /b 1

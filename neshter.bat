@echo off
rem Neser launcher: runs THIS checkout with the locked dependency set (uv.lock).
where uv >nul 2>nul
if not errorlevel 1 goto run
echo uv bulunamadi. Neser'i calistirmak icin uv gerekli (https://docs.astral.sh/uv/).
echo Kurulum Astral'in resmi betigiyle yapilacak (surum 0.13.6). Alternatif: winget install astral-sh.uv
choice /M "uv simdi kurulsun mu"
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
uv run --project "%~dp0." --frozen neshter %*
set "rc=%errorlevel%"
if not "%rc%"=="0" pause
exit /b %rc%
:declined
echo Kurulum iptal edildi. uv'yi elle kurup yeniden deneyin.
pause
exit /b 1
:fail
echo uv kurulamadi. Elle kurun: https://docs.astral.sh/uv/
pause
exit /b 1

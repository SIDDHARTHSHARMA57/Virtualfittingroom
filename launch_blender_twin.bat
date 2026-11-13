@echo off
rem Opens Blender with the live SMPL-X digital twin receiver. Run from anywhere.
rem Start the dashboard too (launch_dashboard.bat); it streams the pose to this receiver.
setlocal
set "ROOT=%~dp0"
set "VFR_ROOT=%ROOT%"

if defined BLENDER if exist "%BLENDER%" goto run
if exist "E:\Apps\Blender\blender.exe" (set "BLENDER=E:\Apps\Blender\blender.exe" & goto run)
for /f "delims=" %%B in ('where blender 2^>nul') do (set "BLENDER=%%B" & goto run)
for /d %%D in ("%ProgramFiles%\Blender Foundation\Blender*") do if exist "%%D\blender.exe" set "BLENDER=%%D\blender.exe"
if defined BLENDER goto run

echo Blender not found. Install it with:
echo     winget install BlenderFoundation.Blender
echo or set BLENDER=C:\path\to\blender.exe and re-run.
exit /b 1

:run
echo Using %BLENDER%
"%BLENDER%" --python "%ROOT%src\blender_smplx_receiver.py"

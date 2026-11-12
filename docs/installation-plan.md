# VirtualFittingRoom installation plan

## Current stage: inspection completed; compatibility not yet proven

1. Review `inspection-report.txt` and confirm available Workspace space.
2. Verify the installed NVIDIA driver and select a PyTorch build compatible
   with the driver's supported CUDA runtime.
3. Select a compatible Python version in a project-local environment.
   Do not modify system Python.
4. Read the official SMPL-X and SMPLify-X installation instructions,
   dependency versions, and license terms before installing dependencies.
5. Obtain SMPL-X model assets and VPoser checkpoints only through their
   authorized distribution process. Record license and storage requirements.
6. Generate a neutral SMPL-X mesh and a second mesh with different beta
   shape parameters. Export and test both.
7. Test the official SMPL Unreal plugin against this exact Unreal Engine
   version and Linux build. Do not assume compatibility.
8. If plugin integration is incompatible, evaluate a verified interchange
   format and document losses such as rigging or pose-corrective shapes.
9. Only after static mesh validation, implement saved-image body fitting.
10. Add camera streaming only after static fitting and Unreal display work.

## Safety rules

- Project assets, environments, caches and outputs should be on Workspace.
- No large downloads, training jobs, system package installation or deletion
  is initiated by the inspection script.
- Licensed SMPL-X assets and checkpoints are not presumed to be included
  in public Git repositories.
- A successful repository connection does not prove plugin compatibility.
- Features must be marked verified only after running their tests.

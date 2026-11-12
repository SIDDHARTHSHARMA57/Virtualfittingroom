#!/bin/bash
# VirtualFittingRoom Dashboard Launcher
cd /media/siddharth-sharam/Workspace/VirtualFittingRoom
export PYTHONPATH="/media/siddharth-sharam/Workspace/VirtualFittingRoom:$PYTHONPATH"
/media/siddharth-sharam/Workspace/VirtualFittingRoom/envs/stage1/bin/python src/camera/dashboard.py "$@"

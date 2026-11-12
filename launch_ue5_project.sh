#!/bin/bash
ENGINE_DIR="/media/siddharth-sharam/Workspace/Unreal Engine/Linux_Unreal_Engine_5.8.3"
PROJECT_FILE="/media/siddharth-sharam/Workspace/VirtualFittingRoom/unreal_project/VirtualFittingRoom.uproject"

echo "Launching Unreal Engine 5.8.3 with VirtualFittingRoom project..."
"$ENGINE_DIR/Engine/Binaries/Linux/UnrealEditor" "$PROJECT_FILE" -log "$@"

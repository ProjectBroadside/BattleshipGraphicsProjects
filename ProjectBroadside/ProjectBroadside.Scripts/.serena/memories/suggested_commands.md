# Suggested Development Commands

## Unity Development Commands
Since this is a Unity project, most development happens within the Unity Editor. However, there are useful terminal commands for project management.

## File Management Commands (Windows PowerShell)
```powershell
# Navigate to project directory
cd "C:\_Development\ProjectBroadside.Scripts"

# List directory contents
ls
dir

# Find specific files
Get-ChildItem -Recurse -Name "*.cs" | Where-Object { $_ -like "*Projectile*" }

# Search for text in files
Select-String -Path "*.cs" -Pattern "BuoyancySystem" -Recurse

# Copy files
Copy-Item "source.cs" "destination.cs"

# Remove files (be careful!)
Remove-Item "filename.cs"
```

## Python Utility Scripts
```powershell
# Run namespace remover utility
python namespace_remover.py

# Run window activation utility
python activate_window.py
```

## Git Commands (Standard)
```powershell
# Check status
git status

# Stage changes
git add .
git add "specific_file.cs"

# Commit changes
git commit -m "Descriptive commit message"

# Push changes
git push

# Pull latest changes
git pull

# Create and switch to new branch
git checkout -b feature/new-feature-name

# View commit history
git log --oneline
```

## Unity-Specific Commands
```powershell
# Force Unity to regenerate project files (if needed)
# This is typically done through Unity Editor: Assets > Reimport All

# Check Unity version (if Unity is in PATH)
Unity.exe -version
```

## Development Workflow
1. **Code Changes**: Make changes in IDE (typically Visual Studio/Rider)
2. **Unity Editor**: Test changes in Unity Editor
3. **Git Management**: Commit changes using git commands above
4. **Build Testing**: Test builds through Unity Editor
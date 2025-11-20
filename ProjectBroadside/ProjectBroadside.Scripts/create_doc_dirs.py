
import os

doc_dirs_to_create = [
    "C:/_Development/ProjectBroadside.Scripts/.Documentation/2_Core_Systems/",
    "C:/_Development/ProjectBroadside.Scripts/.Documentation/3_Gameplay_Features/",
    "C:/_Development/ProjectBroadside.Scripts/.Documentation/4_Technical_Guides/",
    "C:/_Development/ProjectBroadside.Scripts/.Documentation/Archive/"
]

for d in doc_dirs_to_create:
    os.makedirs(d, exist_ok=True)
    print(f"Created documentation directory: {d}")


import os

dirs_to_create = [
    "C:/_Development/ProjectBroadside.Scripts/_STAGING/Core/",
    "C:/_Development/ProjectBroadside.Scripts/_STAGING/Gameplay/",
    "C:/_Development/ProjectBroadside.Scripts/_STAGING/Data/",
    "C:/_Development/ProjectBroadside.Scripts/_STAGING/ECS/",
    "C:/_Development/ProjectBroadside.Scripts/_STAGING/ECS/Authoring/",
    "C:/_Development/ProjectBroadside.Scripts/_STAGING/ECS/Components/",
    "C:/_Development/ProjectBroadside.Scripts/_STAGING/ECS/Systems/",
    "C:/_Development/ProjectBroadside.Scripts/_STAGING/ECS/Systems/Physics/",
    "C:/_Development/ProjectBroadside.Scripts/_STAGING/ECS/Systems/Bridges/",
    "C:/_Development/ProjectBroadside.Scripts/_STAGING/ECS/Queues/",
    "C:/_Development/ProjectBroadside.Scripts/_STAGING/Turrets/",
    "C:/_Development/ProjectBroadside.Scripts/_STAGING/Utilities/"
]

for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)
    print(f"Created directory: {d}")

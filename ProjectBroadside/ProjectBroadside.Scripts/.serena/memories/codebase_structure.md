# Codebase Structure Overview

## Current Directory Structure
```
ProjectBroadside.Scripts/
├── .Documentation/           # Project documentation and design docs
├── .vscode/                 # VS Code configuration
├── Authoring/               # DOTS authoring components
│   ├── ProjectileAuthoring.cs
│   ├── ShipAuthoring.cs
│   └── ShipProperties.cs
├── Components/              # ECS component definitions
│   ├── FireControlComponents.cs
│   ├── ProjectileComponents.cs
│   └── TorpedoComponents.cs
├── Core/                    # Core utility systems
│   └── PoolManager.cs
├── Dependencies/            # External system dependencies
│   ├── EntityProxy.cs
│   ├── FireControlComponents.cs
│   ├── FireControlSystem.cs
│   ├── FireRequestQueue.cs
│   └── PotentialTarget.cs
├── Queues/                  # Communication queues between systems
│   ├── CommandType.cs
│   └── FireRequestQueue.cs
├── Systems/                 # ECS systems
│   └── Physics/
│       ├── BallisticsSystem.cs
│       ├── BuoyancyBridgeSystem.cs
│       ├── BuoyancyComponents.cs
│       ├── ImpactEventBridgeSystem.cs
│       └── PhysicsMetrics.cs
├── Utilities/               # Helper scripts and tools
└── Root Level Scripts/      # Legacy scripts to be reorganized
    ├── Projectile.cs
    ├── ProjectileDefinition.cs
    ├── Turret.cs
    ├── TurretDefinition.cs
    └── Various other .cs files
```

## Key Architecture Components

### DOTS Layer (Performance Critical)
- **Physics Systems**: Ballistics, buoyancy, collisions
- **Entity Management**: Projectile pooling and lifecycle
- **Data Components**: Pure data structures for ECS

### MonoBehaviour Layer (Gameplay Logic)  
- **AI and Command**: Ship AI, formation systems
- **Damage Processing**: Ship damage calculations
- **UI and Player Interaction**: Game interface systems

### Bridge Systems (Inter-layer Communication)
- **FireControlBridgeSystem**: MonoBehaviour fire requests → DOTS projectiles
- **ImpactEventBridgeSystem**: DOTS collisions → MonoBehaviour damage
- **EntityProxy**: Maps DOTS entities to GameObjects

## Development Status
The project is currently in a transitional state with many root-level scripts that need reorganization into the proper folder structure as outlined in the project analysis documentation.
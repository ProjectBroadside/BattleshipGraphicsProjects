# Technical Stack and Dependencies

## Core Technologies
- **Unity 2023.x**: Game engine
- **Unity DOTS**: Data-Oriented Technology Stack for performance-critical systems
  - Unity.Entities
  - Unity.Physics 
  - Unity.Mathematics
  - Unity.Collections
  - Unity.Burst
  - Unity.Transforms
- **Crest Ocean System**: Third-party ocean simulation and rendering
  - WaveHarmonic.Crest
  - WaveHarmonic.Crest.Scripting
  - WaveHarmonic.Crest.Shared

## Language and Platform
- **Primary Language**: C# (Unity/DOTS)
- **Utility Scripts**: Python (namespace management, build utilities)
- **Target Platform**: Windows (development environment)
- **Build System**: Unity Build Pipeline

## Key Dependencies
All dependencies are managed through Unity Package Manager as defined in `ProjectBroadside.asmdef`:
- Unity ECS packages for high-performance simulation
- Crest for ocean rendering and wave simulation
- Standard Unity physics and mathematics libraries

## Assembly Definition
The project uses a single assembly definition (`ProjectBroadside.asmdef`) with `allowUnsafeCode: true` for DOTS performance optimizations.
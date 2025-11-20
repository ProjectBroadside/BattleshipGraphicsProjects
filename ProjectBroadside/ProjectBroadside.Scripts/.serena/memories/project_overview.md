# Project Broadside Overview

## Project Purpose
Project Broadside is a naval combat simulation game built in Unity, focusing on detailed ship-to-ship warfare with realistic ballistics, damage systems, and naval physics. The game features a hybrid architecture combining Unity's Data-Oriented Technology Stack (DOTS) for performance-critical systems with traditional MonoBehaviours for gameplay logic.

## Key Features
- **Realistic Naval Combat**: Detailed ballistics simulation with projectile physics, armor penetration mechanics, and ship damage systems
- **Ocean Physics**: Integration with Crest ocean system for realistic wave simulation and ship buoyancy
- **Ship Customization**: Shipyard system for vessel customization and fleet management
- **AI-Driven Command**: Ship AI and formation systems for squadron-based naval combat
- **Dynamic Weather**: Environmental effects on combat and navigation

## Architecture Philosophy
The project uses a **hybrid DOTS/MonoBehaviour architecture**:
- **DOTS Layer**: High-performance physics simulation (ballistics, buoyancy, collisions)
- **MonoBehaviour Layer**: Gameplay logic, UI, AI systems, damage processing
- **Bridge Systems**: Communication layer between DOTS and GameObject worlds

## Current Development Status
The project is in foundational development phase with core physics systems implemented but most gameplay features existing as placeholders and detailed design documents. The codebase requires consolidation and organization to align with the documented architectural vision.
# Design Patterns and Guidelines

## Architectural Patterns

### Hybrid DOTS/MonoBehaviour Pattern
- **DOTS Systems**: Use for performance-critical, data-parallel operations
- **MonoBehaviours**: Use for complex state machines, UI, and event-driven logic
- **Bridge Systems**: Communication layer between the two paradigms

### Component Composition over Inheritance
- Prefer small, focused components over large monolithic ones
- Use composition to build complex entity behaviors
- Tag components for categorization (e.g., `ActiveProjectileTag`)

### Data-Oriented Design Principles
- **Structure of Arrays**: Group similar data together in components
- **Minimal Dependencies**: Keep systems loosely coupled
- **Burst-Compatible Code**: Write algorithms that can be optimized by Burst compiler

## Performance Guidelines

### Memory Management
- **No Runtime Allocations**: Avoid `new` allocations in update loops
- **Component Pooling**: Reuse entity instances for projectiles and temporary objects
- **Blob Assets**: Use for immutable shared data (e.g., ship configurations)

### System Update Order
- **Physics First**: Update physics systems before gameplay logic
- **Bridge Systems**: Run after their respective domains complete
- **UI Last**: Update presentation systems after all simulation

### Query Optimization
- **RequireForUpdate**: Only run systems when necessary entities exist
- **Component Filtering**: Use specific component combinations in queries
- **Structural Changes**: Batch entity creation/destruction when possible

## Unity-Specific Patterns

### Crest Integration
- **Custom Physics**: Use Crest for visuals, custom DOTS physics for simulation
- **Query Batching**: Limit water surface queries per frame for performance
- **LOD System**: Reduce buoyancy calculation frequency for distant ships

### Asset Pipeline
- **ScriptableObjects**: Use for ship definitions and weapon configurations
- **Authoring Components**: Convert GameObject data to ECS components at bake time
- **Prefab Workflow**: Maintain prefabs for ships and projectile types
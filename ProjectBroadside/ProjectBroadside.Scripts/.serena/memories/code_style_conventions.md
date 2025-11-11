# Code Style and Conventions

## C# Coding Standards

### Naming Conventions
- **Classes**: PascalCase (e.g., `BallisticsSystem`, `ProjectileData`)
- **Methods**: PascalCase (e.g., `OnCreate`, `Execute`)
- **Properties**: PascalCase (e.g., `DeltaTime`, `Gravity`)
- **Fields**: camelCase for public fields in structs (e.g., `TimeAlive`, `MaxLifetime`)
- **Enums**: PascalCase for both enum and values (e.g., `ProjectileType.ArmorPiercing`)

### DOTS/ECS Patterns
- **Systems**: Use `ISystem` interface with `[BurstCompile]` attribute
- **Jobs**: Implement `IJobEntity` for entity iteration with burst compilation
- **Components**: Use `IComponentData` for simple data, `IBufferElementData` for collections
- **Tags**: Simple empty structs for entity categorization (e.g., `ActiveProjectileTag`)

### File Organization
- **Folder Structure**: Feature-based organization (Physics, Visuals, Components, etc.)
- **Component Definitions**: Group related components in single files
- **System Groups**: Organize systems by update order and dependencies

### Documentation Standards
- **XML Comments**: Use for public APIs and complex algorithms
- **Inline Comments**: Explain complex physics calculations and architectural decisions
- **Header Comments**: Include purpose and architectural context for major systems

### Performance Considerations
- **Burst Compilation**: All DOTS jobs and systems should use `[BurstCompile]`
- **Memory Allocation**: Avoid runtime allocations in hot paths
- **Component Queries**: Use `RequireForUpdate` to optimize system execution
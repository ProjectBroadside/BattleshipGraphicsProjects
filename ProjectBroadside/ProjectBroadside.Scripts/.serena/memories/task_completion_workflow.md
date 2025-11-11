# Development Workflow and Task Completion

## When a Task is Completed

### Code Quality Checks
1. **Compilation**: Ensure all code compiles without errors in Unity Editor
2. **DOTS Compatibility**: Verify all ECS components are properly defined as `IComponentData`
3. **Burst Compatibility**: Ensure all systems and jobs can compile with Burst
4. **Performance**: Check that no runtime allocations occur in hot paths

### Testing Protocol
1. **Unity Editor Testing**: Test functionality in Unity Editor play mode
2. **Physics Verification**: Verify ballistics and buoyancy calculations are working
3. **Bridge System Testing**: Ensure communication between DOTS and MonoBehaviour layers
4. **Memory Profiler**: Check for memory leaks or excessive allocations

### Documentation Updates
1. **Code Comments**: Update XML documentation for public APIs
2. **Architecture Notes**: Update design documents if architectural changes made
3. **Component Dependencies**: Document any new component relationships

### Version Control
1. **Commit Message**: Use descriptive commit messages following project conventions
2. **Branch Management**: Use feature branches for significant changes
3. **Code Review**: Review changes for adherence to coding standards

## Common Development Tasks

### Adding New ECS Components
1. Define in appropriate Components/ subdirectory
2. Ensure proper `IComponentData` implementation
3. Add to relevant authoring components
4. Update related systems to use new components

### Adding New Systems
1. Place in appropriate Systems/ subdirectory
2. Use `[BurstCompile]` and proper update group attributes
3. Implement required dependencies with `RequireForUpdate`
4. Add performance monitoring if needed

### Modifying Bridge Systems
1. Test both DOTS and MonoBehaviour sides
2. Verify EntityProxy mappings remain valid
3. Check queue systems for proper threading
4. Validate performance impact
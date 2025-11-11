# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.2] - 2025-07-05

### Improved
- **Force Install Server Button**: Added detailed logging and user feedback
- Install process now shows progress messages in Unity console
- Better error handling and status reporting during npm install/build operations
- Users can now see when installation starts, progresses, and completes

## [1.2.1] - 2025-07-05

### Fixed
- **Package Name in Settings**: Fixed `PackageName` constant in `McpUnitySettings.cs` to match the actual package name
- Resolves "Could not locate Server directory" error in Unity Editor
- Unity can now properly locate the Server~ directory using the correct package name

## [1.2.0] - 2025-07-05

### Added
- **Resource Discovery Fix**: Added `listResources` handler to enable VS Code to discover available Unity resources
- Resources now properly discoverable by MCP clients (VS Code, Claude, etc.)

### Changed  
- **Package Name**: Changed from `com.gamelovers.mcp-unity` to `com.coldaine.mcp-unity-rockstar`
- Package folder will now be `com.coldaine.mcp-unity-rockstar` instead of `com.gamelovers.mcp-unity`
- Updated package metadata to reflect proper ownership and naming

### Fixed
- VS Code MCP clients can now see both tools AND resources (previously only tools were visible)

## [1.1.3] - 2025-07-04

### Added
- New `unity://scenes_hierarchy_simple` resource that provides a lightweight view of the scene hierarchy
- `GetScenesHierarchySimpleResource.cs` - Unity C# resource implementation
- `getScenesHierarchySimpleResource.ts` - Node.js MCP server handler
- Enhanced performance for AI agents that only need basic scene structure information

### Changed
- Updated MCP server to register the new simplified hierarchy resource
- Enhanced resource documentation with lightweight alternative option

### Benefits
- **Performance**: Faster response times for AI agents needing only basic scene info
- **Efficiency**: Reduced data transfer compared to full hierarchy resource
- **Compatibility**: Fully backward compatible, original hierarchy resource still available

### Usage
The new resource can be accessed via `unity://scenes_hierarchy_simple` and returns:
- Scene name
- GameObject name and instance ID
- Hierarchical children structure
- No additional properties (transforms, components, active state, etc.)

This is ideal for AI agents that need to understand scene structure without the overhead of complete GameObject information.

---

## About This Fork

This is **MCP Unity Rockstar**, an enhanced fork of the original [MCP Unity](https://github.com/CoderGamester/mcp-unity) by CoderGamester.

### Enhanced Features
- **Lightweight Scene Hierarchy**: New `unity://scenes-hierarchy-simple` resource
- **Improved Performance**: Optimized for AI agents requiring basic scene structure
- **Better Package Management**: Enhanced Unity Package Manager integration
- **Version Consistency**: Standardized versioning across all components

### Original Project
- **Creator**: CoderGamester
- **Repository**: https://github.com/CoderGamester/mcp-unity
- **License**: MIT

Special thanks to CoderGamester for creating the foundation that made this enhanced version possible.
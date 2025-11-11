// Import MCP SDK components
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { ListResourcesRequestSchema, ReadResourceRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { McpUnity } from './unity/mcpUnity.js';
import { Logger, LogLevel } from './utils/logger.js';
import { registerMenuItemTool } from './tools/menuItemTool.js';
import { registerSelectGameObjectTool } from './tools/selectGameObjectTool.js';
import { registerAddPackageTool } from './tools/addPackageTool.js';
import { registerRunTestsTool } from './tools/runTestsTool.js';
import { registerSendConsoleLogTool } from './tools/sendConsoleLogTool.js';
import { registerGetConsoleLogsTool } from './tools/getConsoleLogsTool.js';
import { registerUpdateComponentTool } from './tools/updateComponentTool.js';
import { registerAddAssetToSceneTool } from './tools/addAssetToSceneTool.js';
import { registerUpdateGameObjectTool } from './tools/updateGameObjectTool.js';
import { registerGetScenesHierarchySimpleTool } from './tools/getScenesHierarchySimpleTool.js';
import { registerGetScenesHierarchyTool } from './tools/getScenesHierarchyTool.js';
import { registerGetScenesHierarchySimpleTool } from './tools/getScenesHierarchySimpleTool.js';
import { registerGetScenesHierarchyTool } from './tools/getScenesHierarchyTool.js';
import { registerGetMenuItemsResource } from './resources/getMenuItemResource.js';
import { registerGetConsoleLogsResource } from './resources/getConsoleLogsResource.js';
import { registerGetHierarchyResource } from './resources/getScenesHierarchyResource.js';
import { registerGetHierarchySimpleResource } from './resources/getScenesHierarchySimpleResource.js';
import { registerGetPackagesResource } from './resources/getPackagesResource.js';
import { registerGetAssetsResource } from './resources/getAssetsResource.js';
import { registerGetTestsResource } from './resources/getTestsResource.js';
import { registerGetGameObjectResource } from './resources/getGameObjectResource.js';
import { registerGameObjectHandlingPrompt } from './prompts/gameobjectHandlingPrompt.js';

// Initialize loggers
const serverLogger = new Logger('Server', LogLevel.INFO);
const unityLogger = new Logger('Unity', LogLevel.INFO);
const toolLogger = new Logger('Tools', LogLevel.INFO);
const resourceLogger = new Logger('Resources', LogLevel.INFO);

// Initialize the MCP server
const server = new McpServer (
  {
    name: "MCP Unity Server",
    version: "1.0.0"
  },
  {
    capabilities: {
      tools: {},
      resources: {},
      prompts: {},
    },
  }
);

// Initialize MCP HTTP bridge with Unity editor
const mcpUnity = new McpUnity(unityLogger);

// Register all tools into the MCP server
registerMenuItemTool(server, mcpUnity, toolLogger);
registerSelectGameObjectTool(server, mcpUnity, toolLogger);
registerAddPackageTool(server, mcpUnity, toolLogger);
registerRunTestsTool(server, mcpUnity, toolLogger);
registerSendConsoleLogTool(server, mcpUnity, toolLogger);
registerGetConsoleLogsTool(server, mcpUnity, toolLogger);
registerUpdateComponentTool(server, mcpUnity, toolLogger);
registerAddAssetToSceneTool(server, mcpUnity, toolLogger);
registerUpdateGameObjectTool(server, mcpUnity, toolLogger);
registerGetScenesHierarchySimpleTool(server, mcpUnity, toolLogger);
registerGetScenesHierarchyTool(server, mcpUnity, toolLogger);

// Register all resources into the MCP server
registerGetTestsResource(server, mcpUnity, resourceLogger);
registerGetGameObjectResource(server, mcpUnity, resourceLogger);
registerGetMenuItemsResource(server, mcpUnity, resourceLogger);
registerGetConsoleLogsResource(server, mcpUnity, resourceLogger);
registerGetHierarchyResource(server, mcpUnity, resourceLogger);
registerGetHierarchySimpleResource(server, mcpUnity, resourceLogger);
registerGetPackagesResource(server, mcpUnity, resourceLogger);
registerGetAssetsResource(server, mcpUnity, resourceLogger);

// Register all prompts into the MCP server
registerGameObjectHandlingPrompt(server);

// Server startup function
async function startServer() {
  try {
    // Initialize STDIO transport for MCP client communication
    const stdioTransport = new StdioServerTransport();
    
    // Connect the server to the transport
    await server.connect(stdioTransport);

    serverLogger.info('MCP Server started');
    
    // Check actual SDK version running
    try {
      const sdkPackageJson = require('@modelcontextprotocol/sdk/package.json');
      serverLogger.info(`MCP SDK version running: ${sdkPackageJson.version}`);
    } catch (e) {
      serverLogger.warn('Could not determine MCP SDK version.');
    }
    
    // Get the client name from the MCP server
    const clientName = server.server.getClientVersion()?.name || 'Unknown MCP Client';
    serverLogger.info(`Connected MCP client: ${clientName}`);
    
    // Add listResources handler for resource discovery
    // This enables VS Code to discover what resources are available
    server.server.setRequestHandler(ListResourcesRequestSchema, async () => {
      resourceLogger.info('Handling listResources request for resource discovery');
      
      return {
        resources: [
          {
            uri: 'unity://scenes_hierarchy_simple',
            name: 'Simple Scene Hierarchy',
            description: 'Lightweight scene hierarchy (name, instanceId, children only)',
            mimeType: 'application/json'
          },
          {
            uri: 'unity://scenes_hierarchy',
            name: 'Scene Hierarchy',
            description: 'Complete scene hierarchy with all GameObject details',
            mimeType: 'application/json'
          },
          {
            uri: 'unity://game_objects',
            name: 'Game Objects',
            description: 'Unity game objects with detailed information',
            mimeType: 'application/json'
          },
          {
            uri: 'unity://packages',
            name: 'Packages',
            description: 'Unity Package Manager packages',
            mimeType: 'application/json'
          },
          {
            uri: 'unity://assets',
            name: 'Assets',
            description: 'Unity project assets from Asset Database',
            mimeType: 'application/json'
          },
          {
            uri: 'unity://tests',
            name: 'Tests',
            description: 'Unity test assemblies and test cases',
            mimeType: 'application/json'
          },
          {
            uri: 'unity://console_logs',
            name: 'Console Logs',
            description: 'Unity console logs and messages',
            mimeType: 'text/plain'
          },
          {
            uri: 'unity://menu_items',
            name: 'Menu Items',
            description: 'Unity Editor menu items',
            mimeType: 'application/json'
          }
        ]
      };
    });
    
    // Add readResource handler for resource access
    // This enables VS Code to actually read the content of resources
    server.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const uri = request.params.uri;
      resourceLogger.info(`Handling readResource request for URI: ${uri}`);
      
      try {
        // The server.resource() registrations should have created internal handlers
        // We need to route the request to the appropriate handler based on URI
        const result = await server.readResource({ uri });
        resourceLogger.info(`Successfully handled readResource for URI: ${uri}`);
        return result;
      } catch (error) {
        resourceLogger.error(`Failed to handle readResource for URI: ${uri}`, error);
        throw error;
      }
    });
    
    // Start Unity Bridge connection with client name in headers
    await mcpUnity.start(clientName);
    
  } catch (error) {
    serverLogger.error('Failed to start server', error);
    process.exit(1);
  }
}

// Start the server
startServer();

// Handle shutdown
process.on('SIGINT', async () => {
  serverLogger.info('Shutting down...');
  await mcpUnity.stop();
  process.exit(0);
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  serverLogger.error('Uncaught exception', error);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason) => {
  serverLogger.error('Unhandled rejection', reason);
});

import * as z from "zod";
import { Logger } from "../utils/logger.js";
import { McpUnity } from "../unity/mcpUnity.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { McpUnityError, ErrorType } from "../utils/errors.js";
import { CallToolResult } from "@modelcontextprotocol/sdk/types.js";

// Constants for the tool
const toolName = "get_scenes_hierarchy_simple";
const toolDescription = "Retrieves a simplified hierarchy of all GameObjects in Unity loaded scenes (lightweight version with only name, instanceId, and children)";
const paramsSchema = z.object({
  // No parameters needed for this tool - it fetches all hierarchy data
});

/**
 * Creates and registers the Get Scenes Hierarchy Simple tool with the MCP server
 * This tool allows retrieving simplified scene hierarchy data as a tool rather than resource
 *
 * @param server The MCP server instance to register with
 * @param mcpUnity The McpUnity instance to communicate with Unity
 * @param logger The logger instance for diagnostic information
 */
export function registerGetScenesHierarchySimpleTool(
  server: McpServer,
  mcpUnity: McpUnity,
  logger: Logger
) {
  logger.info(`Registering tool: ${toolName}`);

  // Register this tool with the MCP server
  server.tool(
    toolName,
    toolDescription,
    paramsSchema.shape,
    async (params: z.infer<typeof paramsSchema>) => {
      try {
        logger.info(`Executing tool: ${toolName}`, params);
        const result = await toolHandler(mcpUnity, params);
        logger.info(`Tool execution successful: ${toolName}`);
        return result;
      } catch (error) {
        logger.error(`Tool execution failed: ${toolName}`, error);
        throw error;
      }
    }
  );
}

/**
 * Handles requests for Unity simplified scene hierarchy
 *
 * @param mcpUnity The McpUnity instance to communicate with Unity
 * @param params The parameters for the tool (empty for this tool)
 * @returns A promise that resolves to the tool execution result
 * @throws McpUnityError if the request to Unity fails
 */
async function toolHandler(
  mcpUnity: McpUnity,
  params: z.infer<typeof paramsSchema>
): Promise<CallToolResult> {
  // Send request to Unity using the same method name as the resource
  // This allows reusing the existing Unity-side implementation
  const response = await mcpUnity.sendRequest({
    method: "get_scenes_hierarchy_simple",
    params: {},
  });

  if (!response.success) {
    throw new McpUnityError(
      ErrorType.TOOL_EXECUTION,
      response.message || "Failed to fetch simplified scene hierarchy from Unity"
    );
  }

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(
          response.hierarchy || response.data || response,
          null,
          2
        ),
      },
    ],
  };
}
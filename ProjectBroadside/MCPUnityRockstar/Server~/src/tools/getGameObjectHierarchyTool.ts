import * as z from "zod";
import { Logger } from "../utils/logger.js";
import { McpUnity } from "../unity/mcpUnity.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { McpUnityError, ErrorType } from "../utils/errors.js";
import { CallToolResult } from "@modelcontextprotocol/sdk/types.js";

const toolName = "get_gameobject_hierarchy";
const toolDescription =
  "Retrieves the full hierarchy for a single GameObject and its children, identified by instanceId or hierarchyPath.";
const paramsSchema = z.object({
  instanceId: z.number().optional(),
  hierarchyPath: z.string().optional(),
});

export function registerGetGameObjectHierarchyTool(
  server: McpServer,
  mcpUnity: McpUnity,
  logger: Logger
) {
  logger.info(`Registering tool: ${toolName}`);

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

async function toolHandler(
  mcpUnity: McpUnity,
  params: z.infer<typeof paramsSchema>
): Promise<CallToolResult> {
  const response = await mcpUnity.sendRequest({
    method: "get_scenes_hierarchy",
    params: {},
  });

  if (!response.success) {
    throw new McpUnityError(
      ErrorType.TOOL_EXECUTION,
      response.message || "Failed to fetch complete scene hierarchy from Unity"
    );
  }

  const hierarchy = response.hierarchy || response.data || response;
  let subtree = null;

  if (params.instanceId !== undefined) {
    subtree = findNodeByInstanceId(hierarchy, params.instanceId);
  } else if (params.hierarchyPath) {
    subtree = findNodeByHierarchyPath(hierarchy, params.hierarchyPath);
  } else {
    throw new McpUnityError(
      ErrorType.TOOL_EXECUTION,
      "Must provide either instanceId or hierarchyPath"
    );
  }

  if (!subtree) {
    throw new McpUnityError(
      ErrorType.TOOL_EXECUTION,
      "GameObject not found in hierarchy"
    );
  }

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(subtree, null, 2),
      },
    ],
  };
}

function findNodeByInstanceId(node: any, instanceId: number): any {
  if (node.instanceId === instanceId) return node;
  if (Array.isArray(node.children)) {
    for (const child of node.children) {
      const found = findNodeByInstanceId(child, instanceId);
      if (found) return found;
    }
  }
  return null;
}

function findNodeByHierarchyPath(node: any, path: string): any {
  const segments = path.split("/").filter(Boolean);
  return findNodeByPathSegments(node, segments);
}

function findNodeByPathSegments(node: any, segments: string[]): any {
  if (segments.length === 0) return node;
  const [head, ...rest] = segments;
  if (Array.isArray(node.children)) {
    for (const child of node.children) {
      if (child.name === head) {
        return findNodeByPathSegments(child, rest);
      }
    }
  }
  return null;
}

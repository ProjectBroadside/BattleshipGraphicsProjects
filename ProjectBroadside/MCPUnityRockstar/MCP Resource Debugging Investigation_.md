

# **Debugging MCP Resource Inaccessibility: A Comprehensive Guide for Unity-Node.js Integrations**

## **1\. Executive Summary**

This report addresses a critical issue encountered in Model Context Protocol (MCP) server implementations where resources are correctly advertised and discovered by clients like VS Code, yet remain inaccessible when queried. This contrasts sharply with the flawless operation of MCP tools within the same environment. The analysis focuses on a specific scenario involving a Node.js MCP server utilizing the @modelcontextprotocol/sdk with stdio transport, communicating with a Unity Editor via a WebSocket bridge, and exposing both simple and parameterized URIs (e.g., unity://gameobject/{id}).

The core of the problem likely resides in the intricate interplay between client-side resource interpretation, the server's URI pattern matching, and the reliability of the underlying Unity-Node.js WebSocket bridge. While resource discovery (listResources) confirms the server's basic health and capability advertisement, the failure of resource access (readResource) points to issues within the specific handling of resource requests, particularly the parsing of parameterized URIs and the subsequent data retrieval from the Unity Editor.

Immediate recommendations for resolution include a systematic debugging approach focusing on detailed logging within the readResource handlers, meticulous validation of URI parameters, and thorough inspection of the WebSocket communication and Unity-side data serialization. Addressing potential client-side misinterpretations of resource types, especially in rapidly evolving protocol versions, is also crucial.

## **2\. Understanding Model Context Protocol (MCP) Fundamentals**

The Model Context Protocol (MCP) serves as an open, standardized framework designed by Anthropic to bridge Large Language Models (LLMs), such as those powering Cursor or VS Code, with external tools and diverse data sources.1 Its primary objective is to equip LLMs with contextual information and the ability to execute actions without necessitating extensive model retraining or embedding specific tool logic directly into the AI.3 Communication within MCP is structured around JSON-RPC 2.0 messages, ensuring a consistent and interoperable data exchange.2 This design flexibility enables developers to construct MCP servers using various programming languages, including Python, JavaScript, and Go, leveraging transports like standard input/output (stdio) or HTTP.1

### **MCP Architecture: Host, Client, Server**

MCP operates on a well-defined client-host-server architecture, each component fulfilling distinct responsibilities.2

* **Host:** This refers to the LLM application itself, such as Cursor or the VS Code MCP panel. The host is responsible for initiating connections, managing multiple client instances, enforcing security policies, controlling permissions, and orchestrating the overall LLM integration.2  
* **Client:** Embedded within the host application, the client establishes isolated, stateful sessions with MCP servers. Its functions include negotiating the protocol, exchanging capabilities, and routing messages bidirectionally between the host and the server.2  
* **Server:** These are independent services designed to expose specialized context and capabilities. MCP servers are typically lightweight, focusing on well-defined responsibilities, and provide the actual data (resources) or functionality (tools) to the client and, by extension, the LLM.2

### **MCP Primitives: Resources, Tools, and Prompts – Core Distinctions**

MCP defines three fundamental primitives through which servers expose capabilities, each with a distinct purpose:

* **Resources:** These represent structured data or content that provides additional context to the LLM.2 Resources are conceptually similar to  
  GET endpoints in a REST API, primarily serving for data retrieval without performing significant computation or causing side effects.3 Examples include static configuration data (e.g.,  
  config://app), dynamic user profiles (e.g., users://{user\_id}/profile), file contents (e.g., file://some/path), or real-time market statistics (e.g., stock://AAPL/earnings).5 Resources are inherently "read-only" and are directly addressable via their unique URIs.9  
* **Tools:** In contrast to resources, tools are executable functions that empower LLMs to perform actions or retrieve information, often involving complex computations or interactions that result in side effects.2 They are analogous to  
  POST endpoints. Practical applications include calculating a Body Mass Index (BMI) or fetching current weather data by interacting with an external API.5  
* **Prompts:** These are pre-defined, reusable templates or instructions that guide and structure the interactions of LLMs with the server.2

A critical distinction for debugging purposes is the typical initiator of these primitives: tools are generally invoked by the LLM (model), whereas resources are typically accessed by the host application (client).3 This differentiation is paramount when diagnosing why resources might be inaccessible even if tools are functioning correctly.

The observation that resources are advertised (discovered) but not accessible, while tools work perfectly, provides a crucial diagnostic clue. If resource discovery, typically performed via a listResources call from the host, is successful, it confirms that the server's declaration of available resources and its fundamental connectivity are operational. However, if the subsequent access attempt, often a readResource call initiated by the host, fails, it strongly suggests that the issue is specific to how the host attempts to access the resource or how the server's readResource handler processes that particular access request. The unimpeded functionality of tools further isolates the problem to the resource handling path, indicating that the core issue likely lies within the readResource method, its URI parsing, or the underlying logic it invokes to retrieve data. This focused understanding significantly narrows the scope for effective debugging.

## **3\. MCP Resource Implementation Best Practices**

Proper implementation of MCP resources is crucial for their reliable operation. This section details the recommended patterns for resource registration, clarifies the distinction between discovery and access, and highlights common pitfalls, particularly concerning parameterized URIs.

### **Official MCP SDK Documentation & Resource Registration Patterns**

MCP resources are defined within server implementations using specific SDK mechanisms. In the Python SDK, for instance, functions are decorated with @mcp.resource() to register them as handlers for defined URI patterns.5 This decorator instructs the SDK's routing layer to invoke the decorated function when a client requests a URI matching the specified pattern.

Resource URIs can be static, serving fixed content, such as config://app, which would consistently return a predefined string.5 Alternatively, they can be dynamic, incorporating placeholders like

users://{user\_id}/profile. In such dynamic cases, the SDK's internal routing mechanism is designed to extract the variable value (e.g., 123 from users://123/profile) and pass it as an argument (e.g., user\_id: str) to the corresponding decorated function.5 The docstrings associated with these functions are not merely comments; they serve as descriptions for the resource, aiding client applications in discovering and understanding the resource's purpose and expected content.5 For the

@modelcontextprotocol/sdk used in Node.js (TypeScript), the registerResource method is the modern, recommended approach for new code, although older resource methods are typically maintained for backward compatibility.10

### **Resource Discovery (listResources) vs. Resource Access (readResource)**

The MCP protocol defines distinct mechanisms for clients to interact with resources: discovery and access.

* **Discovery (listResources):** Client applications, such as the VS Code MCP panel, employ methods like session.list\_resources() (in the Python SDK) or client.listResources() (in the Swift SDK) to enumerate the available resources and their associated URI templates.5 This operation primarily retrieves metadata about the resources, including their URI, name, description, and MIME type, allowing the client to understand  
  *what* resources are offered by the server. The user's report that "Resources appear correctly in VS Code MCP panel" confirms that this discovery phase is functioning as intended.  
* **Access (readResource):** To retrieve the actual content of a resource, clients initiate calls using methods like session.read\_resource("file://some/path") (Python SDK) or client.readResource() (Swift SDK), providing the specific resource URI.5 This action triggers the execution of the corresponding  
  readResource handler on the server, which is then responsible for fetching and returning the requested data. The observed problem—resources being discovered but not accessible—directly indicates a failure point within this resource access phase.

### **Common Pitfalls in Resource URI Pattern Matching and Parameter Handling**

The accurate handling of resource URIs, especially those with dynamic parameters, is a frequent source of implementation challenges.

* **URI Validation and Routing Control:** A fundamental best practice involves rigorously validating all incoming resource URIs against predefined patterns or regular expressions. For dynamic templates, it is imperative to ensure that extracted parameters strictly conform to expected formats and constraints. Requests with unknown or unregistered URI schemes or prefixes should be rejected to prevent malformed or forged URIs from inadvertently accessing unintended resources.12  
* **Input and Path Sanitization:** If resource URIs are designed to map to file paths or database queries (as unity://gameobject/{id} might suggest), stringent sanitization of all path and parameter inputs is essential. This measure is critical for mitigating common vulnerabilities such as directory traversal or SQL injection. For resources that interact with file systems, access should be strictly confined to approved directories.12  
* **Case Sensitivity:** A documented issue in certain MCP server implementations, such as the Filesystem MCP Server on Windows, relates to case sensitivity in path arguments, particularly drive letters. This can lead to "Access denied \- path outside allowed directories" errors if the casing of the requested URI's path components does not precisely match the server's internal canonicalization, even if the path is logically correct.13 This is particularly relevant for the  
  unity://gameobject/{id} URI if the id component or its resolution involves file system paths.  
* **MIME Type Validation and Safety:** Servers should explicitly declare the MIME type for every resource in both ListResourcesResponse and ReadResourceResponse. It is crucial to ensure that the declared MIME type accurately reflects the actual content being returned. Ambiguous or overly generic types like application/octet-stream should be avoided unless absolutely necessary, and unsafe types (e.g., text/html, application/javascript) should be sanitized or rejected to prevent client-side rendering or code execution risks.12  
* **Error Handling:** When parameter validation fails, servers are expected to return specific JSON-RPC errors, such as Invalid Parameters (-32602). This error code typically signifies that while the method exists, the arguments provided (derived from the URI) do not satisfy the handler's requirements.14

The presence of parameterized URIs, such as unity://gameobject/{id}, in the user's setup highlights URI parameter mismatch as a significant potential factor in the resource inaccessibility. The success of resource discovery, which only confirms the advertisement of the URI *template*, contrasts with the failure of actual access. This suggests that when VS Code attempts to readResource for a specific URI like unity://gameobject/123, the server's internal routing or the readResource handler itself fails to correctly parse or validate the {id} parameter from the URI. This could stem from a type mismatch (e.g., the handler expects an integer but receives a string that cannot be converted), a format discrepancy (e.g., the ID has a specific pattern not being met), encoding issues (special characters not correctly URL-encoded or decoded), or even case sensitivity in the ID's interpretation by the Unity side or the Node.js server. Consequently, debugging efforts must intensely focus on the exact format of the URI sent by VS Code for access and how the Node.js server's readResource handler extracts and processes that {id} parameter.

Furthermore, while primarily intended for security, the best practices for secure MCP resource access often lead to "access denied" or "resource not found" errors when misconfigured. If the server's security policies are overly strict or incorrectly set up, they can inadvertently block legitimate resource access attempts. For example, the server might anticipate specific authentication headers for readResource that the VS Code client is not providing, even if tools operate under a different permission model. Similarly, overly aggressive URI validation logic might reject valid URIs due to minor formatting discrepancies, or path sanitization could incorrectly flag a legitimate unity://gameobject/{id} URI as an attempt to access an "outside allowed directory" if the underlying Unity mapping involves a file system path.13 When these security mechanisms are improperly configured, they directly contribute to the "not accessible" symptoms, even if the resource is technically well-defined and its basic URI structure is sound. Therefore, debugging should include a thorough review of any security or access control layers on the MCP server that might be implicitly or explicitly preventing resource access based on URI patterns or client credentials.

| Pitfall | Example URI (Problematic Input) | Symptom(s) | Solution(s) |
| :---- | :---- | :---- | :---- |
| **Mismatched Parameter Types** | unity://gameobject/abc (expecting integer ID) | Invalid Parameters (-32602) | Implement strict input validation; explicitly cast parameters to expected types (e.g., parseInt(id)). |
| **Incorrect Delimiters/Format** | unity://gameobject:123 (expecting / separator) | Method Not Found (-32601) | Ensure URI pattern in server.resource() exactly matches client's request format. |
| **Case Sensitivity in Path Components** | unity://GameObject/{id} (server expects gameobject) | Method Not Found (-32601), Access Denied (if path-based security) | Normalize URI components to a consistent case on the server side; ensure client sends correct casing. Check OS-specific path sensitivity (e.g., Windows drive letters).13 |
| **URL Encoding/Decoding Issues** | unity://gameobject/my%20object (server fails to decode %20) | Invalid Parameters (-32602), Handler not invoked | Ensure proper URL encoding by client and decoding by server-side URI parsing logic. |
| **Missing Required Parameters** | unity://gameobject/ (missing {id}) | Invalid Parameters (-32602), Method Not Found (-32601) | Client must provide all required parameters; server should validate their presence. |

*Table 1: Common URI Pattern Matching Pitfalls and Solutions*

## **4\. Deep Dive into @modelcontextprotocol/sdk Resource Handling**

A thorough understanding of how the @modelcontextprotocol/sdk manages resource registration and request routing is essential for diagnosing the inaccessibility issue. This section delves into the SDK's internal mechanisms, discusses common operational challenges related to parameterized URIs, and highlights the impact of version-specific changes in the MCP specification.

### **Internal Resource Registration and Request Routing**

The @modelcontextprotocol/sdk (typically used in Node.js/TypeScript environments, mirroring the Python SDK's design) provides robust mechanisms for registering resource handlers. The server.resource(name, uri, metadata, handler) method, or its decorator equivalent in other SDKs like @mcp.resource() in Python, serves to associate a specific function with a defined URI pattern.5 When a client initiates a

resources/read request, the SDK's internal routing logic takes over. This logic is responsible for parsing the incoming request, matching the requested URI against the set of registered patterns, extracting any dynamic parameters (such as {id} from unity://gameobject/{id}), and subsequently dispatching the request to the appropriate handler function.5

Within both resource and tool handlers, the Context object (ctx) is a crucial component, providing access to various MCP capabilities. This includes methods for sending structured log messages (e.g., ctx.debug, ctx.info, ctx.warning, ctx.error) back to the client, offering real-time visibility into function execution. Furthermore, ctx.read\_resource allows a handler to programmatically access and read data from other resources within the same server, enabling the composition of more complex data structures or workflows.5 This internal resource access mechanism is vital for building sophisticated MCP servers.

### **Known Operational Challenges with Resource Parameter Templates**

While the provided documentation does not explicitly list "known issues" with resource parameter templates in the @modelcontextprotocol/sdk in the same way one might find bug reports, the frequent occurrence of the Invalid Parameters (-32602) error code 14 strongly suggests that incorrect parameter handling, type mismatches, or missing required fields during URI parsing are common operational problems encountered by developers. This error is explicitly described as "the most common error during active development as you refine your tool arguments and resource references".14

A concrete example illustrating a parameter-related challenge is observed in the Filesystem MCP Server on Windows. This server can exhibit case sensitivity in its allowed path arguments, particularly concerning drive letters. This sensitivity can lead to "Access denied \- path outside allowed directories" errors for seemingly valid subdirectories if the casing of the requested URI's path components does not precisely match the server's internal canonicalization.13 This highlights how subtle differences in URI parameters, even beyond their explicit values, can result in access failures.

Moreover, discussions within the Cursor community forum, which likely informs or influences the behavior of VS Code's MCP panel, have revealed historical challenges with Cursor's earlier implementations of MCP resources. It was noted that resources were "implemented as just tools," leading to scalability issues if a server exposed hundreds of resources.16 This suggests that the client-side interpretation of resource declarations can itself be a significant operational challenge impacting accessibility, even if the server's resource registration is technically correct according to the protocol.

### **Version-Specific Changes in Resource Handling**

The Model Context Protocol specification, along with its associated SDKs, is under active development and undergoes rapid evolution. Several revisions have introduced significant changes that can impact how resources are handled and consumed.17

* **2024-11-05 (Initial Schema Stabilization):** This foundational release introduced the core concepts of tools, resources, and prompts as distinct primitives within the MCP framework.17  
* **2025-03-26:** This revision brought major architectural changes, including the addition of a comprehensive OAuth 2.1 authorization framework, the replacement of Server-Sent Events (SSE) with a more flexible Streamable HTTP transport, and the initial introduction of JSON-RPC batching (a feature later reversed). It also expanded capabilities by adding support for audio data, progress messages, and tool annotations.17  
* **2025-06-18 (Most Recent):** This significant revision *removed JSON-RPC batching* (reversing the 2025-03-26 change), added support for structured tool output, and formally classified MCP servers as OAuth Resource Servers. This classification introduced new requirements for clients to implement Resource Indicators for enhanced security. Crucially for resource handling, this version also **added support for resource links in tool call results**.17 This change provides a standardized mechanism for tools to  
  *reference* resources without embedding their full content, potentially influencing how resources are expected to be consumed or linked within complex workflows.  
* The @modelcontextprotocol/sdk (TypeScript) itself has seen API evolution, with registerResource now being the recommended method over older resource methods, indicating a shift in preferred coding patterns.10

The most critical observation from the SDK analysis, particularly relevant to the user's problem, is the documented client-side challenges with resource handling, specifically from Cursor (which VS Code's MCP panel is likely based on or influenced by). The user's problem statement indicates that VS Code discovers resources but cannot access them. Historical context reveals that Cursor previously "removed" resource functionality in certain versions because "they were implemented as just tools so if someone had 200 resources you'd have 200 tools and break the entire request".16 This suggests a fundamental design challenge or bug in how Cursor's MCP client (and potentially VS Code's) internally handles the

*distinction* between resources and tools. Even if the Node.js MCP server correctly implements resources according to the latest specification, the VS Code client might be attempting to access them using an outdated or incompatible internal mechanism (e.g., trying to invoke them as if they were tools, or failing to properly construct readResource requests). This mismatch in client-side interpretation or implementation, rather than a server-side bug, could be a primary factor in the "advertised but not accessible" symptom. Therefore, verifying the exact version of the VS Code MCP panel/extension and checking for any known issues or limitations it might have regarding resource access, especially compared to tool access, becomes a crucial debugging step. The rapid evolution of the MCP specification further underscores that version compatibility between the server's SDK and the client's implementation is a significant potential point of failure.

Another important consideration is the implicit contract of resource handlers. While resources are defined as "read-only" and are generally expected "not to perform significant computation or have side effects" 5, the

readResource handler *does* perform the necessary computation to retrieve the data. In the user's scenario, resources like unity://gameobject/{id} imply fetching live data from a Unity Editor. This fetching operation inherently requires interaction with the Unity Editor via a WebSocket bridge, which involves I/O operations, potential network latency, and the execution of C\# code within Unity. If this underlying data retrieval process (ee.g., the call to Unity, or Unity's internal lookup for the gameobject/{id}) fails, takes too long, or returns malformed data, the readResource handler might return an empty or erroneous response, or even time out. From the client's perspective, this would manifest as the resource being "not accessible," even if the MCP SDK's routing to the handler itself was successful. This means that debugging needs to extend beyond the Node.js MCP server's readResource handler and delve into the reliability and correctness of the Unity-side data retrieval and the WebSocket bridge communication. The "no side effects" guideline for resources primarily refers to external state changes, not the internal computation required to *retrieve* the data itself.

| MCP Spec Version / Release Date | Key Changes Relevant to Resources | Potential Impact on Resources |
| :---- | :---- | :---- |
| **2024-11-05** (Initial) | Introduction of core Resource concept. | Defines fundamental resource behavior. |
| **2025-03-26** | Comprehensive OAuth 2.1 authorization framework. Replaced SSE with Streamable HTTP. Added JSON-RPC batching (later removed). | New security requirements for resource access. Transport changes might affect connectivity. Batching could impact how multiple resources are requested (though now removed). |
| **2025-06-18** (Most Recent) | **Removed JSON-RPC batching.** Classified MCP servers as OAuth Resource Servers (requiring Resource Indicators). **Added support for resource links in tool call results.** | Elimination of batching simplifies protocol but might require client adjustments if previously relied upon. Enhanced security measures could block access if not properly configured. Resource links in tool results provide new ways for tools to reference resources, impacting expected consumption patterns.17 |

*Table 2: Key MCP SDK Version Changes Affecting Resource Handling*

## **5\. Resource vs. Tool Comparison: Why Tools Work, Resources Don't**

The observation that MCP tools function perfectly while resources are discovered but inaccessible is a strong indicator that the root cause lies in the fundamental distinctions between how these two primitives are implemented and utilized within the MCP ecosystem. Understanding these differences is crucial for effective debugging.

### **Key Differences in Implementation**

| Feature | Resource Characteristics | Tool Characteristics |
| :---- | :---- | :---- |
| **Primary Purpose** | Expose data; read-only access.3 | Perform actions, execute computations, can have side effects.3 |
| **Invocation Initiator** | Typically the **host application** (client).7 | Typically the **LLM (model)**.3 |
| **Invocation Mechanism** | Accessed via URIs (e.g., unity://gameobject/123) using a resources/read method.5 | Invoked by their defined name (e.g., calculate\_bmi) with explicit parameters.3 |
| **Registration Syntax** | Registered with a URI pattern (e.g., @mcp.resource("uri\_pattern")).5 | Registered by name (e.g., @mcp.tool()).5 |
| **Parameter Handling** | Parameters are extracted from the URI path by the server's routing logic (e.g., {id} from unity://gameobject/{id}).5 Relies on accurate URI parsing. | Parameters are passed as explicit, often strongly-typed, arguments to the handler function.5 Bypasses URI parsing complexity. |
| **Expected Side Effects** | None (primarily data retrieval).5 | Expected (e.g., network requests, file modifications).5 |
| **Typical Return Type** | Content (e.g., text, JSON, image) along with its MIME type.5 | Computation results, structured data, or ResourceLink objects.5 |
| **Error Handling Focus** | URI validation, internal handler errors (e.g., data lookup failures). | Business logic errors, external API failures, permission denied. |

*Table 3: MCP Resource vs. Tool Feature Comparison*

### **Identifying Root Causes for Discrepancy**

The discrepancy between working tools and inaccessible resources can often be attributed to several key areas:

* **Client-Side Interpretation and Legacy Behavior:** One of the most significant factors contributing to resources failing while tools work may stem from how the client (VS Code's MCP panel) internally handles resources. As previously noted, Cursor, a client application that likely shares architectural similarities with VS Code's MCP panel, experienced issues where its earlier implementations "implemented resources as just tools," leading to performance and scalability problems when a server exposed numerous resources.16 If VS Code's MCP panel carries this legacy or employs a similar design philosophy, it might not be correctly forming  
  readResource requests, or its internal logic for displaying and accessing resources might differ fundamentally from how it handles tools. This could result in the server correctly advertising a resource, but the client's subsequent access request being malformed or misinterpreted due to this "tool-ification," ultimately leading to the inaccessibility. This represents a client-side design limitation or bug that manifests as a server-side "inaccessible" symptom.  
* **Fragility of URI Parsing vs. Robustness of Explicit Parameters:** The difference in how parameters are handled for resources versus tools introduces a significant point of failure. Tools receive explicit, often strongly-typed, parameters directly as function arguments.5 The SDK typically manages the serialization, deserialization, and validation of these parameters in a robust manner. Conversely, resources, especially parameterized ones like  
  unity://gameobject/{id}, rely on string parsing and pattern matching to extract parameters from the URI.5 This string-based parsing is inherently more susceptible to subtle errors. A single incorrect character, a case mismatch (e.g.,  
  gameobject vs. GameObject), an unexpected URL encoding, or a deviation from the exact URI template can prevent the server's router from correctly identifying the resource handler or extracting the parameters. The Invalid Parameters (-32602) error is a direct symptom of this fragility.12 Tools bypass this complex URI parsing layer, making their parameter handling more direct and less prone to such subtle matching errors. This fundamental difference in parameter acquisition is a direct causal factor for why resources might fail where tools succeed. Therefore, a meticulous verification of the exact URI format sent by VS Code for resource access and ensuring the  
  readResource handler's parameter extraction logic is robust and either tolerant or strictly validated against potential client-side variations is crucial.  
* **Error Handling and Expectations:** Tool handlers are often designed with more comprehensive error handling mechanisms (e.g., try/catch blocks) because they are expected to perform complex operations that may involve external dependencies and potential failure points. Resource handlers, sometimes mistakenly assumed to be simpler "GET" operations, might have less defensive coding. An uncaught exception within a readResource handler (e.g., due to a failure in the Unity bridge) could lead to an Internal Error (-32603) or a silent failure, resulting in the "not accessible" symptom from the client's perspective.14  
* **Permission Model Differences:** While both resources and tools are subject to security considerations, the specific authorization checks applied to them might differ. Tools might be explicitly allowed by client-side flags (e.g., \--allowedTools in the Claude Code SDK) 21, whereas resource access might rely more heavily on URI-based validation, path permissions 13, or internal server policies.12 A subtle misconfiguration in resource-specific access controls could inadvertently block legitimate access attempts.  
* **Implicit Client Expectations:** The LLM client (VS Code) might have different implicit expectations for resources versus tools regarding retry logic, timeout behavior, or how errors are surfaced in the user interface. These unstated expectations can contribute to the perceived inaccessibility of resources.

## **6\. Unity-Specific MCP Implementations and Considerations**

Integrating MCP with Unity introduces unique considerations, particularly concerning the WebSocket bridge pattern and the handling of Unity-specific data types and URI patterns. Understanding existing implementations and potential pitfalls in this domain is crucial for debugging.

### **Other Unity \+ MCP Integration Projects**

The landscape of Unity and MCP integration is growing, with several open-source initiatives aiming to bridge the Unity Editor with LLMs.

* **Unity MCP:** This open-source project is specifically designed to facilitate seamless communication between the Unity Editor and LLMs like Claude Desktop.24 It functions as a robust communication bridge, enabling the automation of workflows and programmatic control over Unity assets and the editor itself.24 Key features include bidirectional communication, comprehensive asset management (e.g., creating, importing, and manipulating assets and prefabs), scene control (e.g., opening, saving, modifying scenes, and manipulating GameObjects), material editing, C\# script integration, and general editor automation tasks.24  
* **YetAnotherUnityMcp:** Another notable implementation, this project also leverages WebSockets for real-time communication between Unity and AI agents. It extends capabilities to allow AI to inspect Unity scenes, execute C\# code directly, and even capture screenshots with AI-driven parameters.24  
* **Official MCP C\# SDK:** The availability of an official C\# SDK for MCP further supports this ecosystem, enabling.NET applications (including Unity, which primarily uses C\#) to implement both MCP clients and servers.5 This indicates a strategic move towards broader and more robust Unity-MCP integration.

### **WebSocket Bridge Patterns (Unity Editor \<-\> Node.js MCP Servers)**

WebSocket is the established and preferred protocol for facilitating real-time, bidirectional communication between the Unity Editor (acting as a C\# client) and a Node.js MCP server.27 Its full-duplex capabilities over a single TCP connection make it highly efficient for continuous message exchange, which is ideal for interactive AI-driven workflows.27

While Node.js versions 21 and newer offer a native WebSocket *client* implementation, creating a WebSocket *server* in Node.js typically still necessitates the use of external libraries such as ws or socket.io.27 A common architectural pattern involves a Node.js server (e.g., built with the

ws module) and a Unity C\# client (often utilizing libraries like WebSocketSharp-netstandard or other NuGet packages).28 Communication across this bridge is predominantly in JSON format, ensuring structured data exchange.29 In this setup, the Node.js server acts as an intermediary, translating incoming MCP requests into WebSocket messages for the Unity Editor and converting Unity's responses back into the appropriate MCP response format.

### **Unity-Specific Considerations for Resource URI Patterns and Data Serialization**

The custom unity:// URI scheme used for resources like unity://gameobject/{id} and unity://packages introduces specific requirements for implementation and data handling.

* **Resource URI Patterns:** The Node.js MCP server's readResource handler must be meticulously designed to correctly parse the unity:// scheme and its subsequent path components (e.g., gameobject, packages). For parameterized URIs like unity://gameobject/{id}, the value of the {id} placeholder needs to be accurately extracted and then used to query the Unity Editor. This id could semantically represent various Unity identifiers, such as a Unity InstanceID (an integer), a GUID (a string), a GameObject name, or a custom identifier defined within the Unity project. Crucially, the Node.js server's handler and the Unity C\# code must have a precise agreement on the semantic meaning and expected format of this id to ensure successful data retrieval.  
* **Data Serialization over WebSocket:** UnityMCP projects commonly employ JSON as the message format for WebSocket communication.29 Unity itself provides built-in serialization capabilities for both JSON and binary formats.30 When a  
  readResource request for unity://gameobject/{id} is processed, the Unity Editor will need to retrieve the relevant GameObject's data (e.g., its name, position, components, hierarchy). This complex C\# object structure must then be reliably serialized into a JSON format that can be transmitted efficiently over the WebSocket bridge to the Node.js MCP server.  
  * **Potential Challenges:**  
    * **Complex Object Serialization:** Unity GameObjects and their components can be deeply nested and contain intricate data types. Relying solely on Unity's default serialization might not always produce a simple, flat JSON structure suitable for MCP. Custom serialization logic might be necessary on the Unity C\# side to transform these complex objects into a format that the Node.js server can easily parse. If this serialization is incomplete or malformed, the Node.js server might receive invalid JSON or incomplete data, leading to perceived inaccessibility.28  
    * **Large Payloads:** Data for complex GameObjects or entire scenes can be substantial. This could lead to WebSocket message size limits, performance bottlenecks, or timeouts if not handled efficiently. Strategies like streaming or data chunking might be necessary, though they are not explicitly detailed for MCP resources in the provided context.  
    * **References vs. Values:** When serializing Unity objects, a critical design decision involves whether to include full object data or merely references (e.g., InstanceIDs or GUIDs). If the client expects comprehensive data but only references are transmitted, the resource will appear as "not accessible" from the client's perspective.

The WebSocket bridge connecting the Node.js MCP server to the Unity Editor represents a critical juncture where data can be lost, corrupted, or misinterpreted, directly contributing to the "not accessible" symptom. The readResource handler on the Node.js MCP server, when processing a request for unity://gameobject/{id}, must communicate with the Unity Editor. This communication occurs over the WebSocket bridge. Failures in accessing the resource could originate deep within this bridge or on the Unity side. For instance, the C\# code in Unity might fail to locate or retrieve the GameObject corresponding to the provided {id} due to an incorrect lookup method, the object being inactive, or an unhandled exception within the Unity script. Even if Unity successfully finds the object, its C\# code might fail to correctly serialize the GameObject's data into valid JSON before sending it over the WebSocket.28 Furthermore, the WebSocket connection itself could experience intermittent issues, data corruption, or be unable to handle large data payloads, leading to incomplete or dropped messages. Finally, the Node.js server might receive malformed JSON from Unity and fail to parse it, resulting in an internal error before it can formulate a proper MCP response. From VS Code's perspective, any of these failures would manifest as the resource being "not accessible," even if the MCP protocol itself is sound up to the point of the

readResource handler. Therefore, debugging efforts must extend into the Unity Editor's C\# scripts, their logging, and the WebSocket communication layer between Unity and Node.js.

Another crucial, often overlooked, aspect is the semantic interpretation of the {id} parameter in unity://gameobject/{id}. Unity GameObjects can be identified in various ways: by their InstanceID (an integer), by a GUID (a string), or by their name (a string). When VS Code queries unity://gameobject/123, it is imperative to determine whether 123 is intended to be an InstanceID, a name, or some other custom identifier. The Node.js readResource handler extracts {id} as a string. The Unity C\# code that receives this string must then correctly *interpret* it to locate the corresponding GameObject. If the client sends unity://gameobject/MyCube (a name) but the Unity C\# code attempts GameObject.FindObjectFromInstanceID(int.Parse(id)) (expecting an integer InstanceID), the lookup will inevitably fail, resulting in no data being returned. Similarly, case sensitivity in GameObject names (e.g., myCube vs. MyCube) can cause lookup failures. Therefore, the user needs to meticulously verify the exact format of the {id} being transmitted by VS Code and ensure that the Unity-side C\# logic for resolving this id to a GameObject is robust and matches the expected identifier type and format. Explicit logging of the id received by the Unity C\# code is an essential step in this verification process.

## **7\. Systematic Debugging Checklist for "Resources Advertised but Not Accessible"**

Resolving the issue of MCP resources being discovered but not accessible requires a systematic debugging approach that spans the entire communication stack, from the VS Code client to the Node.js MCP server and the Unity Editor via the WebSocket bridge.

### **Initial Server & Client Health Checks**

Before diving into protocol specifics, fundamental checks can quickly identify common environmental issues.

* **Verify Server Process:** Ensure the Node.js MCP server process is actively running and stable. Examine its console output for any immediate crashes, startup errors, or unhandled exceptions that might prevent it from fully initializing.20 Running the server manually in a dedicated terminal can provide direct error messages and confirm its stable operation.20  
* **Address Port Conflicts:** If the MCP server utilizes a WebSocket transport, confirm that the specified port (e.g., 8080\) is not already in use by another application. System utilities like netstat (on Windows/Linux/macOS) or lsof (on Linux/macOS) can help identify conflicting processes, which can then be terminated or the server configured to use an alternative port.20  
* **Check Network/Firewalls:** For any networked transport, verify that local system firewalls or network firewalls are not blocking the connection on the designated port. If the server is not running on localhost, ensure it is reachable from the client's network location.20  
* **Confirm VS Code MCP Panel Display:** Re-confirm that all 8 resources, along with their proper URIs, are still correctly listed within the VS Code MCP panel. This step validates that the listResources mechanism is fully functional and that the server is successfully advertising its capabilities to the client.

### **MCP Protocol-Level Debugging**

Once basic health checks are passed, focus shifts to the MCP protocol itself.

* **Inspect MCP Logs in VS Code:** This is the primary and most direct source of client-side MCP activity. Access the Output panel in VS Code (typically Ctrl+Shift+U or Cmd+Shift+U) and select "MCP Logs" from the dropdown menu.1 Scrutinize these logs for any connection errors, authentication failures, server crashes, or, most importantly, messages related to  
  resources/read attempts. Pay close attention to the raw JSON-RPC requests sent by VS Code and the responses received from the server.1  
* **Use mcp-cli or Similar Tools:** If available, a command-line utility like mcp-cli can be invaluable. This tool allows direct interaction with the MCP server, bypassing the VS Code client. Use it to listResources and then attempt to readResource for one of the problematic URIs. This helps isolate whether the issue is specific to the VS Code client's interaction or a more general server-side problem.20  
* **Verify initialize Handshake & Protocol Version:** Examine the server and host logs for the initialize request sent by the client and the corresponding response from the server. Crucially, verify that the protocolVersion fields in both messages are compatible. Mismatches in protocol versions can lead to subtle communication failures or unexpected behavior.20

### **Resource-Specific Validation**

Given that resources are discovered but not accessible, a deep dive into resource handling is required.

* **URI Validation (Most Likely Culprit):**  
  * **Exact Pattern Match:** Verify that the URI requested by VS Code (as observed in MCP logs or mcp-cli output) *exactly* matches the URI pattern registered with the server.resource() method on your Node.js server. Even minor discrepancies can prevent a successful match.  
  * **Parameter Extraction:** Confirm that the {id} parameter (from unity://gameobject/{id}) is being correctly extracted by the SDK's routing mechanism and accurately passed to your readResource handler function. Add logging statements at the absolute *beginning* of your readResource handler to log the raw incoming URI and the extracted parameter value.  
  * **Casing Sensitivity:** For unity:// schemes, especially if the id or other path components map to case-sensitive names or paths within Unity or the underlying file system, ensure consistent casing between the client's request and the server's expectation.13  
  * **URL Encoding/Decoding:** Ensure that any special characters present in the id are correctly URL-encoded by the client and properly decoded by your server-side handler before use.  
* **Handler Invocation:**  
  * **Entry Point Logging:** Insert explicit console.log or ctx.debug statements as the very *first line* within your readResource handler function. If these logs *do not appear* in your server's console or the MCP logs, it is a strong indication that the URI pattern matching or the SDK's routing is failing *before* your handler is even invoked.  
  * **Parameter Logging:** Inside the handler, log the exact values of all parameters received (e.g., the extracted id). This confirms if the parameters are being passed correctly after extraction from the URI.  
* **Internal Handler Logic (Node.js & Unity):**  
  * **Step-by-step Debugging:** Utilize a Node.js debugger to step through the execution of your readResource handler. Trace the flow, paying particular attention to any points where it interacts with the WebSocket bridge to Unity.  
  * **WebSocket Bridge Communication:** Implement extensive logging within your Node.js WebSocket client/server code to capture the precise messages being sent to and received from the Unity Editor.  
  * **Unity-Side Logic:** Within the Unity Editor, debug the C\# code responsible for resolving the gameobject/{id} request to an actual GameObject. Verify if the GameObject is successfully found and if any errors or exceptions occur within this C\# logic. Crucially, confirm the semantic meaning of {id}: if VS Code sends a name, but Unity expects an InstanceID, the lookup will fail.  
  * **Data Retrieval & Serialization:** Confirm that the Unity-side logic successfully retrieves the necessary data from the GameObject. Then, ensure this data is correctly serialized into valid JSON before being transmitted back over the WebSocket bridge to the Node.js server. Malformed JSON can lead to parsing errors on the Node.js side, resulting in an inaccessible resource.28  
* **Permissions & Access Control:**  
  * Review any server-side access control lists (ACLs), authentication requirements, or authorization logic that might be implicitly or explicitly blocking resource access.12  
  * Check for specific client-side permissions required by VS Code for accessing MCP resources, such as \--allowedTools flags if resources are being treated as tools by the client.21  
  * If the server implements path-based security, investigate for "Access denied \- path outside allowed directories" errors, particularly on Windows where file path case sensitivity can be an issue.13

### **Common Error Patterns and Solutions**

Understanding common MCP error codes and their typical causes can significantly expedite the debugging process.

| MCP Error Code | Standard Error Message | Common Causes | Debugging Steps |
| :---- | :---- | :---- | :---- |
| **\-32700** | Parse error | Invalid JSON syntax in message sent/received. | Validate JSON serialization/deserialization on both client and server. Use JSON validators.14 |
| **\-32600** | Invalid Request | Valid JSON, but message violates MCP protocol structure (e.g., missing jsonrpc field). | Verify fundamental JSON-RPC request/response structure.14 |
| **\-32601** | Method Not Found | Requested MCP method (resources/read) or URI pattern does not match any registered server handler. | Double-check URI registration patterns against actual requests. Ensure consistency in URI schemes (e.g., unity://).14 |
| **\-32602** | Invalid Parameters | Method exists, but arguments (extracted from URI) do not meet handler requirements (e.g., type mismatch, missing fields). Most common for resources.14 | Implement strict validation of input parameters in handler. Ensure client sends parameters in expected format. Review URI pattern definition and extraction logic.14 |
| **\-32603** | Internal Error | Uncaught exception or runtime error within server's readResource handler (e.g., Unity bridge failure, data retrieval error).14 | Implement robust try/catch blocks in handler. Log full stack traces. Debug internal logic, especially calls to Unity via WebSocket.14 |
| **Access Denied** | (Varies, e.g., Path Outside Allowed Directories) | Server-side security policies blocking access (e.g., path validation, missing authentication, insufficient permissions).13 | Review server's security configuration, allowed directories, authentication/authorization. Check for case sensitivity in paths on Windows.12 |
| **No Response/Timeout** | (Client-side timeout error) | readResource handler is stuck, waiting for external dependency (Unity bridge) or taking too long. | Implement timeouts for all external calls. Use asynchronous operations correctly. Add progress reporting for long tasks.20 |

*Table 4: Common MCP Error Codes and Debugging Steps*

### **Proper Logging and Debugging Techniques for MCP Resource Access**

Effective logging is paramount for diagnosing complex distributed systems like MCP integrations.

* **Context-Aware Logging:** Leverage the Context object (ctx) available within your readResource handlers to send structured log messages (debug, info, warning, error) directly back to the MCP client, which will appear in VS Code's "MCP Logs" panel.15 For example:  
  * await ctx.debug(f"Entering readResource handler for URI: {uri}")  
  * await ctx.info(f"Extracted GameObject ID: {id}")  
  * await ctx.error(f"Failed to retrieve GameObject data from Unity: {error.message}", { stack\_trace: error.stack })  
* **Structured Logging:** Beyond ctx logging, implement structured logging in your Node.js server for persistent records. Include timestamps, request IDs (ctx.request\_id), and relevant data in logs for easier analysis and correlation across requests.31  
* **External Logging:** For production environments, consider directing logs to files or a centralized logging system (e.g., Prometheus, Datadog mentioned in 31) separate from  
  stdio to ensure comprehensive historical data.  
* **Unity-Side Logging:** Ensure your Unity C\# scripts incorporate robust Debug.Log statements or a custom logging system. This is crucial for tracing their execution, especially when handling incoming WebSocket messages and performing GameObject lookups or data serialization.  
* **MCP Inspector:** The MCP Inspector tool 5 provides a direct interface to test resource access and observe the raw JSON-RPC messages and responses exchanged with your server. This tool is invaluable for validating server behavior independently of the VS Code client.

## **8\. Conclusion and Recommendations**

The challenge of MCP resources being discovered but inaccessible, while tools function correctly, typically stems from a confluence of factors related to URI parameter handling, client-side resource interpretation, and the reliability of the underlying Unity-Node.js WebSocket bridge. The successful discovery of resources indicates that the server is correctly advertising its capabilities. However, the failure during access points to issues within the readResource method's execution path.

The analysis highlights that URI parameter parsing is a particularly fragile component. Subtle mismatches in type, format, casing, or encoding between the client's request and the server's handler can lead to Invalid Parameters or Method Not Found errors, even if the URI template is correctly advertised. Furthermore, historical behaviors of MCP clients, such as Cursor's past tendency to "tool-ify" resources, suggest that the VS Code MCP panel might be attempting to access resources in a way that is incompatible with the server's readResource implementation, creating a fundamental communication breakdown. Finally, the Unity-Node.js WebSocket bridge represents a critical "black box" where failures in Unity's C\# logic (e.g., GameObject lookup), data serialization to JSON, or the WebSocket transmission itself can silently lead to resource inaccessibility from the client's perspective.

To resolve this issue and ensure robust MCP resource implementations in Unity-Node.js integrations, the following recommendations are prioritized:

1. **Prioritize URI Parameter Validation and Logging:**  
   * **Action:** Implement meticulous validation of all parameters extracted from resource URIs within your Node.js readResource handlers. This includes type checking, format validation, and consistent casing.  
   * **Action:** Add extensive ctx.debug and ctx.info logging at the very beginning of your readResource handlers to log the raw incoming URI and the exact values of all extracted parameters. This will immediately reveal if the routing or parameter parsing is failing before your core logic executes.  
   * **Action:** Ensure proper URL encoding by the client and decoding by the server for any dynamic URI components.  
2. **Deep Dive into the Unity-Node.js WebSocket Bridge:**  
   * **Action:** Implement detailed logging within your Node.js WebSocket client/server code to capture all messages sent to and received from Unity.  
   * **Action:** Enhance logging within your Unity C\# scripts, especially for the logic that resolves unity://gameobject/{id} to an actual GameObject. Log the incoming id, the result of the GameObject lookup, and any errors or exceptions that occur during this process.  
   * **Action:** Verify the semantic interpretation of the {id} parameter. Ensure that the type of identifier sent by VS Code (e.g., name, InstanceID, GUID) matches what your Unity C\# code expects for GameObject lookup.  
   * **Action:** Scrutinize the Unity C\# data serialization logic. Confirm that complex GameObject data is correctly converted into valid JSON before being sent over the WebSocket. Test with various GameObject complexities to identify potential serialization failures or large payload issues.  
3. **Investigate Client-Side Compatibility and Versioning:**  
   * **Action:** Identify the exact version of the VS Code MCP panel/extension being used. Search for known issues, changelogs, or community discussions related to resource handling in that specific version, particularly any historical tendencies to treat resources as tools.  
   * **Action:** Verify the MCP protocol version negotiation between your Node.js server and the VS Code client through the initialize handshake logs. Ensure compatibility, considering the rapid evolution of the MCP specification. If a mismatch is identified, consider updating your @modelcontextprotocol/sdk to a compatible version.  
4. **Review Server-Side Access Control and Error Handling:**  
   * **Action:** Examine any custom security or access control logic on your Node.js MCP server that might inadvertently block resource access based on URI patterns or client credentials. Pay attention to path-based security, especially on Windows where case sensitivity can cause Access Denied errors.  
   * **Action:** Implement robust try/catch blocks around all critical logic within your readResource handlers, particularly around external calls to the Unity bridge. Ensure that caught exceptions are logged with full stack traces and translated into appropriate JSON-RPC error responses (e.g., Internal Error \-32603) to provide clear feedback to the client.

By systematically addressing these areas, developers can pinpoint the exact cause of resource inaccessibility, move beyond mere discovery, and establish a robust, reliable MCP integration between Node.js and Unity.

#### **Works cited**

1. Model Context Protocol (MCP) \- Cursor, accessed July 7, 2025, [https://docs.cursor.com/context/mcp](https://docs.cursor.com/context/mcp)  
2. The Model Context Protocol (MCP) — A Complete Tutorial | by Dr. Nimrita Koul \- Medium, accessed July 7, 2025, [https://medium.com/@nimritakoul01/the-model-context-protocol-mcp-a-complete-tutorial-a3abe8a7f4ef](https://medium.com/@nimritakoul01/the-model-context-protocol-mcp-a-complete-tutorial-a3abe8a7f4ef)  
3. MCP vs Toolformer: Two Approaches to Enabling Tool Capabilities in LLMs \- MonsterAPI, accessed July 7, 2025, [https://blog.monsterapi.ai/mcp-vs-toolformer/](https://blog.monsterapi.ai/mcp-vs-toolformer/)  
4. The Model Context Protocol: A Game-Changer for AI Integration | by Jai Lad | Medium, accessed July 7, 2025, [https://medium.com/@lad.jai/the-model-context-protocol-a-game-changer-for-ai-integration-b1d3c6a4cdd4](https://medium.com/@lad.jai/the-model-context-protocol-a-game-changer-for-ai-integration-b1d3c6a4cdd4)  
5. AI-App/ModelContextProtocol.Python-SDK: The official ... \- GitHub, accessed July 7, 2025, [https://github.com/AI-App/ModelContextProtocol.Python-SDK](https://github.com/AI-App/ModelContextProtocol.Python-SDK)  
6. The official Go SDK for Model Context Protocol servers and clients. Maintained in collaboration with Google. \- GitHub, accessed July 7, 2025, [https://github.com/modelcontextprotocol/go-sdk](https://github.com/modelcontextprotocol/go-sdk)  
7. mcp-course/README · Early Tool vs Resource Examples \- Hugging Face, accessed July 7, 2025, [https://huggingface.co/spaces/mcp-course/README/discussions/2](https://huggingface.co/spaces/mcp-course/README/discussions/2)  
8. Building MCP Servers: Part 2 — Extending Resources with Resource Templates | by Christopher Strolia-Davis | Medium, accessed July 7, 2025, [https://medium.com/@cstroliadavis/building-mcp-servers-315917582ad1](https://medium.com/@cstroliadavis/building-mcp-servers-315917582ad1)  
9. What are MCP Resources? \- Speakeasy, accessed July 7, 2025, [https://www.speakeasy.com/mcp/resources](https://www.speakeasy.com/mcp/resources)  
10. @modelcontextprotocol/sdk \- npm, accessed July 7, 2025, [https://www.npmjs.com/package/@modelcontextprotocol/sdk](https://www.npmjs.com/package/@modelcontextprotocol/sdk)  
11. mcp-swift-sdk \- Swift Package Index, accessed July 7, 2025, [https://swiftpackageindex.com/modelcontextprotocol/swift-sdk](https://swiftpackageindex.com/modelcontextprotocol/swift-sdk)  
12. Best Practices for Secure MCP Resource Access (Client & Server ..., accessed July 7, 2025, [https://medium.com/@yany.dong/best-practices-for-secure-mcp-resource-access-client-server-25a9f98055d4](https://medium.com/@yany.dong/best-practices-for-secure-mcp-resource-access-client-server-25a9f98055d4)  
13. Filesystem MCP Server (Windows/NPX): Flawed path validation ..., accessed July 7, 2025, [https://github.com/modelcontextprotocol/servers/issues/1838](https://github.com/modelcontextprotocol/servers/issues/1838)  
14. MCP Error Codes | mcpevals.io, accessed July 7, 2025, [https://www.mcpevals.io/blog/mcp-error-codes](https://www.mcpevals.io/blog/mcp-error-codes)  
15. MCP Context \- FastMCP, accessed July 7, 2025, [https://gofastmcp.com/servers/context](https://gofastmcp.com/servers/context)  
16. MCP resources not working in 0.46.9 \- Page 2 \- Bug Reports ..., accessed July 7, 2025, [https://forum.cursor.com/t/mcp-resources-not-working-in-0-46-9/59960?page=2](https://forum.cursor.com/t/mcp-resources-not-working-in-0-46-9/59960?page=2)  
17. MCP release notes \- Speakeasy, accessed July 7, 2025, [https://www.speakeasy.com/mcp/release-notes](https://www.speakeasy.com/mcp/release-notes)  
18. Key Changes \- Model Context Protocol, accessed July 7, 2025, [https://modelcontextprotocol.io/specification/2025-03-26/changelog](https://modelcontextprotocol.io/specification/2025-03-26/changelog)  
19. Key Changes \- Model Context Protocol, accessed July 7, 2025, [https://modelcontextprotocol.io/specification/2025-06-18/changelog](https://modelcontextprotocol.io/specification/2025-06-18/changelog)  
20. Debugging Model Context Protocol (MCP) Servers: Tips and Best Practices | mcpevals.io, accessed July 7, 2025, [https://www.mcpevals.io/blog/debugging-mcp-servers-tips-and-best-practices](https://www.mcpevals.io/blog/debugging-mcp-servers-tips-and-best-practices)  
21. Claude Code SDK \- Anthropic API, accessed July 7, 2025, [https://docs.anthropic.com/en/docs/claude-code/sdk](https://docs.anthropic.com/en/docs/claude-code/sdk)  
22. Authorization \- Model Context Protocol, accessed July 7, 2025, [https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)  
23. Treat the MCP server as an OAuth resource server rather than an authorization server · Issue \#205 \- GitHub, accessed July 7, 2025, [https://github.com/modelcontextprotocol/modelcontextprotocol/issues/205](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/205)  
24. Unity MCP Server: Use AI to Control Your Unity Projects with Claude \- Apidog, accessed July 7, 2025, [https://apidog.com/blog/unity-mcp-server/](https://apidog.com/blog/unity-mcp-server/)  
25. apidog.com, accessed July 7, 2025, [https://apidog.com/blog/unity-mcp-server/\#:\~:text=Unity%20MCP%2C%20or%20Model%20Context,control%20the%20Unity%20Editor%20programmatically.](https://apidog.com/blog/unity-mcp-server/#:~:text=Unity%20MCP%2C%20or%20Model%20Context,control%20the%20Unity%20Editor%20programmatically.)  
26. modelcontextprotocol/csharp-sdk: The official C\# SDK for ... \- GitHub, accessed July 7, 2025, [https://github.com/modelcontextprotocol/csharp-sdk](https://github.com/modelcontextprotocol/csharp-sdk)  
27. Node.js WebSocket, accessed July 7, 2025, [https://nodejs.org/en/learn/getting-started/websocket](https://nodejs.org/en/learn/getting-started/websocket)  
28. WebSocket Client & Server (Unity & NodeJS) | by Ahmed Schrute ..., accessed July 7, 2025, [https://medium.com/unity-nodejs/websocket-client-server-unity-nodejs-e33604c6a006](https://medium.com/unity-nodejs/websocket-client-server-unity-nodejs-e33604c6a006)  
29. Arodoid/UnityMCP \- GitHub, accessed July 7, 2025, [https://github.com/Arodoid/UnityMCP](https://github.com/Arodoid/UnityMCP)  
30. Introduction to Unity Serialization | Serialization | 3.1.2 \- Unity \- Manual, accessed July 7, 2025, [https://docs.unity3d.com/Packages/com.unity.serialization@latest/](https://docs.unity3d.com/Packages/com.unity.serialization@latest/)  
31. How to Build an MCP Server: Setup and Management Guide, accessed July 7, 2025, [https://blog.promptlayer.com/how-to-build-mcp-server/](https://blog.promptlayer.com/how-to-build-mcp-server/)  
32. Logging \- Model Context Protocol, accessed July 7, 2025, [https://modelcontextprotocol.io/specification/2025-03-26/server/utilities/logging](https://modelcontextprotocol.io/specification/2025-03-26/server/utilities/logging)  
33. Best Practices for Handling MCP Errors in API Requests \- Arsturn, accessed July 7, 2025, [https://www.arsturn.com/blog/handling-api-requests-best-practices-for-handling-mcp-errors](https://www.arsturn.com/blog/handling-api-requests-best-practices-for-handling-mcp-errors)  
34. Model Context Protocol \- Explained\! (with Python example) \- YouTube, accessed July 7, 2025, [https://www.youtube.com/watch?v=JF14z6XO4Ho\&pp=0gcJCfwAo7VqN5tD](https://www.youtube.com/watch?v=JF14z6XO4Ho&pp=0gcJCfwAo7VqN5tD)
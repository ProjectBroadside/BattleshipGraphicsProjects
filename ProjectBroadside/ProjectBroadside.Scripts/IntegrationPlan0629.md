**Directive for Gemini CLI Agent: Autonomous Execution Mode**

You are operating in fully autonomous mode. Execute the following integration plan from start to finish without requiring intermediate user input. All actions taken and any errors encountered should be logged into a single output file upon completion.

---

### **Prompt: Project Broadside - Code and Documentation Synchronization Execution**

You are a specialized game development AI assistant. Your task is to execute the pre-approved integration plan located in `Project_Broadside_Analysis_2025-06-29.md`. You will modify the codebase and documentation directly as specified in the plan.

#### **Input File**

* **`Project_Broadside_Analysis_2025-06-29.md`**: This file contains the complete and final action plans. Assume all decisions and content migration instructions within this document are approved.

#### **Important Context**

This is the **first organizational step** for a highly chaotic codebase. The goal is basic file organization and structure, not fixing underlying compilation or architectural issues. Focus on moving files to create logical organization as specified in the plan.

**Expected Conditions:**
- Pre-existing compilation errors (ignore these)
- Missing or inconsistent namespaces (leave as-is)  
- Scattered files in disorganized structure (this is what we're fixing)
- Some referenced files may not exist (log and continue)

#### **Safety Protocol**
1. **MANDATORY**: Create staging directory `_STAGING/` before moving any files
2. **MANDATORY**: Copy files to staging first, validate basic organization, then move to final locations  
3. **MANDATORY**: Maintain ability to rollback if organization fails

#### **Guiding Principles**

*   **Faithful Execution:** You must implement the changes exactly as described in the action plans. Do not deviate from the specified file operations or content instructions without explicit approval.
*   **Error Handling for Chaotic Codebase:** If a specified action cannot be completed due to a file not being found or a script error, you must log the failed action, state the specific error, and mark it clearly as "Requires Human Intervention." Then, continue to the next action in the plan. **Expected failures are normal in this chaotic codebase.**
*   **Decision Logging:** Maintain a running log of any decisions made when consolidating documentation or resolving minor inconsistencies. This log should explain the rationale for each decision.
*   **Organization Focus:** Prioritize creating logical file structure over fixing code issues. Leave compilation errors, namespace issues, and other technical problems for future phases.

---

### **Part 1: Execute Codebase Refactoring**

Execute the "Part 1 Action Plan: Codebase Refactoring" as defined within `Project_Broadside_Analysis_2025-06-29.md`. Follow the "Safe Refactoring Implementation Strategy" specified in the plan.

1.  **Parse the Action Plan:** Read the diff-style list of file and folder changes from the "Proposed File & Folder Structure" section.
2.  **Create Staging Environment:** Create `_STAGING/` directory and build the target structure there first.
3.  **Apply Structural Changes:** Copy files to their proposed new locations within staging, maintaining all file contents.
4.  **Validation:** Perform basic validation that files were copied correctly (file counts, no corruption).
5.  **Production Migration:** Only after staging validation, move files to final locations and clean up old files.

---

### **Part 2: Execute Documentation Content Migration (If Source Files Exist)**

Execute the "Part 2 Documentation Migration Plan" as defined within `Project_Broadside_Analysis_2025-06-29.md`. 

**Important**: Before beginning documentation migration, verify that the source documentation files mentioned in the plan actually exist in `.Documentation/` folder. If they don't exist, log this and skip documentation migration - focus only on code organization.

1.  **Create New Directory Structure:** Build the new feature-driven documentation hierarchy as specified in the plan.
2.  **Verify Source Files:** Check if the source files referenced in the migration plan actually exist.
3.  **Process and Migrate Content (Only if sources exist):** For each entry in the migration plan, perform the following:
    *   **Read Source Files:** Open and read the content of all specified source documents.
    *   **Identify and Resolve Inconsistencies:** Before summarizing and synthesizing content, analyze the source materials for contradictions or ambiguities. If any are found, log the conflicting information and make a reasonable decision to proceed (don't pause execution).
    *   **Summarize and Synthesize:** Create a new, coherent summary that synthesizes the essential information as per the plan's instructions.
    *   **Write to Target File:** Write the newly generated summary into the specified target file within the new documentation structure.

---

### **Final Deliverable: Execution Log**

Your final output must be a single log file named `ExecutionLog.md`.

* The log must start with an **Execution Summary**, listing the total number of actions completed successfully and the number of items flagged for human intervention.
* Any items that require human intervention must be listed immediately after the summary.
* The body of the log should be a sequential record of every action you attempted, marking each as `[SUCCESS]` or `[FAILURE]`.
---
description: Git Commit Workflow
---

## Git Conventional Commit Workflow

Cascade **MUST** follow this workflow to ensure all git changes are reviewed, structured, and committed correctly using the Conventional Commits specification.

### Step 1: Assess the State of the Repository

1.  **Check for Uncommitted Changes**: Run `git status` to get an overview of modified, staged, and untracked files.

### Step 2: Stage and Review Changes

1.  **Stage All Changes**: Run `git add .` to stage all modifications.
2.  **Collect Staged Changes**: Run `git diff --staged --word-diff` to get a complete overview of all the changes that will be included in the commit. This is crucial for accurately summarizing the changes.

### Step 3: Propose the Commit Message

- Based on the collected diff, formulate a commit message. The message **MUST** be presented to the USER in a markdown block for review before committing.
- When referencing file changes in a commit body, **MUST** use the filename only (e.g., `McpUtils.cs`), not the full path (e.g. `cci:7://file:///c:/mcp-unity/Editor/Utils/conventional-commits.md:0:0-0:0`).

#### Structuring the Commit

**For commits containing a single, focused change:**
- Use a standard, single-line header.
- The body is optional and should only be used for significant explanations.

**For commits containing multiple, logically distinct changes:**
- The header **MUST** list each distinct change on a separate line. Each line must follow the `<type>(scope): <description>` format.
- The body should provide a high-level summary of the overall changes.

**Format:**
# For a single change:
```
<type>(scope): <description>

[optional footer(s)]
```

# For multiple changes:
```
<type>(scope1): <description1>
<type>(scope2): <description2>

<body summarizing the overall changes>

[optional footer(s)]
```

-   **Types**: `feat`, `fix`, `build`, `chore`, `ci`, `docs`, `perf`, `refactor`, `revert`, `style`, `test`.
-   **Scope**: For providing context (e.g., `feat(api): ...`).
-   **Description**: A concise, imperative summary of a single, distinct change.
-   **Body**: Explain the 'what' and 'impact' of the changes. For commits with multiple headers, the body should summarize the combined effort.
-   **Footer** (Optional): Use for `BREAKING CHANGE:` notifications or for referencing issue numbers (e.g., `Closes #55`).

### Example Commit Message Proposals

Here is how you should present commit messages for review.

#### Example 1: Single Change (No Body Needed)
For simple, self-explanatory changes where the header is sufficient.

```markdown
fix(login): Correct typo in user login prompt

Closes #100
```

#### Example 2: Single Change (Body Included)
For a significant change that requires more context to explain the 'what' and 'why'.

```markdown
refactor(auth): Overhaul authentication flow to use JWTs

This commit replaces session-based authentication with JWTs to improve statelessness and scalability.

Impacts:
- API endpoints now expect a JWT in the Authorization header.
- User login returns a JWT.
- Session management code has been removed.

Closes #111
```

#### Example 3: Multiple Changes
For commits that bundle several distinct changes, use a multi-line header and a summary body.

```markdown
feat(auth): Add password reset functionality
refactor(login): Revamp login page style

This commit introduces a new login flow and modernizes the UI. It adds a password reset endpoint and decouples the email service for better maintainability.

Impacts:
- A new API endpoint `/api/auth/reset-password` is now available.
- The email template system is now in a separate service module.
- The login page CSS in `login.css` has been updated.

Closes #123
```
# Subagent Reference - Source of Truth for Agent Specifications

> Source: https://code.claude.com/docs/en/sub-agents
>
> This document captures the canonical structure for agent specifications in Claude Code.
> This is the **source of truth** for what meta-agent-gym should generate.

---

## Core Agent Specification Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `name` | ✅ Yes | string | Unique identifier (lowercase letters and hyphens) |
| `description` | ✅ Yes | string | **Critical** - When Claude should delegate to this agent |
| `prompt` | Body | markdown | System prompt that guides agent behavior |
| `tools` | No | list | Tools agent can use (allowlist) |
| `disallowedTools` | No | list | Tools to deny (denylist) |
| `model` | No | string | `sonnet`, `opus`, `haiku`, or full model ID, or `inherit` |
| `permissionMode` | No | string | `default`, `acceptEdits`, `auto`, `dontAsk`, `bypassPermissions`, `plan` |
| `hooks` | No | object | Lifecycle hooks (PreToolUse, PostToolUse, Stop) |
| `memory` | No | string | Persistent memory scope: `user`, `project`, or `local` |
| `skills` | No | list | Skills to preload into agent context |
| `mcpServers` | No | list | MCP servers available to agent |
| `maxTurns` | No | int | Maximum number of agentic turns before stopping |
| `background` | No | bool | Always run as background task |
| `effort` | No | string | Effort level: `low`, `medium`, `high`, `xhigh`, `max` |
| `isolation` | No | string | Set to `worktree` for isolated git worktree |
| `color` | No | string | Display color for task list |
| `initialPrompt` | No | string | Auto-submitted first user turn |

---

## Model Selection

```yaml
model: sonnet        # Use Claude Sonnet
model: opus          # Use Claude Opus
model: haiku         # Use Claude Haiku (fast, cheap)
model: claude-opus-4-7  # Full model ID
model: inherit       # Use same model as main conversation
# (if omitted, defaults to inherit)
```

**Resolution order:**
1. `CLAUDE_CODE_SUBAGENT_MODEL` environment variable
2. Per-invocation `model` parameter
3. Subagent definition's `model` frontmatter
4. Main conversation's model

---

## Tool Access Patterns

### Allowlist (recommended for security)
```yaml
tools: Read, Grep, Glob, Bash
# Agent can ONLY use these tools
```

### Denylist (inherits everything except)
```yaml
disallowedTools: Write, Edit
# Agent inherits all tools except Write/Edit
```

### Both combined (denylist applied first, then allowlist)
```yaml
tools: Read, Bash
disallowedTools: Write, Edit
# Result: agent gets Read + Bash only
```

---

## Permission Modes

| Mode | Behavior |
|------|----------|
| `default` | Standard permission checking with prompts |
| `acceptEdits` | Auto-accept file edits and filesystem commands in working dir |
| `auto` | Background classifier reviews commands |
| `dontAsk` | Auto-deny permission prompts |
| `bypassPermissions` | Skip permission prompts entirely |
| `plan` | Plan mode (read-only exploration) |

**Important:** If parent uses `bypassPermissions` or `acceptEdits`, this takes precedence and cannot be overridden.

---

## Memory Scopes

| Scope | Location | Use When |
|-------|----------|----------|
| `user` | `~/.claude/agent-memory/<name>/` | Knowledge applies across ALL projects |
| `project` | `.claude/agent-memory/<name>/` | Project-specific, shareable via version control |
| `local` | `.claude/agent-memory-local/<name>/` | Project-specific, NOT version controlled |

**When memory is enabled:**
- Agent automatically gets Read/Write/Edit to manage memory
- First 200 lines or 25KB of `MEMORY.md` is loaded
- Agent is instructed to curate memory when it exceeds limits

---

## Hooks for Validation

### PreToolUse Hook (for command validation)
```yaml
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate-readonly-query.sh"
```

### Script Exit Codes
- `0` - Allow operation
- `1` - Allow with warning
- `2` - Block operation

### Hook Input (JSON via stdin)
```json
{
  "tool_name": "Bash",
  "tool_input": {
    "command": "SELECT * FROM users",
    "timeout": 30000
  }
}
```

---

## Example Agent Specifications

### Code Reviewer (Read-Only)
```yaml
---
name: code-reviewer
description: Expert code review specialist. Proactively reviews code for quality, security, and maintainability.
tools: Read, Grep, Glob, Bash
model: inherit
---

You are a senior code reviewer ensuring high standards of code quality and security.

When invoked:
1. Run git diff to see recent changes
2. Focus on modified files
3. Begin review immediately

Review checklist:
- Code is clear and readable
- Functions and variables are well-named
- No duplicated code
- Proper error handling
- No exposed secrets or API keys
- Input validation implemented
- Good test coverage
- Performance considerations addressed

Provide feedback organized by priority:
- Critical issues (must fix)
- Warnings (should fix)
- Suggestions (consider improving)
```

### Debugger (Read + Write)
```yaml
---
name: debugger
description: Debugging specialist for errors, test failures, and unexpected behavior.
tools: Read, Edit, Bash, Grep, Glob
---

You are an expert debugger specializing in root cause analysis.

Debugging process:
- Analyze error messages and logs
- Check recent code changes
- Form and test hypotheses
- Add strategic debug logging
- Inspect variable states

For each issue:
- Root cause explanation
- Evidence supporting diagnosis
- Specific code fix
- Testing approach
- Prevention recommendations
```

### Database Reader (Validated Bash)
```yaml
---
name: db-reader
description: Execute read-only database queries.
tools: Bash
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate-readonly-query.sh"
---

You are a database analyst with read-only access.
```

### Data Scientist (Domain-Specific)
```yaml
---
name: data-scientist
description: Data analysis expert for SQL queries, BigQuery operations, and data insights.
tools: Bash, Read, Write
model: sonnet
---

You are a data scientist specializing in SQL and BigQuery analysis.

Key practices:
- Write optimized SQL queries with proper filters
- Use appropriate aggregations and joins
- Include comments explaining complex logic
- Format results for readability
- Provide data-driven recommendations
```

---

## Key Patterns for Meta-Agent-Gym

### 1. Description is Critical
The `description` field is what Claude uses to decide when to delegate. Good descriptions:
- **"Expert code review specialist. Proactively reviews code..."** ← includes "proactively"
- **"Debugging specialist for errors..."** ← clear domain
- **"Data analysis expert for..."** ← specific expertise

### 2. Tool Safety
- Read-only agents: don't include `Write`, `Edit`
- Fix-capable agents: include `Edit`
- Domain-specific: include domain tools

### 3. Model Selection
- Use `haiku` for fast, cheap operations (exploration)
- Use `sonnet` for balanced capability/cost
- Use `opus` for complex reasoning

### 4. Memory for Learning
Agents can build knowledge over time:
```yaml
memory: project  # Recommended - shareable via git
memory: user     # Broad patterns across projects
memory: local    # Project-specific, not versioned
```

### 5. Hooks for Fine-Grained Control
When `tools` field isn't enough, use `PreToolUse` hooks for:
- Conditional validation
- Dynamic blocking
- Custom security rules

---

## Differences: Agents vs Skills

| Aspect | Agents | Skills |
|--------|--------|--------|
| Context | Isolated, separate context window | Main conversation context |
| System Prompt | Custom, replaces default | Augments existing |
| Tool Access | Can restrict independently | Inherits from parent |
| When to Use | Verbose output, specialized tools | Reusable workflows in main context |

---

## Application to Meta-Agent-Gym

### What We Should Generate

Our AgentSpec model should generate valid subagent specifications:

```yaml
---
name: <agent-name>
description: <when-to-use>
tools: <comma-separated-tools>
model: <model-choice>
---

<system-prompt-body>
```

### Validation Checklist

Generated specs must have:
1. ✅ Valid `name` (lowercase, hyphens)
2. ✅ Clear `description` with delegation guidance
3. ✅ Appropriate `tools` for task domain
4. ✅ Suitable `model` for task complexity
5. ✅ System prompt body that guides behavior
6. ✅ Valid YAML frontmatter format

### Scoring Criteria

When evaluating generated specs:

1. **Tool Relevance** - Do tools match the task?
   - Web scraping → `http_request`, `html_parser`
   - Data analysis → `data_transformer`, `aggregator`
   - Code review → `Read`, `Grep`, `Glob` (no Write/Edit)

2. **Description Quality** - Is delegation clear?
   - ✅ "Expert code reviewer. Use proactively after code changes."
   - ❌ "Code agent" (too vague)

3. **Model Selection** - Is model appropriate?
   - Simple tasks → `haiku` (cost-effective)
   - Complex reasoning → `sonnet` or `opus`
   - Fast iteration → `haiku`

4. **Safety** - Are permissions appropriate?
   - Read-only tasks → No `Write`/`Edit`
   - Fix tasks → Include `Edit`
   - Sensitive operations → No `bypassPermissions`

---

*Reference: https://code.claude.com/docs/en/sub-agents*
*Last Updated: 2025-04-22*

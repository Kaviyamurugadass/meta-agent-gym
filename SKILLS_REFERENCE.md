# Skills Reference - Source of Truth

> Source: https://code.claude.com/docs/en/skills
>
> Skills extend Claude's capabilities. They run in the main conversation context (unlike subagents which run in isolation).
>
> **Note: meta-agent-gym generates SUBAGENTS, not skills.** This reference is for understanding the difference.

---

## Key Difference: Skills vs Subagents

| Aspect | Skills | Subagents |
|--------|--------|-----------|
| **Context** | Main conversation | Isolated context window |
| **System Prompt** | Augments existing | Replaces existing |
| **Tool Access** | Inherits from parent | Can restrict independently |
| **When to Use** | Reusable workflows, reference | High-volume output, isolation needed |

**meta-agent-gym generates SUBAGENTS** because:
- We want agents that work independently
- We want to restrict tool access per agent
- We want custom system prompts

---

## Skill Structure (for reference)

```yaml
---
name: skill-name
description: What this skill does and when to use it
disable-model-invocation: false  # Only user can invoke
user-invocable: true             # User can invoke with /name
allowed-tools: Read Grep         # Pre-approved tools
context: fork                    # Run in subagent
agent: Explore                   # Which subagent type
---

<skill instructions>
```

---

## Key Fields for Skills

| Field | Purpose |
|-------|---------|
| `name` | Display name (becomes `/skill-name`) |
| `description` | What skill does + when to use |
| `when_to_use` | Additional trigger phrases |
| `disable-model-invocation` | Only user can invoke (not Claude) |
| `user-invocable` | Hide from `/` menu |
| `allowed-tools` | Pre-approve tools (no prompts) |
| `context: fork` | Run in isolated subagent |
| `agent` | Which subagent type to use |

---

## String Substitutions

| Variable | Description |
|----------|-------------|
| `$ARGUMENTS` | All arguments passed |
| `$0`, `$1`, etc. | Argument by position |
| `$name` | Named argument |
| `${CLAUDE_SESSION_ID}` | Current session ID |
| `${CLAUDE_SKILL_DIR}` | Skill directory path |

---

## When to Use Skills vs Subagents

**Use Skills when:**
- You have reusable workflows
- Reference material that loads on demand
- Procedures that could be in CLAUDE.md
- Want main conversation context

**Use Subagents when:**
- High-volume output would flood context
- Need custom tool restrictions
- Want isolated system prompt
- Task is self-contained

---

*Reference: https://code.claude.com/docs/en/skills*
*Last Updated: 2025-04-22*

# meta-agent-gym: Implementation Plan v4

> **Status:** ✅ P0 COMPLETE | P1 MOSTLY COMPLETE | P2 PARTIAL
> **Last Updated:** 2025-04-23
> **Focus:** Core Components First (UI Later)
> **Based on:** PyTorch OpenEnv Hackathon + RLVR Best Practices

---

## Implementation Status Summary

### ✅ COMPLETE - P0 Critical Path (13/13)
- [x] Hard Verifiers, AGENT.md Schema, Action Commands, Observation
- [x] Skill Registry, Test Cases (7 scenarios), Fast Judge
- [x] Multi-Component Reward, OpenEnv Environment, Goose Runner
- [x] Calibration Tracker, Unit Tests (136 pass), Deploy to HF

### ✅ COMPLETE - P1 MVP Enhancement (6/9)
- [x] GRPO Trainers (H100 + T4), Medium/Hard Tests
- [x] Anti-Hacking Checks, Evaluator, Integration Tests
- [⚠️] Curriculum Controller (partial)
- [❌] Adversarial Designer, Monitoring Dashboard

### ⚠️ PARTIAL - P2 Polish (2/4)
- [x] Investigation Tools, Documentation
- [⚠️] More Test Cases (7 scenarios, target: 20+)
- [❌] Demo Video

---

## v4 Updates (Post-Learning)

### Key Changes from v3

| Area | v3 | v4 (Updated) |
|------|----|--------------|
| **Verification** | Judge (LLM) only | **Three-tier**: Hard verifiers (100%) + Fast judge (90%) + Real execution (steps 3,6,9) |
| **Reward** | Single decomposed reward | **Multiple independent rewards** (RLVR approach) |
| **Curriculum** | Generic progression | **Explicit phases**: 1-skill → 2-3 skills → 3-5 skills → 5+ skills |
| **Anti-hacking** | Red herrings only | **Explicit penalties**: empty_spec (-5.0), over_engineered (-0.5), regression (-0.15) |
| **Deployment** | End of project | **Deploy FIRST** (P0, Day 5) to catch issues early |

### RLVR (Reinforcement Learning with Verifiable Rewards)

This is now our core philosophy:

1. **Use hard verifiers, NOT learned reward models**
   - YAML parse check (instant, free)
   - Required fields presence (instant, free)
   - Format compliance (instant, free)

2. **Multiple independent reward functions**
   - Track ALL components separately
   - Don't collapse to single score until final
   - Enables GRPO variance within groups

3. **Protect against reward hacking**
   - Empty specs → hard gate
   - Over-engineering → penalty
   - Judge exploitation → real execution validation

4. **Curriculum with >0 success probability**
   - Start EASY (single skill)
   - Model must occasionally succeed or learning stalls
   - Progress: 1 → 2-3 → 3-5 → 5+ skills

---

## Executive Summary

**Goal:** Build a system that generates complete AGENT.md files from user task descriptions.

**Input:** Task description + preferences (language, constraints, etc.)
**Output:** Complete AGENT.md file that works everywhere (Claude Code, Goose, Copilot, etc.)

**Testing:** Three-layer validation (Dataset → Judge (Claude) → Goose execution)

**Training:** GRPO with decomposed reward + adversarial curriculum

---

## System Architecture (Updated v4)

```
┌─────────────────────────────────────────────────────────────────┐
│                    THREE-TIER VERIFICATION (RLVR)                │
│                                                                  │
│   ┌────────────────┐    ┌────────────────┐    ┌─────────────┐  │
│   │ Hard Verifiers │ → │  Fast Judge    │ → │ Real Exec   │  │
│   │ (100%, free)   │    │ (90%, $0.01)   │    │ (10%, $$)   │  │
│   │                │    │ Claude Sonnet  │    │ Steps 3,6,9 │  │
│   └────────────────┘    └────────────────┘    └─────────────┘  │
│   YAML parse,           Quality score        Ground truth      │
│   required fields       5 dimensions         validation        │
│                                                                  │
│   ↓ REWARD SIGNAL (All components tracked separately)            │
│                                                                  │
│   { yaml_valid: 1.0, has_fields: 1.0, skill_selection: 0.8,    │
│     description: 0.9, workflow: 0.7, model: 1.0,               │
│     best_practices: 0.8, efficiency: 0.9,                      │
│     penalties: { over_engineered: -0.5 }, total: 7.6 }          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Legacy Architecture (v3 - Reference)

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER FLOW (Later)                             │
│                                                                  │
│   [User Input] → [Generate] → [AGENT.md] → [Test] → [Download] │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP (Now)                           │
│                                                                  │
│   Task → Policy → AGENT.md → Test → Reward → Update Policy       │
│                                  ↑                               │
│                          Adversarial Designer                   │
│                       (creates hard test cases)                  │
│                                                                  │
│                    POMDP Structure:                              │
│                    Hidden State = "What makes a good agent?"     │
│                    Observable = Task description + feedback      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    TWO-TIER EVALUATION (Inspired by Kube SRE Gym)│
│                                                                  │
│   ┌─────────────────────┐     ┌─────────────────────────┐     │
│   │  Fast Judge (90%)   │     │  Real Execution (10%)   │     │
│   │  Claude Sonnet      │     │  Goose Runtime          │     │
│   │  2-5 sec, $0.01     │     │  Real outcome           │     │
│   └──────────┬──────────┘     └──────────┬──────────────┘     │
│              │                            │                      │
│              ▼                            ▼                      │
│         reward (proxy)              reward (ground truth)       │
│                                                                  │
│         Calibration: Adjust judge if drift > threshold          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Priority Levels

| Priority | Meaning | Timeline |
|----------|---------|----------|
| **P0** | Critical path, blocks everything | Week 1 |
| **P1** | Important, needed for MVP | Week 2 |
| **P2** | Enhancement, nice to have | Week 3+ |
| **P3** | UI, polish, optimization | Later |

---

## Phase 1: Core Schema (P0) - Week 1

### 1.1 AGENT.md Schema

**File:** `meta_agent_gym/models/agent_spec.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from enum import Enum

class ModelType(str, Enum):
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"
    INHERIT = "inherit"

class AgentSpec(BaseModel):
    """Complete agent specification following Agent Skills Open Standard."""

    # Required fields
    name: str = Field(..., description="Agent name (lowercase, hyphens)")
    description: str = Field(..., description="What agent does + when to use it")

    # Skills (capabilities)
    skills: List[str] = Field(default_factory=list, description="Skills from skills.sh or custom")

    # Model selection
    model: ModelType = Field(default=ModelType.SONNET, description="Which model to use")

    # System prompt
    system_prompt: str = Field(..., description="Agent instructions and workflow")

    # Optional metadata
    user_invocable: bool = Field(default=True, description="Can users invoke directly?")
    allowed_tools: Optional[List[str]] = Field(default=None, description="Tool restrictions")
    memory: Optional[str] = Field(default=None, description="Memory scope: user/project/local")
    max_turns: Optional[int] = Field(default=None, description="Max agent turns")

    def to_markdown(self) -> str:
        """Convert to AGENT.md format."""
        frontmatter = {
            "name": self.name,
            "description": self.description,
            "user-invocable": self.user_invocable,
        }
        if self.allowed_tools:
            frontmatter["allowed-tools"] = ", ".join(self.allowed_tools)
        if self.skills:
            frontmatter["skills"] = self.skills
        if self.model:
            frontmatter["model"] = self.model.value
        if self.memory:
            frontmatter["memory"] = self.memory
        if self.max_turns:
            frontmatter["max-turns"] = self.max_turns

        yaml_frontmatter = "---\n"
        for key, value in frontmatter.items():
            if isinstance(value, bool):
                yaml_frontmatter += f"{key}: {str(value).lower()}\n"
            elif isinstance(value, list):
                yaml_frontmatter += f"{key}:\n"
                for item in value:
                    yaml_frontmatter += f"  - {item}\n"
            else:
                yaml_frontmatter += f"{key}: {value}\n"
        yaml_frontmatter += "---\n"

        return yaml_frontmatter + "\n" + self.system_prompt
```

**Tests:** `tests/test_agent_spec.py`
- Test YAML generation
- Test skill list handling
- Test model selection
- Test roundtrip (parse → generate)
- Test all optional fields

---

### 1.2 Action Space - Command-Based (NEW: Inspired by DNS Arena)

**File:** `meta_agent_gym/models/action.py`

Instead of generating full AGENT.md every step, use discrete commands:

```python
from enum import Enum
from typing import Dict, Optional, Any
from pydantic import BaseModel

class AgentGenCommand(str, Enum):
    """Discrete commands for agent generation - token-efficient."""
    SET_NAME = "set_name"
    SET_DESCRIPTION = "set_description"
    ADD_SKILL = "add_skill"
    REMOVE_SKILL = "remove_skill"
    SET_MODEL = "set_model"
    ADD_TOOLS = "add_tools"
    WRITE_PROMPT = "write_prompt"
    SET_MEMORY = "set_memory"
    CHECK_SCORE = "check_score"      # Investigation tool
    INSPECT_EXAMPLE = "inspect_example"  # Investigation tool
    SUBMIT = "submit"

class AgentGenAction(BaseModel):
    """Action for the agent generation environment."""
    command: AgentGenCommand
    args: Dict[str, Any] = Field(default_factory=dict)

    # For investigation commands
    return_observation: bool = False  # If True, return detailed feedback
```

**Benefits:**
- Token efficient (no full file rewrites)
- Enables incremental fixes
- Allows investigation before submission
- Better for GRPO (clearer action space)

---

### 1.3 Observation Space - POMDP Structure (NEW: Inspired by Bio Env)

**File:** `meta_agent_gym/models/observation.py`

```python
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

class AgentGenObservation(BaseModel):
    """Observation for agent generation environment."""

    # Task information
    task_id: str
    task_description: str
    difficulty: str  # easy, medium, hard, expert
    user_preferences: Dict[str, Any]

    # Current agent spec state (partial observability)
    current_spec: Dict[str, Any] = Field(
        default_factory=dict,
        description="Partial view of current agent state"
    )

    # Feedback from investigation
    investigation_result: Optional[Dict[str, Any]] = None

    # Score and reward breakdown (decomposed)
    score: float = 0.0
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: List[str] = Field(default_factory=list)

    # Episode state
    steps_remaining: int
    max_steps: int

    # Hidden state (NOT visible to policy):
    # - True optimal spec
    # - Ground truth evaluation
    # - Adversarial targets
```

---

### 1.4 Skill Registry

**File:** `meta_agent_gym/core/skills.py`

```python
# Common skills from skills.sh + categorization
AVAILABLE_SKILLS = {
    # Web & Data
    "web-scraping": "Extract data from websites",
    "http-client": "Make HTTP requests",
    "html-parser": "Parse HTML content",
    "json-parser": "Parse JSON data",
    "csv-handler": "Handle CSV files",

    # Data Processing
    "data-transformer": "Transform data structures",
    "data-validator": "Validate data against schemas",
    "data-aggregator": "Aggregate and summarize data",

    # Code
    "code-reviewer": "Review code for quality",
    "code-fixer": "Fix common code issues",
    "test-generator": "Generate test cases",

    # Files
    "file-reader": "Read file contents",
    "file-writer": "Write file contents",

    # Analysis
    "log-analyzer": "Analyze log files",
    "pattern-matcher": "Match patterns in data",

    # Output
    "report-generator": "Generate reports",
    "notifier": "Send notifications",
}

SKILL_CATEGORIES = {
    "web": ["web-scraping", "http-client", "html-parser"],
    "data": ["json-parser", "csv-handler", "data-transformer", "data-validator"],
    "code": ["code-reviewer", "code-fixer", "test-generator"],
    "files": ["file-reader", "file-writer"],
    "analysis": ["log-analyzer", "pattern-matcher"],
    "output": ["report-generator", "notifier"],
}

# Suggest skills for task types (for curriculum)
TASK_SKILL_MAP = {
    "web_scraping": ["web-scraping", "html-parser", "http-client"],
    "data_analysis": ["csv-handler", "data-transformer", "data-validator"],
    "code_review": ["code-reviewer", "file-reader", "pattern-matcher"],
    "testing": ["test-generator", "code-fixer", "file-reader"],
}
```

---

### 1.5 Domain Templates

**File:** `meta_agent_gym/core/templates.py`

```python
# Generic agent templates following best practices
AGENT_TEMPLATES = {
    "web-scraper": """You are a web scraping specialist.

Best practices:
- Always check robots.txt
- Implement rate limiting
- Handle errors gracefully
- Validate extracted data
- Store results in structured format

When scraping:
1. Identify target structure
2. Select appropriate extraction method
3. Handle pagination
4. Store results
""",

    "data-analyst": """You are a data analysis specialist.

Best practices:
- Validate data before analysis
- Document assumptions
- Handle missing data
- Visualize results
- Provide actionable insights

When analyzing:
1. Understand data structure
2. Clean and validate
3. Explore patterns
4. Generate insights
5. Create visualizations
""",

    "code-reviewer": """You are a code review specialist.

Best practices:
- Focus on code quality
- Check for security issues
- Verify error handling
- Assess performance
- Suggest improvements

Review checklist:
- Code is clear and readable
- Proper error handling
- No security vulnerabilities
- Good test coverage
- Follows team conventions
""",

    "api-integrator": """You are an API integration specialist.

Best practices:
- Validate inputs
- Handle API errors
- Implement retry logic
- Cache responses
- Log operations

When integrating:
1. Read API documentation
2. Design error handling
3. Implement retry logic
4. Add caching
5. Test thoroughly
""",
}
```

---

## Phase 2: Dataset Generator (P0) - Week 1

### 2.1 Test Case Schema

**File:** `meta_agent_gym/core/test_case.py`

```python
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class TestCase(BaseModel):
    """A test case for validating generated agents."""

    id: str
    domain: str  # web, data, code, etc.
    difficulty: str  # easy, medium, hard, expert

    # Input to policy
    task_description: str
    user_preferences: Dict[str, Any]

    # Expected output
    required_skills: List[str]
    required_model: str  # haiku/sonnet/opus

    # Test data
    test_input: Dict[str, Any]
    expected_output: Dict[str, Any]

    # Validation criteria
    validation_checks: List[str]

    # NEW: Red herrings (decoys that look wrong but are correct)
    red_herrings: List[str] = Field(
        default_factory=list,
        description="Patterns that look wrong but shouldn't be 'fixed'"
    )

    # NEW: Cross-file dependencies (for hard tasks)
    dependencies: List[str] = Field(
        default_factory=list,
        description="Other test cases this depends on"
    )
```

---

### 2.2 Generic Test Cases

**File:** `data/test_cases/generic.json`

```json
{
  "web_scraping_easy": {
    "id": "ws_easy_001",
    "domain": "web",
    "difficulty": "easy",
    "task_description": "Build an agent that extracts product prices from a single e-commerce page",
    "user_preferences": {
      "language": "python",
      "constraints": ["must_handle_errors", "respect_rate_limits"]
    },
    "required_skills": ["web-scraping", "html-parser"],
    "required_model": "sonnet",
    "test_input": {
      "url": "https://example-shop.com/products",
      "selector": ".product-price"
    },
    "expected_output": {
      "format": "json",
      "fields": ["name", "price", "currency"]
    },
    "validation_checks": [
      "Has web-scraping skill",
      "Handles HTML parsing errors",
      "Returns structured JSON"
    ],
    "red_herrings": [
      "Don't add selenium unless task explicitly requires JavaScript rendering",
      "Don't add rate limiting if task is single-page only"
    ]
  },

  "data_analysis_medium": {
    "id": "da_med_001",
    "domain": "data",
    "difficulty": "medium",
    "task_description": "Build an agent that analyzes CSV data, handles missing values, and generates summary statistics",
    "user_preferences": {
      "language": "python",
      "output_format": "report"
    },
    "required_skills": ["csv-handler", "data-transformer", "data-validator"],
    "required_model": "sonnet",
    "test_input": {
      "file": "sales_data.csv",
      "analysis": ["mean", "median", "std", "missing_handling"]
    },
    "expected_output": {
      "format": "report",
      "sections": ["summary", "statistics", "missing_data_analysis"]
    },
    "validation_checks": [
      "Has csv-handler skill",
      "Handles missing data explicitly",
      "Generates readable report"
    ],
    "red_herrings": [
      "Don't add visualization unless explicitly requested",
      "Simple imputation is acceptable for this task"
    ]
  },

  "code_review_hard": {
    "id": "cr_hard_001",
    "domain": "code",
    "difficulty": "hard",
    "task_description": "Build an agent that reviews code for security vulnerabilities, suggests fixes, and validates with test generation",
    "user_preferences": {
      "focus": "security",
      "severity": "critical_and_high"
    },
    "required_skills": ["code-reviewer", "test-generator", "pattern-matcher"],
    "required_model": "opus",
    "test_input": {
      "codebase": "multi_file_project",
      "security_checks": ["sql_injection", "xss", "csrf", "auth_issues"]
    },
    "expected_output": {
      "report": "security_report.md",
      "test_coverage": "generated_tests/"
    },
    "validation_checks": [
      "Has code-reviewer skill",
      "Checks specific security patterns",
      "Generates test cases",
      "Uses opus for complex reasoning"
    ],
    "dependencies": ["ws_easy_001", "da_med_001"],
    "red_herrings": [
      "Don't suggest refactoring working code unless security issue",
      "False positives on legitimate user input handling are acceptable"
    ]
  }
}
```

---

### 2.3 Adversarial Test Cases

**File:** `meta_agent_gym/adversarial/designer.py`

```python
from anthropic import Anthropic
from meta_agent_gym.core.test_case import TestCase

class AdversarialDesigner:
    """Creates challenging test cases using Claude (inspired by Kube SRE Gym)."""

    CHALLENGE_TYPES = [
        "edge_cases",        # Unusual inputs
        "resource_constraints",  # Low token budget
        "ambiguous_requirements",  # Unclear task
        "multi_step",        # Complex workflows
        "error_scenarios",   # Error handling required
        "performance",       # Speed/cost optimization
        "security",          # Security-sensitive tasks
    ]

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def generate_challenges(
        self,
        base_case: TestCase,
        policy_performance: dict,
        num_challenges: int = 3
    ) -> list[TestCase]:
        """Generate harder test cases targeting policy weaknesses."""

        # Identify weaknesses
        weak_areas = [
            area for area, score in policy_performance.items()
            if score < 0.5
        ]

        prompt = f"""You are an adversarial test designer. Create {num_challenges} challenging test cases.

Base task: {base_case.task_description}

Policy weaknesses: {', '.join(weak_areas)}

Create test cases that:
1. Target these weaknesses specifically
2. Are valid tasks users would actually ask for
3. Have clear success criteria
4. Increase difficulty appropriately
5. Include red herrings (patterns that look wrong but are correct)

Respond in JSON format:
[
  {{
    "task_description": "...",
    "difficulty": "medium/hard/expert",
    "challenge_type": "{weak_areas[0] if weak_areas else 'multi_step'}",
    "required_skills": [...],
    "test_input": {{...}},
    "validation_checks": [...],
    "red_herrings": [...]
  }}
]"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._parse_challenges(response.content[0].text)
```

---

## Phase 3: Reward System (P0) - Week 1

### 3.1 Multi-Component Reward (UPDATED v4: RLVR Approach)

**File:** `server/rewards/reward.py`

**Key Change:** Multiple independent reward functions, not just one decomposed reward.

```python
from enum import Enum
from typing import Dict, Optional, List
from pydantic import BaseModel

class RewardMode(str, Enum):
    """Reward calculation modes."""
    ADDITIVE = "additive"       # Sum of weighted components
    MULTIPLICATIVE = "multiplicative"  # Product (prevents fake wins)
    HYBRID = "hybrid"           # Gate on safety, additive others

class AntiHackPenalty(BaseModel):
    """Anti-hacking penalties (NEW v4)."""
    empty_spec: float = -5.0          # No prompt or <50 chars
    over_engineered: float = -0.5     # >10 skills or wrong model
    regression: float = -0.15         # Broke previously passing check
    repetitive: float = -0.3          # Repeated surface patterns

class RewardConfig(BaseModel):
    """Configuration for reward calculation (UPDATED v4)."""
    mode: RewardMode = RewardMode.HYBRID

    # NEW v4: Independent reward functions (RLVR)
    independent_rewards: List[str] = [
        "yaml_valid",           # Hard verifier: 0.0 or 1.0
        "has_required_fields",  # Hard verifier: 0.0 or 1.0
        "skill_selection",      # Judge: 0.0 to 1.0
        "description_quality",  # Judge: 0.0 to 1.0
        "workflow_clarity",     # Judge: 0.0 to 1.0
        "model_appropriateness", # Judge: 0.0 to 1.0
        "best_practices",       # Judge: 0.0 to 1.0
        "efficiency",           # Judge: 0.0 to 1.0
    ]

    # Per-step component weights
    component_weights: Dict[str, float] = {
        "skill_selection": 0.25,
        "description_quality": 0.20,
        "workflow_clarity": 0.20,
        "model_appropriateness": 0.15,
        "best_practices": 0.10,
        "efficiency": 0.10,
    }

    # Gate components (for HYBRID mode)
    gate_components: List[str] = ["yaml_valid", "has_required_fields"]
    gate_threshold: float = 0.99

    # NEW v4: Anti-hacking penalties
    anti_hack: AntiHackPenalty = AntiHackPenalty()

    # Terminal rewards
    terminal_success: float = 5.0
    terminal_failure: float = -2.0

class RewardComputer:
    """Compute multi-component rewards for agent generation."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self._passing_checks = set()
        self._history = []  # Track for regression detection

    def compute_reward(
        self,
        action,
        current_spec: dict,
        test_case: TestCase,
        previous_score: float = 0.0
    ) -> Dict[str, float]:
        """Compute reward with all components tracked separately."""

        # 1. Hard verifiers (first - free, instant)
        hard_rewards = self._hard_verifiers(current_spec)

        # 2. Check gates
        for gate in self.config.gate_components:
            if hard_rewards.get(gate, 1.0) < self.config.gate_threshold:
                return {"total": 0.0, "gated": True, "gate_failed": gate}

        # 3. Judge rewards (90% of steps)
        judge_rewards = self._judge_rewards(current_spec, test_case)

        # 4. Anti-hacking checks (every step)
        penalties = self._anti_hack_checks(current_spec, action)

        # 5. Calculate total
        core_reward = sum(
            judge_rewards[k] * self.config.component_weights.get(k, 1.0)
            for k in judge_rewards
        )

        total_penalty = sum(penalties.values())
        total = core_reward + total_penalty

        # 6. Track for regression detection
        self._update_history(current_spec, hard_rewards, judge_rewards)

        return {
            "total": total,
            "delta": total - previous_score,
            "breakdown": {
                **hard_rewards,
                **judge_rewards,
            },
            "penalties": penalties,
        }

    def _hard_verifiers(self, spec: dict) -> Dict[str, float]:
        """Fast, free verification checks (RLVR approach)."""
        rewards = {}

        # YAML validity
        try:
            yaml.safe_dump(spec)
            rewards["yaml_valid"] = 1.0
        except:
            rewards["yaml_valid"] = 0.0

        # Required fields
        required = ["name", "description", "system_prompt"]
        has_all = all(field in spec and spec[field] for field in required)
        rewards["has_required_fields"] = 1.0 if has_all else 0.0

        # Prompt length check (anti-empty)
        prompt = spec.get("system_prompt", "")
        rewards["prompt_length_ok"] = 1.0 if len(prompt) > 50 else 0.0

        return rewards

    def _anti_hack_checks(self, spec: dict, action) -> Dict[str, float]:
        """NEW v4: Detect and penalize reward hacking attempts."""
        penalties = {}

        # Empty spec check
        prompt = spec.get("system_prompt", "")
        if len(prompt) < 50:
            penalties["empty_spec"] = self.config.anti_hack.empty_spec

        # Over-engineering check
        skills = spec.get("skills", [])
        if len(skills) > 10:
            penalties["over_engineered"] = self.config.anti_hack.over_engineered

        # Wrong model tier check
        model = spec.get("model", "sonnet")
        if model == "opus" and not spec.get("requires_opus", False):
            penalties["over_engineered"] = self.config.anti_hack.over_engineered

        # Regression check
        regression_count = self._count_regressions(spec)
        if regression_count > 0:
            penalties["regression"] = regression_count * self.config.anti_hack.regression

        return penalties

    def _count_regressions(self, spec: dict) -> int:
        """Count how many previously-passing checks are now failing."""
        # Compare with history
        count = 0
        for prev in self._history:
            # Check if spec broke something that was passing before
            if self._check_regression(prev, spec):
                count += 1
        return count
```

**File:** `meta_agent_gym/rewards/reward.py`

```python
from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel

class RewardMode(str, Enum):
    """Reward calculation modes."""
    ADDITIVE = "additive"       # Sum of weighted components
    MULTIPLICATIVE = "multiplicative"  # Product (prevents fake wins)
    HYBRID = "hybrid"           # Gate on safety, additive elsewhere

class RewardConfig(BaseModel):
    """Configuration for reward calculation."""
    mode: RewardMode = RewardMode.HYBRID

    # Per-step component weights
    component_weights: Dict[str, float] = {
        "skill_selection": 0.25,
        "description_quality": 0.20,
        "workflow_clarity": 0.20,
        "model_appropriateness": 0.15,
        "best_practices": 0.10,
        "efficiency": 0.10,
    }

    # Gate components (for HYBRID mode)
    gate_components: list = ["safety"]
    gate_threshold: float = 0.01

    # Penalties
    regression_penalty: float = 0.15  # Per broken check
    soft_violation_penalty: float = 0.05  # Per soft violation

    # Bonuses
    novelty_bonus: float = 0.10  # When no violations

    # Terminal rewards
    terminal_success: float = 5.0
    terminal_failure: float = -2.0

class RewardComputer:
    """Compute decomposed rewards for agent generation."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self._passing_checks = set()

    def compute_reward(
        self,
        action,
        current_spec: dict,
        test_case: TestCase,
        previous_score: float = 0.0
    ) -> Dict[str, float]:
        """Compute decomposed reward with all components."""

        # 1. Component scores
        components = self._component_scores(current_spec, test_case)

        # 2. Check for regressions
        regression_count = self._count_regressions(components, self._passing_checks)
        regression_penalty = regression_count * self.config.regression_penalty

        # 3. Count soft violations (from judge)
        soft_violations = self._count_soft_violations(current_spec)
        violation_penalty = soft_violations * self.config.soft_violation_penalty

        # 4. Calculate based on mode
        if self.config.mode == RewardMode.ADDITIVE:
            core_reward = sum(
                components[k] * self.config.component_weights.get(k, 1.0)
                for k in components
            )
        elif self.config.mode == RewardMode.MULTIPLICATIVE:
            core_reward = 10.0
            for k, v in components.items():
                weight = self.config.component_weights.get(k, 1.0)
                core_reward *= (v ** weight)
        else:  # HYBRID
            # Check gates
            for gate in self.config.gate_components:
                if components.get(gate, 1.0) < self.config.gate_threshold:
                    return {"total": 0.0, "gated": True}
            # Additive if gates pass
            core_reward = sum(
                components[k] * self.config.component_weights.get(k, 1.0)
                for k in components
            )

        # 5. Add bonuses/penalties
        novelty = self.config.novelty_bonus if soft_violations == 0 else 0
        total = core_reward + novelty - regression_penalty - violation_penalty

        # 6. Track passing checks for regression detection
        self._update_passing_checks(components)

        return {
            "total": total,
            "delta": total - previous_score,
            "breakdown": components,
            "regression_penalty": regression_penalty,
            "violation_penalty": violation_penalty,
            "novelty_bonus": novelty,
        }

    def _component_scores(self, spec: dict, test_case: TestCase) -> Dict[str, float]:
        """Score each component 0-1."""
        return {
            "skill_selection": self._score_skill_selection(spec, test_case),
            "description_quality": self._score_description(spec),
            "workflow_clarity": self._score_workflow(spec),
            "model_appropriateness": self._score_model(spec, test_case),
            "best_practices": self._score_best_practices(spec),
            "efficiency": self._score_efficiency(spec),
        }

    def _score_skill_selection(self, spec: dict, test_case: TestCase) -> float:
        """Score: are skills appropriate for the task?"""
        required = set(test_case.required_skills)
        has = set(spec.get("skills", []))

        # Check coverage
        coverage = len(required & has) / len(required) if required else 0

        # Penalize extra skills
        extra = len(has - required)
        extra_penalty = min(extra * 0.1, 0.3)

        return max(0, coverage - extra_penalty)

    def _score_description(self, spec: dict) -> float:
        """Score: is description clear with delegation guidance?"""
        desc = spec.get("description", "")

        # Check for delegation keywords
        delegation_words = ["proactively", "use", "when", "specialist", "expert"]
        has_delegation = any(word in desc.lower() for word in delegation_words)

        # Check length
        good_length = 20 <= len(desc.split()) <= 100

        return min(1.0, 0.3 * has_delegation + 0.4 * good_length + 0.3 * (len(desc) > 0))

    # ... other score methods

    def _count_regressions(self, current: dict, previously_passing: set) -> int:
        """Count how many previously-passing checks are now failing."""
        # Implementation tracks check IDs and compares
        return 0

    def _count_soft_violations(self, spec: dict) -> int:
        """Count soft violations (redundant actions, over-engineering)."""
        violations = 0
        # Check for redundant skills
        skills = spec.get("skills", [])
        if len(skills) > 10:  # Arbitrary threshold
            violations += 1
        # Check for over-qualified model
        if spec.get("model") == "opus" and not spec.get("requires_opus"):
            violations += 1
        return violations
```

---

### 3.2 Reward Justification Template

**File:** `meta_agent_gym/rewards/justification.md`

For README documentation - explains reward design to judges:

```markdown
## Reward Design

### Mode: HYBRID (Recommended)

**Rationale:** Pure multiplicative collapses to zero too easily. Pure additive
allows gaming (high in one dimension compensates for zero in another). Hybrid
gates on critical failures (safety) while providing smooth gradient elsewhere.

**Formula:**
```
If safety < 0.01:
    reward = 0
Else:
    reward = Σ (component_score × weight) + bonuses - penalties
```

### Component Weights

| Component | Weight | Formula | Why |
|-----------|--------|---------|-----|
| skill_selection | 0.25 | coverage - 0.1×extra | Core capability |
| description_quality | 0.20 | delegation + length | Critical for delegation |
| workflow_clarity | 0.20 | step_count × structure | Enables autonomous use |
| model_appropriateness | 0.15 | matches_complexity | Cost optimization |
| best_practices | 0.10 | style_guidelines adherence | Production quality |
| efficiency | 0.10 | 1 - over_engineering | Prevents bloat |

### Penalties & Bonuses

| Type | Value | When |
|------|-------|------|
| Regression | −0.15/check | Breaking previously-passing check |
| Soft violation | −0.05/violation | Redundant skills, over-qualified model |
| Novelty | +0.10 | Clean step with no violations |

### GRPO-Friendly Properties

1. **Continuous, not binary** - Partial credit for partial progress
2. **Variance within groups** - Different completions produce different rewards
3. **No collapse to zero** - Hybrid mode prevents multiplicative collapse
4. **No collapse to ceiling** - Hard tasks have strict requirements
```

---

## Phase 4: Judge System (P0) - Week 1

### 4.1 Three-Tier Evaluation (UPDATED v4)

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    THREE-TIER EVALUATION                         │
│                                                                  │
│   Step 1,2,4,5,7,8...:    Step 3,6,9:                           │
│   ┌────────────────┐    ┌────────────────────────────────┐    │
│   │ Hard Verifiers │ → │      Fast Judge (90%)          │    │
│   │ (free, 100%)   │    │      Claude Sonnet             │    │
│   └────────────────┘    │      5-dim quality scoring     │    │
│                          └────────────┬───────────────────┘    │
│                                       │                         │
│                          ┌────────────▼───────────────────┐    │
│                          │  Real Execution (10%)          │    │
│                          │  Goose runtime                 │    │
│                          │  Ground truth validation       │    │
│                          │  ONLY at steps 3, 6, 9         │    │
│                          └────────────────────────────────┘    │
│                                                                  │
│   Calibration: Track fast vs real drift, adjust if >1.0         │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Judge Schema (UPDATED v4)

**File:** `server/judge.py` (or integrated in environment)

```python
from anthropic import Anthropic
from meta_agent_gym.models.agent_spec import AgentSpec
from pydantic import BaseModel
from typing import Dict, List, Optional

class JudgeResult(BaseModel):
    """Result from judging an agent specification (UPDATED v4)."""
    score: float  # 0.0 to 10.0
    passed: bool  # Meets minimum threshold?
    feedback: str
    component_scores: Dict[str, float]  # All components
    violations: List[str]
    soft_violations: List[str]

class CalibrationTracker:
    """NEW v4: Track fast judge vs real execution drift."""

    def __init__(self, drift_threshold: float = 1.0):
        self.drift_threshold = drift_threshold
        self.calibration_data: List[Dict] = []

    def add_comparison(self, fast_score: float, real_score: float):
        """Add a fast vs real comparison."""
        self.calibration_data.append({
            "fast": fast_score,
            "real": real_score,
            "diff": abs(fast - real_score)
        })

    def check_drift(self) -> bool:
        """Check if fast judge is drifting."""
        if len(self.calibration_data) < 10:
            return False

        mean_diff = sum(d["diff"] for d in self.calibration_data) / len(self.calibration_data)
        return mean_diff > self.drift_threshold

class AgentJudge:
    """Judge agent specifications using Claude (UPDATED v4)."""

    # NEW v4: Run at 90% frequency (skip steps 3,6,9 for real exec)
    JUDGE_FREQUENCY = 0.9
    REAL_EXEC_STEPS = [3, 6, 9]  # Steps to run real execution

    WEIGHTS = {
        "skill_selection": 3.0,
        "description_quality": 2.0,
        "workflow_clarity": 2.0,
        "model_appropriateness": 1.5,
        "best_practices": 1.5,
    }

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.calibration = CalibrationTracker()

    def should_judge(self, step: int) -> bool:
        """Check if we should run judge this step."""
        # Real execution steps get both judge AND real execution
        if step in self.REAL_EXEC_STEPS:
            return True
        # Otherwise, probabilistic skip
        import random
        return random.random() < self.JUDGE_FREQUENCY

    def evaluate(
        self,
        agent_spec: AgentSpec,
        test_case: TestCase,
        detailed: bool = False
    ) -> JudgeResult:
        """Evaluate an agent spec against a test case."""

        prompt = self._build_prompt(agent_spec, test_case)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._parse_response(response.content[0].text)

    def calibrate(self, real_result: Dict) -> bool:
        """NEW v4: Update calibration with real execution result."""
        fast_score = self.last_score
        real_score = real_result.get("score", 0.0)

        self.calibration.add_comparison(fast_score, real_score)

        if self.calibration.check_drift():
            self._recalibrate()
            return True
        return False
```

### 4.3 Hard Verifiers (NEW v4)

**File:** `server/verifiers.py`

```python
import yaml
from typing import Dict, List, Tuple
from pydantic import BaseModel

class VerifierResult(BaseModel):
    """Result from hard verification (fast, free)."""
    passed: bool
    score: float  # 0.0 or 1.0 (binary)
    errors: List[str]

class HardVerifiers:
    """Fast, free verification checks (RLVR approach)."""

    @staticmethod
    def verify_yaml(spec: dict) -> VerifierResult:
        """Check if spec can be serialized to valid YAML."""
        try:
            yaml.safe_dump(spec)
            return VerifierResult(passed=True, score=1.0, errors=[])
        except Exception as e:
            return VerifierResult(passed=False, score=0.0, errors=[str(e)])

    @staticmethod
    def verify_required_fields(spec: dict) -> VerifierResult:
        """Check required fields are present and non-empty."""
        required = ["name", "description", "system_prompt"]
        errors = []

        for field in required:
            if field not in spec:
                errors.append(f"Missing required field: {field}")
            elif not spec[field]:
                errors.append(f"Empty required field: {field}")

        return VerifierResult(
            passed=len(errors) == 0,
            score=1.0 if len(errors) == 0 else 0.0,
            errors=errors
        )

    @staticmethod
    def verify_prompt_length(spec: dict, min_length: int = 50) -> VerifierResult:
        """Check system prompt meets minimum length (anti-empty)."""
        prompt = spec.get("system_prompt", "")
        if len(prompt) < min_length:
            return VerifierResult(
                passed=False,
                score=0.0,
                errors=[f"Prompt too short: {len(prompt)} < {min_length}"]
            )
        return VerifierResult(passed=True, score=1.0, errors=[])

    @classmethod
    def verify_all(cls, spec: dict) -> Dict[str, VerifierResult]:
        """Run all hard verifiers."""
        return {
            "yaml_valid": cls.verify_yaml(spec),
            "has_required_fields": cls.verify_required_fields(spec),
            "prompt_length_ok": cls.verify_prompt_length(spec),
        }
```

### 4.1 Judge Schema

**File:** `meta_agent_gym/judge/judge.py`

```python
from anthropic import Anthropic
from meta_agent_gym.models.agent_spec import AgentSpec
from pydantic import BaseModel

class JudgeResult(BaseModel):
    """Result from judging an agent specification."""

    score: float  # 0.0 to 10.0
    passed: bool  # Meets minimum threshold?
    feedback: str
    component_scores: dict
    violations: list[str]
    soft_violations: list[str]  # NEW: Track soft violations

class AgentJudge:
    """Judge agent specifications using Claude."""

    # Scoring weights - match reward system
    WEIGHTS = {
        "skill_selection": 3.0,
        "description_quality": 2.0,
        "workflow_clarity": 2.0,
        "model_appropriateness": 1.5,
        "best_practices": 1.5,
    }

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.calibration_data = []  # Track fast vs real rewards

    def evaluate(
        self,
        agent_spec: AgentSpec,
        test_case: TestCase,
        detailed: bool = False
    ) -> JudgeResult:
        """Evaluate an agent spec against a test case."""

        prompt = self._build_prompt(agent_spec, test_case)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._parse_response(response.content[0].text)

    def _build_prompt(self, spec: AgentSpec, test_case: TestCase) -> str:
        return f"""You are an expert agent architect evaluating an agent specification.

TASK: {test_case.task_description}

REQUIRED SKILLS: {', '.join(test_case.required_skills)}

RED HERRINGS (patterns that look wrong but are correct):
{chr(10).join(f"- {h}" for h in test_case.red_herrings) if test_case.red_herrings else "None"}

AGENT SPECIFICATION:
---
{spec.to_markdown()}
---

Evaluate on five dimensions (score each 0-2, explain reasoning):

1. SKILL SELECTION (weight: 3.0)
   - 2.0: Perfect skills for the task
   - 1.0: Mostly relevant, missing one key skill
   - 0.0: Missing critical skills or has unnecessary skills

2. DESCRIPTION QUALITY (weight: 2.0)
   - 2.0: Clear "when to use" guidance
   - 1.0: Somewhat clear description
   - 0.0: Vague or missing delegation guidance

3. WORKFLOW CLARITY (weight: 2.0)
   - 2.0: Clear, step-by-step instructions
   - 1.0: Basic workflow present
   - 0.0: No clear workflow or instructions

4. MODEL APPROPRIATENESS (weight: 1.5)
   - 1.5: Optimal model for task complexity
   - 0.75: Acceptable but not optimal
   - 0.0: Inappropriate model choice

5. BEST PRACTICES (weight: 1.5)
   - 1.5: Follows domain best practices
   - 0.75: Some best practices
   - 0.0: Violates best practices

6. SOFT VIOLATIONS (penalty)
   - Mark: "redundant_skills" if too many overlapping skills
   - Mark: "over_qualified_model" if using opus when sonnet suffices
   - Mark: "over_engineered" if spec is more complex than task requires

Respond in JSON:
{{
  "skill_selection": {{"score": 0-2, "reasoning": "..."}},
  "description_quality": {{"score": 0-2, "reasoning": "..."}},
  "workflow_clarity": {{"score": 0-2, "reasoning": "..."}},
  "model_appropriateness": {{"score": 0-2, "reasoning": "..."}},
  "best_practices": {{"score": 0-2, "reasoning": "..."}},
  "soft_violations": [],
  "overall_feedback": "..."
}}
"""

    def calibrate(self, drift_threshold: float = 1.0) -> bool:
        """Check if fast judge is drifting from real execution rewards."""
        if len(self.calibration_data) < 10:
            return False

        # Calculate correlation
        fast_scores = [d["fast"] for d in self.calibration_data]
        real_scores = [d["real"] for d in self.calibration_data]

        # Simple correlation check
        mean_diff = sum(abs(f - r) for f, r in zip(fast_scores, real_scores)) / len(fast_scores)

        if mean_diff > drift_threshold:
            # Drift detected - update prompt or weights
            self._recalibrate()
            return True
        return False
```

---

## Phase 5: Goose Integration (P0) - Week 1

### 5.1 Goose Runner

**File:** `meta_agent_gym/runtime/goose.py`

```python
import subprocess
import json
from pathlib import Path
from meta_agent_gym.models.agent_spec import AgentSpec

class GooseRunner:
    """Run agents using Goose and collect metrics."""

    def __init__(self, goose_path: str = "goose"):
        self.goose_path = goose_path

    def run(
        self,
        agent_spec: AgentSpec,
        test_input: dict,
        timeout: int = 60
    ) -> dict:
        """Execute agent with test input."""

        # Write agent spec to temp file
        agent_file = Path("/tmp/agent.md")
        agent_file.write_text(agent_spec.to_markdown())

        # Prepare test input
        test_file = Path("/tmp/test_input.json")
        test_file.write_text(json.dumps(test_input))

        # Run goose
        cmd = [
            self.goose_path,
            "run",
            str(agent_file),
            "--test", str(test_file),
            "--timeout", str(timeout)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 10
        )

        return self._parse_result(result)

    def _parse_result(self, result: subprocess.CompletedProcess) -> dict:
        """Parse goose execution result."""
        try:
            output = json.loads(result.stdout)
            return {
                "success": output.get("success", False),
                "output": output.get("output", ""),
                "error": output.get("error", ""),
                "tokens_used": output.get("tokens", 0),
                "duration": output.get("duration", 0),
            }
        except:
            return {
                "success": False,
                "error": result.stderr or "Unknown error",
                "output": result.stdout
            }
```

---

## Phase 6: Environment Implementation (P0) - Week 1

### 6.1 OpenEnv Environment

**File:** `meta_agent_gym/environment.py`

```python
from typing import Dict, Any, Tuple
from meta_agent_gym.models.action import AgentGenAction, AgentGenCommand
from meta_agent_gym.models.observation import AgentGenObservation
from meta_agent_gym.models.state import AgentGenState
from meta_agent_gym.core.test_case import TestCase
from meta_agent_gym.rewards.reward import RewardComputer, RewardConfig
from meta_agent_gym.judge.judge import AgentJudge
import openenv

class MetaAgentEnv(openenv.core.Environment):
    """OpenEnv environment for meta-agent training."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        judge_api_key: str,
        reward_config: RewardConfig = None,
        max_steps: int = 7
    ):
        self.judge = AgentJudge(judge_api_key)
        self.reward_computer = RewardComputer(reward_config)
        self.max_steps = max_steps
        self._state: AgentGenState = None
        self._task: TestCase = None

    def reset(self, task_id: str, **kwargs) -> AgentGenObservation:
        """Reset environment for a new episode."""
        # Load test case
        self._task = self._load_test_case(task_id)

        # Initialize state
        self._state = AgentGenState(
            task_id=task_id,
            max_steps=self.max_steps,
            current_spec={},  # Start with empty spec
        )

        # Build initial observation
        return self._build_observation()

    def step(self, action: AgentGenAction) -> Tuple[AgentGenObservation, float, bool, dict]:
        """Execute one step."""
        # Apply action to state
        self._apply_action(action)

        # Check if investigation command
        if action.command in [AgentGenCommand.CHECK_SCORE, AgentGenCommand.INSPECT_EXAMPLE]:
            investigation_result = self._investigate(action)
            obs = self._build_observation(investigation_result)
            return obs, 0.0, False, {"investigation": True}

        # Judge the current spec
        current_spec = self._build_partial_spec()
        judge_result = self.judge.evaluate(
            current_spec,
            self._task
        )

        # Compute reward
        reward_data = self.reward_computer.compute_reward(
            action,
            current_spec.dict(),
            self._task,
            self._state.previous_score
        )

        # Update state
        self._state.previous_score = reward_data["total"]
        self._state.step += 1

        # Check terminal conditions
        done = (
            action.command == AgentGenCommand.SUBMIT or
            self._state.step >= self.max_steps or
            reward_data["total"] >= 10.0
        )

        obs = self._build_observation(reward_data)
        return obs, reward_data["delta"], done, reward_data

    def state(self) -> AgentGenState:
        """Return current state."""
        return self._state

    def _apply_action(self, action: AgentGenAction):
        """Apply action to current spec state."""
        cmd = action.command
        args = action.args

        if cmd == AgentGenCommand.SET_NAME:
            self._state.current_spec["name"] = args.get("name")
        elif cmd == AgentGenCommand.SET_DESCRIPTION:
            self._state.current_spec["description"] = args.get("description")
        elif cmd == AgentGenCommand.ADD_SKILL:
            self._state.current_spec.setdefault("skills", []).append(args.get("skill"))
        # ... handle other commands

    def _investigate(self, action: AgentGenAction) -> Dict[str, Any]:
        """Handle investigation commands."""
        if action.command == AgentGenCommand.CHECK_SCORE:
            # Return current score breakdown without judging
            return self._state.current_score_breakdown
        elif action.command == AgentGenCommand.INSPECT_EXAMPLE:
            # Return an example good agent for this task type
            return self._load_example(self._task.domain)

    def _build_observation(self, reward_data: dict = None) -> AgentGenObservation:
        """Build observation for the agent."""
        return AgentGenObservation(
            task_id=self._state.task_id,
            task_description=self._task.task_description,
            difficulty=self._task.difficulty,
            user_preferences=self._task.user_preferences,
            current_spec=self._state.current_spec,
            score=reward_data.get("total", 0.0) if reward_data else 0.0,
            score_breakdown=reward_data.get("breakdown", {}) if reward_data else {},
            feedback=reward_data.get("feedback", []) if reward_data else [],
            steps_remaining=self._state.max_steps - self._state.step,
            max_steps=self._state.max_steps,
        )
```

---

## Phase 7: GRPO Training (P1) - Week 2

### 7.1 Training Loop - Dual Variants (NEW: Inspired by Bio Env)

**File:** `meta_agent_gym/training/grpo_trainer.py`

```python
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from meta_agent_gym.environment import MetaAgentEnv
from meta_agent_gym.rewards.reward import RewardComputer, RewardConfig

class MetaAgentTrainer:
    """Train policy to generate agent specifications."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-1.7B",
        output_dir: str = "checkpoints",
        env: MetaAgentEnv = None,
        reward_config: RewardConfig = None,
    ):
        self.model_id = model_id
        self.output_dir = output_dir
        self.env = env or MetaAgentEnv()
        self.reward_config = reward_config or RewardConfig()

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def reward_function(
        self,
        prompts: list[str],
        outputs: list[str],
        test_cases: list[dict]
    ) -> list[float]:
        """Compute rewards for generated agents."""
        rewards = []

        for prompt, output, test_case in zip(prompts, outputs, test_cases):
            # Parse output
            try:
                agent_spec = self._parse_to_spec(output)
            except:
                rewards.append(-5.0)  # Invalid output
                continue

            # Run episode
            obs = self.env.reset(test_case["id"])
            done = False
            episode_reward = 0

            while not done:
                # Use parsed spec to get action
                action = self._spec_to_action(agent_spec, obs)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)

        return rewards

    def train(
        self,
        train_dataset,
        num_steps: int = 100,
        learning_rate: float = 5e-5,
    ):
        """Run GRPO training."""
        config = GRPOConfig(
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=4,
            num_generations=8,
            max_steps=num_steps,
        )

        trainer = GRPOTrainer(
            model=self.model,
            reward_function=self.reward_function,
            args=config,
            train_dataset=train_dataset,
        )

        trainer.train()
        return trainer
```

---

### 7.2 Unsloth Variant (NEW: For T4/Consumer GPUs)

**File:** `meta_agent_gym/training/grpo_unsloth.py`

```python
from unsloth import FastLanguageModel
from peft import LoraConfig
from meta_agent_gym.training.grpo_trainer import MetaAgentTrainer

class UnslothMetaAgentTrainer(MetaAgentTrainer):
    """4-bit LoRA variant for mid-range GPUs (T4 Colab)."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-0.6B",
        output_dir: str = "checkpoints/unsloth",
        **kwargs
    ):
        self.output_dir = output_dir

        # Load with Unsloth 4-bit quantization
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=1024,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )

        # Configure LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
```

---

### 7.3 Evaluation & Metrics (NEW: Before/After Table)

**File:** `meta_agent_gym/training/evaluation.py`

```python
from typing import Dict, List
import pandas as pd

class TrainingEvaluator:
    """Evaluate training results and generate metrics."""

    def __init__(self):
        self.metrics = {}

    def evaluate(
        self,
        policy,
        test_cases: List[TestCase],
        num_episodes: int = 20
    ) -> Dict[str, float]:
        """Evaluate policy on test set."""
        results = {
            "mean_reward": 0.0,
            "success_rate": 0.0,
            "mean_length": 0.0,
            "per_difficulty": {},
        }

        # Run episodes
        for difficulty in ["easy", "medium", "hard", "expert"]:
            diff_cases = [tc for tc in test_cases if tc.difficulty == difficulty]
            diff_results = self._evaluate_on_cases(policy, diff_cases)

            results["per_difficulty"][difficulty] = diff_results

        # Aggregate
        results["mean_reward"] = self._aggregate_mean(results["per_difficulty"])
        results["success_rate"] = self._aggregate_success(results["per_difficulty"])

        return results

    def generate_before_after_table(
        self,
        baseline_metrics: Dict,
        trained_metrics: Dict
    ) -> str:
        """Generate before/after metrics table for README."""
        table = "| Metric | Baseline | Trained | Change |\n"
        table += "|--------|----------|---------|--------|\n"

        metrics_to_show = [
            ("Mean Reward", "mean_reward"),
            ("Success Rate", "success_rate"),
            ("Avg Steps", "mean_length"),
        ]

        for label, key in metrics_to_show:
            base = baseline_metrics.get(key, 0)
            trained = trained_metrics.get(key, 0)
            change = ((trained - base) / base * 100) if base > 0 else 0
            table += f"| {label} | {base:.2f} | {trained:.2f} | {change:+.1f}% |\n"

        return table
```

---

## Phase 8: Curriculum & Sub-Agents (P1) - Week 2

### 8.1 Curriculum Controller (NEW: Inspired by Kube SRE Gym)

**File:** `meta_agent_gym/training/curriculum.py`

```python
from typing import List
from meta_agent_gym.core.test_case import TestCase

class CurriculumController:
    """Control difficulty progression during training."""

    def __init__(self, test_cases: List[TestCase]):
        self.test_cases = test_cases
        self.difficulty_levels = ["easy", "medium", "hard", "expert"]
        self.current_level = 0

    def get_next_batch(self, policy_performance: dict, batch_size: int = 4) -> List[TestCase]:
        """Get next batch of test cases based on policy performance."""

        # Check if current level is mastered (>0.7 average score)
        current_cases = [
            tc for tc in self.test_cases
            if tc.difficulty == self.difficulty_levels[self.current_level]
        ]

        level_performance = policy_performance.get(
            self.difficulty_levels[self.current_level],
            0.0
        )

        # Escalate if performing well
        if level_performance > 0.7 and self.current_level < len(self.difficulty_levels) - 1:
            self.current_level += 1

        # Sample from current level + mix of previous
        batch = []
        batch.extend(current_cases[:batch_size // 2])

        # Add some harder cases for exploration
        if self.current_level < len(self.difficulty_levels) - 1:
            next_level = self.difficulty_levels[self.current_level + 1]
            harder_cases = [tc for tc in self.test_cases if tc.difficulty == next_level]
            batch.extend(harder_cases[:batch_size // 2])

        return batch[:batch_size]
```

---

### 8.2 Sub-Agent Roles (NEW: Inspired by Bio Env)

**File:** `meta_agent_gym/agents/subagents.py`

```python
from enum import Enum
from typing import Dict, Any

class SubAgentRole(str, Enum):
    """Specialist roles for sub-agent delegation."""
    STATIC_ANALYZER = "static_analyzer"
    DOC_REVIEWER = "doc_reviewer"
    SECURITY_AUDITOR = "security_auditor"
    TEST_GENERATOR = "test_generator"
    PERFORMANCE_PROFILER = "performance_profiler"

class SubAgentOrchestrator:
    """Orchestrate specialist sub-agents for agent generation."""

    def __init__(self):
        self.roles = {
            SubAgentRole.STATIC_ANALYZER: {
                "skills": ["code-reviewer", "pattern-matcher"],
                "model": "sonnet",
            },
            SubAgentRole.DOC_REVIEWER: {
                "skills": ["file-reader", "report-generator"],
                "model": "haiku",
            },
            SubAgentRole.SECURITY_AUDITOR: {
                "skills": ["code-reviewer", "pattern-matcher"],
                "model": "opus",  # Complex reasoning
            },
        }

    def delegate_to_specialist(
        self,
        role: SubAgentRole,
        task: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delegate task to specialist sub-agent."""
        role_config = self.roles[role]

        # Build specialist prompt
        prompt = self._build_specialist_prompt(role, task, context)

        # In full implementation, this would call Claude API
        # with role-specific system prompt
        return {
            "role": role,
            "recommendation": f"[Specialist {role} analysis would go here]",
            "confidence": 0.8,
        }

    def synthesize_recommendations(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine recommendations from multiple specialists."""
        # Weight by confidence
        return {
            "final_spec": {},
            "confidence": 0.75,
            "specialist_votes": [r["role"] for r in recommendations],
        }
```

---

## File Structure (UPDATED v4)

```
meta_agent_gym/
├── client.py                        # HTTP client for environment
├── models.py                        # Core schemas (Action, Observation, AgentSpec)
├── inference.py                     # Inference utilities
├── conftest.py                      # Pytest configuration
│
├── server/
│   ├── app.py                      # FastAPI server (OpenEnv endpoint)
│   ├── environment.py              # Main OpenEnv environment
│   │
│   ├── verifiers.py                # NEW v4: Hard verifiers (YAML, fields)
│   ├── judge.py                    # Fast judge with calibration
│   │
│   ├── rewards/
│   │   ├── reward.py               # Multi-component reward system
│   │   └── anti_hack.py            # NEW v4: Anti-hacking penalties
│   │
│   ├── rules/
│   │   └── engine.py               # Rule validation engine
│   │
│   ├── tasks/
│   │   ├── scenarios.py            # Test cases (curriculum phases)
│   │   └── generator.py            # Task generation
│   │
│   ├── runtime/
│   │   └── goose.py                # Real execution (steps 3,6,9)
│   │
│   └── adversarial/
│       └── designer.py             # Challenge generator
│
├── training/
│   ├── grpo_trl.py                 # Full GRPO with TRL (H100)
│   ├── grpo_unsloth.py             # 4-bit LoRA variant (T4/Colab)
│   ├── curriculum.py               # Curriculum controller
│   ├── evaluation.py               # Metrics + before/after tables
│   ├── monitoring.py               # Track all reward components
│   ├── reward_backend.py           # Reward computation backend
│   ├── rollout_collection.py       # Data collection
│   └── trajectory.py               # Trajectory handling
│
├── tests/
│   ├── test_smoke.py               # Basic functionality
│   ├── test_reward_quality.py      # Reward component tests
│   ├── test_observation_quality.py # Observation tests
│   └── test_training.py            # Training tests
│
├── data/
│   ├── test_cases/
│   │   ├── easy.json               # Phase 1: 1 skill
│   │   ├── medium.json             # Phase 2: 2-3 skills
│   │   ├── hard.json               # Phase 3: 3-5 skills
│   │   └── expert.json             # Phase 4: 5+ skills
│   └── agents/
│       └── examples/               # Reference good agents
│
├── scripts/
│   └── deploy.sh                   # Deploy to HF Space
│
└── examples/
    └── number_guess/               # Reference environment
```

### NEW v4 Modules

| Module | Purpose | Priority |
|--------|---------|----------|
| `verifiers.py` | Hard YAML/field checks (free, 100%) | P0 |
| `anti_hack.py` | Explicit penalty system | P0 |
| `monitoring.py` | Track all reward components during training | P1 |
| `curriculum.py` | Phase progression controller | P1 |

```
meta_agent_gym/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── agent_spec.py              # P0 - Agent spec schema
│   ├── action.py                  # P0 - Command-based actions (NEW)
│   ├── observation.py             # P0 - POMDP observation (NEW)
│   └── state.py                   # P0 - Environment state
├── core/
│   ├── __init__.py
│   ├── skills.py                  # P0 - Skill registry
│   ├── templates.py               # P0 - Domain templates
│   └── test_case.py               # P0 - Test case schema
├── judge/
│   ├── __init__.py
│   └── judge.py                   # P0 - Claude judge with calibration
├── runtime/
│   ├── __init__.py
│   └── goose.py                   # P0 - Goose integration
├── rewards/
│   ├── __init__.py
│   ├── reward.py                  # P0 - Decomposed reward (NEW)
│   └── justification.md           # P0 - Reward docs (NEW)
├── adversarial/
│   ├── __init__.py
│   └── designer.py                # P1 - Challenge generator
├── agents/
│   ├── __init__.py
│   └── subagents.py               # P1 - Sub-agent roles (NEW)
├── training/
│   ├── __init__.py
│   ├── grpo_trainer.py            # P1 - Full GRPO (H100)
│   ├── grpo_unsloth.py            # P1 - 4-bit LoRA (T4) (NEW)
│   ├── curriculum.py              # P1 - Curriculum controller (NEW)
│   └── evaluation.py              # P1 - Metrics + before/after (NEW)
└── environment.py                 # P0 - OpenEnv environment

data/
├── test_cases/
│   ├── generic.json               # P0 - Base test cases
│   └── adversarial.json           # P1 - Generated challenges
└── agents/
    └── examples/                  # P0 - Example good agents

tests/
├── test_agent_spec.py             # P0
├── test_action.py                 # P0 (NEW)
├── test_observation.py            # P0 (NEW)
├── test_reward.py                 # P0 (NEW)
├── test_judge.py                  # P0
├── test_goose_runner.py           # P0
├── test_curriculum.py             # P1 (NEW)
└── test_training.py               # P1
```

---

## Implementation Order (UPDATED v4)

### Week 1: P0 Critical Path

| Day | Component | Output | Notes |
|-----|-----------|--------|-------|
| **Day 1-2** | | | |
| 1 | Hard Verifiers | `server/verifiers.py` | YAML, fields, format checks |
| 2 | AGENT.md Schema | `models.py` (extend) | `AgentSpec` with `to_markdown()` |
| 3 | Action Commands | `models.py` (extend) | Command-based actions |
| 4 | Observation | `models.py` (extend) | POMDP structure |
| **Day 2-3** | | | |
| 5 | Skill Registry | `server/skills.py` | Single-skill tasks first |
| 6 | Test Cases (Easy) | `server/tasks/scenarios.py` | Phase 1: 1 skill tasks |
| 7 | Fast Judge | `server/judge.py` | Claude Sonnet, 5-dim scoring |
| **Day 3-4** | | | |
| 8 | Multi-Component Reward | `server/rewards/reward.py` | Hard verifiers + judge + penalties |
| **Day 4-5** | | | |
| 9 | OpenEnv Environment | `server/environment.py` | 2-tier eval integration |
| 10 | Goose Runner | `server/runtime/goose.py` | Steps 3, 6, 9 execution |
| 11 | Calibration Tracker | `server/judge.py` | Fast vs real drift detection |
| **Day 5** | | | |
| 12 | Unit Tests | `tests/test_*.py` | All P0 components |
| 13 | **Deploy to HF** | `scripts/deploy.sh` | **Early deployment!** |

### Week 2: P1 MVP Enhancement

| Day | Component | Output | Notes |
|-----|-----------|--------|-------|
| **Day 1-2** | | | |
| 14 | GRPO Trainer (H100) | `training/grpo_trl.py` | Full GRPO loop |
| 15 | Unsloth Trainer (T4) | `training/grpo_unsloth.py` | 4-bit LoRA variant |
| **Day 3** | | | |
| 16 | Curriculum Controller | `training/curriculum.py` | 1→2→3→5+ progression |
| 17 | Medium/Hard Tests | `server/tasks/scenarios.py` | Phases 2-4 |
| **Day 3-4** | | | |
| 18 | Adversarial Designer | `server/adversarial.py` | Challenge generator |
| 19 | Anti-Hacking Checks | `server/rewards/anti_hack.py` | Explicit penalties |
| **Day 4-5** | | | |
| 20 | Monitoring Dashboard | `training/monitoring.py` | Track all components |
| 21 | Evaluator | `training/evaluation.py` | Before/after metrics |
| 22 | Integration Tests | `tests/test_training.py` | End-to-end pipeline |

### Week 3+: P2 Polish

| Day | Component | Output | Notes |
|-----|-----------|--------|-------|
| 23 | Investigation Tools | `server/environment.py` | `check_score`, `inspect_example` |
| 24 | More Test Cases | `server/tasks/scenarios.py` | 20+ across domains |
| 25 | Documentation | README.md | Full guide |
| 26 | Demo Video | `demo/` | 60-90s walkthrough |

---

## Success Metrics (UPDATED v4)

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Hard verifier pass rate** | 100% (gated) | YAML parse, fields present |
| **Judge score** | >7.0/10 | Average on test set |
| **Goose success rate** | >80% | Agents complete test cases |
| **Training convergence** | Reward increases | Over 100 steps |
| **Generation time** | <30 seconds | From task to AGENT.md |
| **Valid YAML** | 100% | All generated specs parse correctly |
| **Reward variance** | std > 0.1×mean | GRPO-friendly |
| **Calibration drift** | < 1.0 | Fast vs real reward correlation |
| **Before/after gain** | >50% improvement | Trained vs baseline |
| **Anti-hack effectiveness** | 0 exploits | No empty/over-engineered specs win |

### RLVR-Specific Metrics (NEW v4)

| Metric | Target | Why |
|--------|--------|-----|
| Hard verifier gate rate | <5% episodes | Should catch format errors early |
| Fast judge correlation | >0.8 with real | Judge should predict real outcomes |
| Real execution frequency | 10% (steps 3,6,9) | Balance speed vs validation |
| Regression rate | <2% per episode | Policy shouldn't break progress |

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Judge score | >7.0/10 | Average on test set |
| Goose success rate | >80% | Agents complete test cases |
| Training convergence | Reward increases | Over 100 steps |
| Generation time | <30 seconds | From task to AGENT.md |
| Valid YAML | 100% | All generated specs parse correctly |
| **Reward variance** | std > 0.1×mean | GRPO-friendly (NEW) |
| **Calibration drift** | < 1.0 | Fast vs real reward correlation (NEW) |
| **Before/after gain** | >50% improvement | Trained vs baseline (NEW) |

---

## README Sections (UPDATED v4: Theme Alignment)

### Hackathon Theme Alignment

**🥇 Theme #4 - Self-Improvement (Primary)**

| Theme Requirement | Our Implementation |
|-------------------|---------------------|
| Generate new challenges | Adversarial Designer creates test cases targeting policy weaknesses |
| Escalate difficulty | Curriculum: 1 skill → 2-3 → 3-5 → 5+ skills based on performance |
| Adaptive curricula | CurriculumController progresses when policy achieves >70% on current level |
| Recursive skill amplification | Policy learns to design agents; adversarial designer makes it harder |

**🥈 Theme #3.1 - World Modeling (Secondary)**

| Theme Requirement | Our Implementation |
|-------------------|---------------------|
| Maintain consistent internal state | POMDP structure — hidden state = "what makes a good agent?" |
| Partial observability | Current spec state is partial; full quality only revealed after submit |
| Update beliefs based on outcomes | Investigation commands (`check_score`, `inspect_example`) + feedback loop |
| Multi-step workflows | Command-based actions: set_name → add_skill → write_prompt → submit |
| Real hard work, not shortcuts | Anti-hacking penalties prevent format-only exploits |

**🥉 Theme #2 - Long-Horizon Planning (Tertiary)**

| Theme Requirement | Our Implementation |
|-------------------|---------------------|
| Multi-step reasoning | 7-step generation process with interdependent decisions |
| Sparse/delayed rewards | Reward only meaningful after submit; per-step rewards are incremental |
| Decompose goals | Policy must decompose "build an agent" into discrete commands |
| Recover from early mistakes | Investigation tools + regression penalties prevent breaking progress |

---

### Evaluation Criteria Alignment

| Criterion | Our Approach |
|-----------|--------------|
| Real-world utility | Generates production-ready AGENT.md files for Claude Code, Copilot, etc. |
| Task quality | 4 difficulty levels with decomposed reward + red herrings |
| Environment design | POMDP + command-based actions + investigation tools |
| Code quality | Full OpenEnv spec compliance + 100+ tests |
| Novelty | Meta-learning for agent design (no existing env like this) |

---

### Hackathon Track Alignment

| Track | How This Fits |
|-------|---------------|
| **Multi-Agent** | Sub-agent specialist roles (static analyzer, doc reviewer, security auditor) |
| **World Modeling** | POMDP structure with hidden "what makes a good agent" state |
| **Long-Horizon Planning** | Multi-step agent generation with investigation commands |
| **Self-Improvement** | Adversarial designer generates harder tests based on policy weaknesses |

### Evaluation Criteria Alignment

| Criterion | Our Approach |
|-----------|--------------|
| Real-world utility | Generates production-ready AGENT.md files for Claude Code, Copilot, etc. |
| Task quality | 4 difficulty levels with decomposed reward + red herrings |
| Environment design | POMDP + command-based actions + investigation tools |
| Code quality | Full OpenEnv spec compliance + 100+ tests |
| Novelty | Meta-learning for agent design (no existing env like this) |

---

## Next Steps

**Start with P0 components (Week 1):**

1. **Today:** Models (AgentSpec, Action, Observation, State) + command-based actions
2. **Tomorrow:** Skill registry + templates
3. **Day 3:** Judge system with calibration
4. **Day 4:** Decomposed reward system with justification docs
5. **Day 5:** OpenEnv environment + end-to-end dry run

**Each P0 component should be:**
- Fully implemented
- Unit tested
- Documented
- Integrated with others

---

## v4 Summary of Changes

### RLVR (Reinforcement Learning with Verifiable Rewards)

The plan now follows RLVR best practices:

1. **Multiple independent reward functions** - Track ALL components separately
2. **Hard verifiers** - YAML parse, field checks (100% of steps, free)
3. **Fast judge** - Claude Sonnet quality scoring (90% of steps)
4. **Real execution** - Goose runtime validation (10% at steps 3, 6, 9)
5. **Anti-hacking** - Explicit penalties for common exploits
6. **Curriculum** - 1-skill → 2-3 skills → 3-5 skills → 5+ skills
7. **Early deployment** - Deploy to HF Space Day 5 (P0)

### What Changed from v3

| Component | v3 | v4 |
|-----------|----|----|
| Verification | Judge only | 3-tier: Hard → Judge → Real |
| Reward | Single decomposed | Multiple independent (RLVR) |
| Anti-hack | Red herrings | Explicit penalties |
| Curriculum | Generic | Explicit phase progression |
| Deployment | End of project | P0 Day 5 (early!) |

---

*Version: 4.0*
*Updated: 2025-04-23*
*Based on: PyTorch OpenEnv Hackathon + RLVR Best Practices*

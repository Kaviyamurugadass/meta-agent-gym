# meta-agent-gym: Implementation Plan v2

> **Status:** Active Development
> **Last Updated:** 2025-04-23
> **Focus:** Core Components First (UI Later)

---

## Executive Summary

**Goal:** Build a system that generates complete AGENT.md files from user task descriptions.

**Input:** Task description + preferences (language, constraints, etc.)
**Output:** Complete AGENT.md file that works everywhere (Claude Code, Goose, Copilot, etc.)

**Testing:** Three-layer validation (Dataset → Judge (Claude) → Goose execution)

---

## System Architecture

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
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    TESTING LAYER                                 │
│                                                                  │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐     │
│   │  Dataset    │──▶│  Judge      │──▶│    Goose        │     │
│   │             │   │  (Claude)   │   │  (Real Runner)  │     │
│   │ Test cases  │   │  Scores     │   │  Exec & Metrics │     │
│   └─────────────┘   └─────────────┘   └─────────────────┘     │
│                                                                  │
│            All three feed into GRPO reward signal                │
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
from typing import List, Optional, Literal
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

**Example output:**
```markdown
---
name: price-scraper
description: Extract prices from e-commerce sites. Use proactively when gathering pricing data.
user-invocable: true
skills:
  - web-scraping
  - price-parser
  - data-validator
---

You are a price scraping specialist.

Workflow:
1. Identify the e-commerce platform structure
2. Use /web-scraping to extract product data
3. Use /price-parser to extract pricing information
4. Use /data-validator to verify data integrity

Always handle rate limiting and respect robots.txt.
```

**Tests:** `tests/test_agent_spec.py`
- Test YAML generation
- Test skill list handling
- Test model selection
- Test roundtrip (parse → generate)

---

### 1.2 Skill Registry

**File:** `meta_agent_gym/core/skills.py`

```python
# Common skills from skills.sh
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
```

---

### 1.3 Domain Templates

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
```

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
    ]
  },
  
  "data_analysis_easy": {
    "id": "da_easy_001",
    "domain": "data",
    "difficulty": "easy",
    "task_description": "Build an agent that analyzes CSV data and generates summary statistics",
    "user_preferences": {
      "language": "python"
    },
    "required_skills": ["csv-handler", "data-analyzer"],
    "required_model": "sonnet",
    "test_input": {
      "file": "sales_data.csv",
      "analysis": ["mean", "median", "std"]
    },
    "expected_output": {
      "format": "report"
    },
    "validation_checks": [
      "Has csv-handler skill",
      "Handles missing data",
      "Generates readable report"
    ]
  }
}
```

### 2.3 Adversarial Test Cases

**File:** `meta_agent_gym/core/adversarial.py`

```python
class AdversarialDesigner:
    """Creates challenging test cases to push agent quality."""
    
    CHALLENGE_TYPES = [
        "edge_cases",        # Unusual inputs
        "resource_constraints",  # Low token budget
        "ambiguous_requirements",  # Unclear task
        "multi_step",        # Complex workflows
        "error_scenarios",   # Error handling required
        "performance",       # Speed/cost optimization
        "security",          # Security-sensitive tasks
    ]
    
    def generate_adversarial_case(self, base_case: TestCase, challenge: str) -> TestCase:
        """Generate a harder variant of a test case."""
        # Implementation varies by challenge type
        pass
```

---

## Phase 3: Judge System (P0) - Week 1

### 3.1 Judge Schema

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
    violations: List[str]

class AgentJudge:
    """Judge agent specifications using Claude."""
    
    # Scoring weights
    WEIGHTS = {
        "skill_selection": 3.0,
        "description_quality": 2.0,
        "workflow_clarity": 2.0,
        "model_appropriateness": 1.5,
        "best_practices": 1.5,
    }
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    def evaluate(
        self,
        agent_spec: AgentSpec,
        test_case: TestCase
    ) -> JudgeResult:
        """Evaluate an agent spec against a test case."""
        
        prompt = self._build_prompt(agent_spec, test_case)
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_response(response.content[0].text)
    
    def _build_prompt(self, spec: AgentSpec, test_case: TestCase) -> str:
        return f"""You are an expert agent architect evaluating an agent specification.

TASK: {test_case.task_description}

REQUIRED SKILLS: {', '.join(test_case.required_skills)}

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

Respond in JSON:
{{
  "skill_selection": {{"score": 0-2, "reasoning": "..."}},
  "description_quality": {{"score": 0-2, "reasoning": "..."}},
  "workflow_clarity": {{"score": 0-2, "reasoning": "..."}},
  "model_appropriateness": {{"score": 0-2, "reasoning": "..."}},
  "best_practices": {{"score": 0-2, "reasoning": "..."}},
  "overall_feedback": "..."
}}
"""
```

---

## Phase 4: Goose Integration (P0) - Week 1

### 4.1 Goose Runner

**File:** `meta_agent_gym/runner/goose.py`

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

## Phase 5: GRPO Training (P1) - Week 2

### 5.1 Training Loop

**File:** `meta_agent_gym/training/grpo_trainer.py`

```python
from trl import GRPOTrainer, GRPOConfig
from meta_agent_gym.models.agent_spec import AgentSpec
from meta_agent_gym.judge.judge import AgentJudge
from meta_agent_gym.runner.goose import GooseRunner

class MetaAgentTrainer:
    """Train policy to generate agent specifications."""
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-1.7B",
        output_dir: str = "checkpoints",
    ):
        self.model_id = model_id
        self.output_dir = output_dir
        self.judge = AgentJudge()
        self.goose = GooseRunner()
    
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
                agent_spec = AgentSpec.from_markdown(output)
            except:
                rewards.append(-5.0)  # Invalid output
                continue
            
            # Judge score (40%)
            judge_result = self.judge.evaluate(agent_spec, test_case)
            judge_score = judge_result.score / 10.0  # Normalize to 0-1
            
            # Goose execution (40%)
            try:
                goose_result = self.goose.run(agent_spec, test_case["test_input"])
                execution_score = 1.0 if goose_result["success"] else 0.0
            except:
                execution_score = 0.0
            
            # Efficiency (20%)
            efficiency = self._compute_efficiency(agent_spec)
            
            # Combined reward
            reward = (
                judge_score * 0.4 +
                execution_score * 0.4 +
                efficiency * 0.2
            )
            
            rewards.append(reward)
        
        return rewards
    
    def train(self, train_dataset, num_steps: int = 100):
        """Run GRPO training."""
        config = GRPOConfig(
            output_dir=self.output_dir,
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            num_generations=8,
            max_steps=num_steps,
        )
        
        trainer = GRPOTrainer(
            model=self.model_id,
            reward_function=self.reward_function,
            args=config,
            train_dataset=train_dataset,
        )
        
        trainer.train()
        return trainer
```

---

## Phase 6: Adversarial Designer (P1) - Week 2

### 6.1 Challenge Generator

**File:** `meta_agent_gym/adversarial/designer.py`

```python
from anthropic import Anthropic
from meta_agent_gym.core.test_case import TestCase

class AdversarialDesigner:
    """Creates challenging test cases using Claude."""
    
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

Respond in JSON format:
[
  {{
    "task_description": "...",
    "difficulty": "medium/hard/expert",
    "challenge_type": "{weak_areas[0]}",
    "required_skills": [...],
    "test_input": {{...}},
    "validation_checks": [...]
  }}
]"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse and return test cases
        return self._parse_challenges(response.content[0].text)
```

---

## File Structure

```
meta_agent_gym/
├── __init__.py
├── models/
│   ├── __init__.py
│   └── agent_spec.py              # P0 - Agent spec schema
├── core/
│   ├── __init__.py
│   ├── skills.py                  # P0 - Skill registry
│   ├── templates.py               # P0 - Domain templates
│   └── test_case.py               # P0 - Test case schema
├── judge/
│   ├── __init__.py
│   └── judge.py                   # P0 - Claude judge
├── runner/
│   ├── __init__.py
│   └── goose.py                   # P0 - Goose integration
├── adversarial/
│   ├── __init__.py
│   └── designer.py                # P1 - Challenge generator
└── training/
    ├── __init__.py
    └── grpo_trainer.py            # P1 - Training loop

data/
├── test_cases/
│   ├── generic.json               # P0 - Base test cases
│   └── adversarial.json           # P1 - Generated challenges
└── agents/
    └── examples/                  # P0 - Example good agents

tests/
├── test_agent_spec.py             # P0
├── test_judge.py                  # P0
├── test_goose_runner.py           # P0
└── test_training.py               # P1
```

---

## Implementation Order

| Week | Component | Priority | Deliverable |
|------|-----------|----------|-------------|
| **Week 1** | | | |
| Day 1-2 | AGENT.md Schema | P0 | Complete spec model |
| Day 2-3 | Skill Registry | P0 | Common skills catalog |
| Day 3-4 | Judge (Claude) | P0 | Evaluation system |
| Day 4-5 | Goose Runner | P0 | Execution integration |
| Day 5 | Dataset (Generic) | P0 | Base test cases |
| **Week 2** | | | |
| Day 1-2 | GRPO Training | P1 | Basic training loop |
| Day 3-4 | Adversarial Designer | P1 | Challenge generator |
| Day 5 | End-to-end test | P1 | Full pipeline working |
| **Week 3+** | | | |
| Later | UI | P3 | User interface |
| Later | Optimization | P3 | Performance tuning |
| Later | Publishing | P2 | skills.sh integration |

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Judge score | >7.0/10 | Average on test set |
| Goose success rate | >80% | Agents complete test cases |
| Training convergence | Reward increases | Over 100 steps |
| Generation time | <30 seconds | From task to AGENT.md |
| Valid YAML | 100% | All generated specs parse correctly |

---

## Next Steps

**Start with P0 components (Week 1):**

1. **Today:** AGENT.md schema + skill registry
2. **Tomorrow:** Judge system with Claude
3. **Day 3:** Goose runner integration
4. **Day 4:** Generic test dataset
5. **Day 5:** End-to-end dry run

**Each P0 component should be:**
- Fully implemented
- Unit tested
- Documented
- Integrated with others

---

*Version: 2.0*
*Updated: 2025-04-23*
*Focus: Core Components (UI Deferred)*

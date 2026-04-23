"""Enhanced reward system for onsite training phase.

Key improvements:
1. Better component weighting based on training insights
2. Enhanced anti-hacking detection
3. Improved progress rewards for better GRPO signal
4. Stronger best_practices and model_appropriateness scoring
5. Edge case handling for robust onsite performance
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Set

from models import (
    Action,
    ActionCommand,
    AgentSpec,
    ModelType,
    RewardConfig,
    RewardMode,
    RuleViolation,
    State,
    TaskSpec,
)

try:
    from server.rewards.reward import MetaAgentRewardComputer
except ImportError:
    MetaAgentRewardComputer = None  # type: ignore[assignment,misc]

logger = logging.getLogger("server.rewards.enhanced_reward")


class EnhancedRewardComputer(MetaAgentRewardComputer):
    """Enhanced reward computer with improved scoring for onsite phase.
    
    Improvements based on training analysis:
    - Stronger best_practices detection (was weakest component)
    - Better model appropriateness scoring
    - Enhanced anti-hacking penalties
    - More granular progress rewards
    - Edge case robustness
    """

    def __init__(self, config: RewardConfig) -> None:
        super().__init__(config)
        # Enhanced component weights based on training gaps
        self.enhanced_weights = {
            "skill_selection": 0.25,
            "description_quality": 0.20,
            "workflow_clarity": 0.20,
            "model_appropriateness": 0.15,  # Increased from 0.15
            "best_practices": 0.15,  # Increased from 0.10
            "efficiency": 0.05,  # Decreased to focus on core components
        }
        
        # Enhanced anti-hacking thresholds
        self.empty_spec_patterns = [
            r"^.{0,49}$",  # Very short prompts
            r"^(test|placeholder|todo|fix)$",  # Common empty patterns
            r"^(.{1,10}\s+){5,}$",  # Repeated short words
        ]
        
        # Over-engineering patterns
        self.over_engineering_patterns = [
            r"(selenium|javascript|rendering)",  # When not needed
            r"(machine learning|ml|ai model)",  # When task is simple
            r"(database|sql|mongodb)",  # When file-based is sufficient
        ]

    def _judge_component_rewards(
        self,
        spec: dict,
        task: TaskSpec,
        action: Action,
    ) -> dict[str, float]:
        """Enhanced component scoring with better detection."""
        
        # Skip judge on investigation commands
        if action.command in [ActionCommand.CHECK_SCORE, ActionCommand.INSPECT_EXAMPLE]:
            return {}

        return {
            "skill_selection": self._enhanced_score_skill_selection(spec, task),
            "description_quality": self._enhanced_score_description(spec, task),
            "workflow_clarity": self._enhanced_score_workflow(spec, task),
            "model_appropriateness": self._enhanced_score_model(spec, task),
            "best_practices": self._enhanced_score_best_practices(spec, task),
            "efficiency": self._enhanced_score_efficiency(spec, task),
        }

    def _enhanced_score_skill_selection(self, spec: dict, task: TaskSpec) -> float:
        """Enhanced skill selection with red herring detection."""
        required = set(task.required_skills)
        recommended = set(task.recommended_skills)
        has = set(spec.get("skills", []))
        
        if not required:
            return 1.0

        # Coverage scoring
        required_coverage = len(required & has) / len(required)
        recommended_bonus = len(recommended & has) / len(recommended) if recommended else 0
        
        # Red herring detection - penalize adding skills explicitly warned against
        red_herrings = task.red_herrings or []
        red_herring_penalty = 0.0
        for warning in red_herrings:
            # Extract skill names from warnings
            for skill in has:
                if skill.lower() in warning.lower():
                    red_herring_penalty += 0.3
        
        # Over-engineering penalty (more aggressive)
        extra = len(has - required - recommended)
        extra_penalty = min(extra * 0.15, 0.5)  # Increased penalty
        
        base_score = max(0.0, required_coverage + (recommended_bonus * 0.3))
        return max(0.0, base_score - extra_penalty - red_herring_penalty)

    def _enhanced_score_description(self, spec: dict, task: TaskSpec) -> float:
        """Enhanced description scoring with domain awareness."""
        desc = spec.get("description", "")
        
        # Basic checks
        if len(desc) < 10:
            return 0.0
            
        score = 0.0
        
        # Delegation keywords (expanded)
        delegation_words = [
            "proactively", "use", "when", "specialist", "expert", "handles",
            "autonomously", "independently", "without supervision"
        ]
        has_delegation = sum(1 for word in delegation_words if word in desc.lower())
        score += min(0.4, has_delegation * 0.1)
        
        # Length appropriateness (task-dependent)
        good_length = self._check_appropriate_length(desc, task)
        score += 0.3 * good_length
        
        # Domain-specific keywords
        domain_keywords = self._get_domain_keywords(task.domain)
        domain_matches = sum(1 for kw in domain_keywords if kw in desc.lower())
        score += min(0.3, domain_matches * 0.1)
        
        return min(1.0, score)

    def _enhanced_score_workflow(self, spec: dict, task: TaskSpec) -> float:
        """Enhanced workflow scoring with step detection."""
        prompt = spec.get("system_prompt", "")
        
        if len(prompt) < 20:
            return 0.0
            
        score = 0.0
        
        # Step indicators (enhanced patterns)
        step_patterns = [
            r"^\d+\.",  # Numbered lists
            r"step\s+\d+",  # "Step 1", "Step 2"
            r"(first|then|next|finally)",  # Sequence words
            r"(workflow|process|procedure)",  # Process keywords
        ]
        
        step_matches = sum(1 for pattern in step_patterns if re.search(pattern, prompt, re.IGNORECASE))
        score += min(0.5, step_matches * 0.15)
        
        # Error handling mentions
        error_patterns = [
            r"error", r"exception", r"fail", r"invalid", r"missing",
            r"handle", r"catch", r"validate", r"check"
        ]
        error_matches = sum(1 for pattern in error_patterns if re.search(pattern, prompt, re.IGNORECASE))
        score += min(0.3, error_matches * 0.05)
        
        # Output specification
        output_patterns = [
            r"return", r"output", r"generate", r"produce", r"create"
        ]
        output_matches = sum(1 for pattern in output_patterns if re.search(pattern, prompt, re.IGNORECASE))
        score += min(0.2, output_matches * 0.05)
        
        return min(1.0, score)

    def _enhanced_score_model(self, spec: dict, task: TaskSpec) -> float:
        """Enhanced model appropriateness with complexity detection."""
        model = spec.get("model", "sonnet")
        
        # Enhanced difficulty mapping
        complexity_indicators = self._assess_task_complexity(task)
        
        # Base recommendations
        if complexity_indicators["is_simple"]:
            recommended = ModelType.HAIKU
            acceptable = [ModelType.HAIKU, ModelType.SONNET]
        elif complexity_indicators["is_medium"]:
            recommended = ModelType.SONNET
            acceptable = [ModelType.SONNET, ModelType.HAIKU]
        elif complexity_indicators["is_hard"]:
            recommended = ModelType.SONNET
            acceptable = [ModelType.SONNET, ModelType.OPUS]
        else:  # expert
            recommended = ModelType.OPUS
            acceptable = [ModelType.OPUS, ModelType.SONNET]
        
        # Scoring
        if model == recommended.value:
            return 1.0
        elif model in [acc.value for acc in acceptable]:
            return 0.8
        else:
            # Wrong tier
            if model == ModelType.OPUS.value and complexity_indicators["is_simple"]:
                return 0.3  # Significant overkill
            elif model == ModelType.HAIKU.value and complexity_indicators["is_expert"]:
                return 0.2  # Under-powered
            else:
                return 0.5  # Mismatch

    def _enhanced_score_best_practices(self, spec: dict) -> float:
        """Enhanced best practices detection (was weakest component)."""
        prompt = spec.get("system_prompt", "")
        
        if len(prompt) < 20:
            return 0.0
            
        score = 0.0
        
        # Error handling (expanded)
        error_patterns = [
            r"try\s*:", r"except", r"handle error", r"catch exception",
            r"error handling", r"graceful degradation", r"fallback"
        ]
        error_matches = sum(1 for pattern in error_patterns if re.search(pattern, prompt, re.IGNORECASE))
        score += min(0.3, error_matches * 0.08)
        
        # Validation patterns
        validation_patterns = [
            r"validate", r"verify", r"check", r"ensure", r"confirm",
            r"sanitiz", r"escape", r"proper format"
        ]
        validation_matches = sum(1 for pattern in validation_patterns if re.search(pattern, prompt, re.IGNORECASE))
        score += min(0.25, validation_matches * 0.06)
        
        # Security patterns
        security_patterns = [
            r"secure", r"safely", r"protect", r"authenticate", r"authorize",
            r"sanitize input", r"sql injection", r"xss"
        ]
        security_matches = sum(1 for pattern in security_patterns if re.search(pattern, prompt, re.IGNORECASE))
        score += min(0.2, security_matches * 0.07)
        
        # Performance patterns
        performance_patterns = [
            r"efficient", r"optimize", r"cache", r"batch", r"stream",
            r"memory", r"performance", r"fast"
        ]
        performance_matches = sum(1 for pattern in performance_patterns if re.search(pattern, prompt, re.IGNORECASE))
        score += min(0.15, performance_matches * 0.05)
        
        # Documentation patterns
        doc_patterns = [
            r"document", r"comment", r"explain", r"clear", r"readable",
            r"maintainable", r"understandable"
        ]
        doc_matches = sum(1 for pattern in doc_patterns if re.search(pattern, prompt, re.IGNORECASE))
        score += min(0.1, doc_matches * 0.03)
        
        return min(1.0, score)

    def _enhanced_score_efficiency(self, spec: dict) -> float:
        """Enhanced efficiency scoring with context awareness."""
        skills = spec.get("skills", [])
        skill_count = len(skills)
        
        # Dynamic efficiency based on task complexity
        if skill_count <= 2:
            return 1.0
        elif skill_count <= 4:
            return 0.9
        elif skill_count <= 6:
            return 0.7
        elif skill_count <= 8:
            return 0.5
        else:
            return 0.2  # Too many skills

    def _enhanced_anti_hack_penalties(self, spec: dict, action: Action) -> dict[str, float]:
        """Enhanced anti-hacking detection."""
        penalties: dict[str, float] = {}
        
        # Empty spec detection (enhanced patterns)
        prompt = spec.get("system_prompt", "")
        if any(re.search(pattern, prompt) for pattern in self.empty_spec_patterns):
            penalties["empty_spec"] = self.config.anti_hack_empty_spec
        
        # Over-engineering detection
        full_spec_text = " ".join([
            spec.get("name", ""),
            spec.get("description", ""),
            prompt,
            " ".join(spec.get("skills", []))
        ])
        
        if any(re.search(pattern, full_spec_text, re.IGNORECASE) for pattern in self.over_engineering_patterns):
            penalties["over_engineered"] = self.config.anti_hack_over_engineered
        
        # Repetitive action detection
        if hasattr(self, '_last_command') and self._last_command == action.command:
            if action.command not in [ActionCommand.NOOP, ActionCommand.CHECK_SCORE]:
                penalties["repetitive_action"] = -0.3
        
        self._last_command = action.command
        
        return penalties

    def _enhanced_progress_reward(self, spec: dict, action: Action) -> float:
        """Enhanced progress rewards for better GRPO signal."""
        base_progress = 0.1
        
        # Meaningful action bonuses
        action_bonuses = {
            ActionCommand.SET_NAME: 0.25,
            ActionCommand.SET_DESCRIPTION: 0.35,
            ActionCommand.ADD_SKILL: 0.2,
            ActionCommand.REMOVE_SKILL: 0.15,  # Good for optimization
            ActionCommand.SET_MODEL: 0.15,
            ActionCommand.WRITE_PROMPT: 0.4,
            ActionCommand.ADD_TOOLS: 0.1,
            ActionCommand.SET_MEMORY: 0.1,
            ActionCommand.SET_MAX_TURNS: 0.05,
        }
        
        progress = base_progress + action_bonuses.get(action.command, 0.0)
        
        # Quality bonuses for prompt writing
        if action.command == ActionCommand.WRITE_PROMPT:
            prompt_len = len(spec.get("system_prompt", ""))
            if prompt_len >= 100:
                progress += 0.3
            elif prompt_len >= 50:
                progress += 0.15
        
        # Completion bonus
        if action.command == ActionCommand.SUBMIT:
            required_fields = ["name", "description", "system_prompt"]
            if all(spec.get(field) for field in required_fields):
                progress += 0.5
        
        return progress

    # Helper methods
    def _check_appropriate_length(self, desc: str, task: TaskSpec) -> float:
        """Check if description length is appropriate for task."""
        word_count = len(desc.split())
        
        if task.difficulty == "easy":
            return 1.0 if 10 <= word_count <= 30 else 0.5
        elif task.difficulty == "medium":
            return 1.0 if 20 <= word_count <= 50 else 0.5
        elif task.difficulty == "hard":
            return 1.0 if 30 <= word_count <= 70 else 0.5
        else:  # expert
            return 1.0 if 40 <= word_count <= 100 else 0.5

    def _get_domain_keywords(self, domain: str) -> List[str]:
        """Get domain-specific keywords for description scoring."""
        domain_keywords = {
            "web": ["scraping", "html", "css", "javascript", "http", "api", "website"],
            "data": ["csv", "json", "data", "analysis", "transform", "validate", "aggregate"],
            "code": ["review", "debug", "test", "fix", "refactor", "optimize", "security"],
            "files": ["read", "write", "file", "directory", "path", "format", "backup"],
            "analysis": ["analyze", "pattern", "trend", "report", "insight", "metric"],
            "output": ["report", "generate", "format", "present", "visualize", "summary"],
        }
        return domain_keywords.get(domain, [])

    def _assess_task_complexity(self, task: TaskSpec) -> Dict[str, bool]:
        """Assess task complexity for model selection."""
        required_skills = len(task.required_skills)
        max_steps = task.max_steps
        
        complexity = {
            "is_simple": task.difficulty == "easy" and required_skills <= 1 and max_steps <= 7,
            "is_medium": task.difficulty == "medium" and 2 <= required_skills <= 3 and max_steps <= 10,
            "is_hard": task.difficulty == "hard" and 3 <= required_skills <= 5 and max_steps <= 15,
            "is_expert": task.difficulty == "expert" and required_skills >= 5 and max_steps >= 15,
        }
        
        return complexity

    def compute(
        self,
        action: Action,
        state: State,
        task: TaskSpec,
        violations: list[RuleViolation],
    ) -> float:
        """Use enhanced scoring with original computation flow."""
        
        # Use enhanced weights
        original_weights = self.config.component_weights
        self.config.component_weights = self.enhanced_weights
        
        try:
            # Use enhanced progress rewards
            original_progress = self._progress_reward
            self._progress_reward = self._enhanced_progress_reward
            
            # Use enhanced anti-hack penalties
            original_anti_hack = self._anti_hack_penalties
            self._anti_hack_penalties = self._enhanced_anti_hack_penalties
            
            # Call parent compute with enhanced methods
            result = super().compute(action, state, task, violations)
            
            return result
            
        finally:
            # Restore original methods
            self.config.component_weights = original_weights
            self._progress_reward = original_progress
            self._anti_hack_penalties = original_anti_hack

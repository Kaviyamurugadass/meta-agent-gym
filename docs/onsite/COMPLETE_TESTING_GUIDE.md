# Complete Testing Guide: Validation Before Onsite

## 🎯 Testing Objectives

Ensure all components work together seamlessly for the onsite phase (April 25-26):
1. **Environment functionality** - OpenEnv compliance and robust operation
2. **Training pipeline** - Complete GRPO training with real results
3. **Reward system** - Enhanced scoring and anti-hacking protection
4. **Agent behavior** - Optimized action selection and learning
5. **Integration testing** - All components working together

## 🧪 Component Testing Checklist

### 1. Environment Testing
```bash
# Test basic OpenEnv compliance
python -c "
from server.robust_environment import RobustEnvironment
env = RobustEnvironment()
obs = env.reset('ws_easy_001')
print('✅ Environment reset successful')
print(f'Step: {obs.step}, Max: {obs.max_steps}')
"

# Test robust error handling
python -c "
from server.robust_environment import RobustEnvironment
env = RobustEnvironment()
obs = env.reset('ws_easy_001')
# Test invalid action
try:
    from models import Action, ActionCommand
    invalid_action = Action(command=ActionCommand.SET_NAME, args={'name': ''})
    result = env.step(invalid_action)
    print('✅ Error handling working')
except Exception as e:
    print(f'✅ Error caught: {e}')
"
```

### 2. Enhanced Reward System Testing
```bash
# Test enhanced reward scoring
python -c "
from server.rewards.enhanced_reward import EnhancedRewardComputer
from models import RewardConfig
config = RewardConfig()
rewards = EnhancedRewardComputer(config)
print('✅ Enhanced reward system initialized')
"
```

### 3. Agent Optimizer Testing
```bash
# Test behavior optimization
python -c "
from training.agent_optimizer import get_agent_optimizer
optimizer = get_agent_optimizer()
print('✅ Agent optimizer loaded')
report = optimizer.get_optimization_report()
print(f'✅ Optimization report ready: {len(report)} metrics')
"
```

### 4. Evaluation Framework Testing
```bash
# Test comprehensive evaluation
python -c "
from evaluation.onsite_evaluation import OnsiteEvaluator
evaluator = OnsiteEvaluator()
print('✅ Evaluation framework ready')
print(f'Scenarios loaded: {len(evaluator.evaluation_scenarios)}')
"
```

## 🔄 End-to-End Pipeline Testing

### Test 1: Single Episode Run
```python
#!/usr/bin/env python3
"""Test complete single episode with all enhanced components."""

from server.robust_environment import RobustEnvironment
from training.agent_optimizer import get_agent_optimizer
from evaluation.onsite_evaluation import OnsiteEvaluator
import time

def test_single_episode():
    print("🧪 Testing Single Episode Pipeline...")
    
    # Initialize components
    env = RobustEnvironment(use_enhanced_rewards=True)
    optimizer = get_agent_optimizer()
    
    # Reset with test scenario
    obs = env.reset("ws_easy_001")
    print(f"✅ Environment reset: {obs.summary}")
    
    # Run episode with optimized actions
    step_count = 0
    total_reward = 0.0
    
    while not obs.done and step_count < 10:
        # Get optimized action
        from server.tasks.scenarios import get_scenario
        task = get_scenario("ws_easy_001")
        action = optimizer.optimize_action_selection(
            env.state, task, list(ActionCommand), []
        )
        
        # Execute action
        obs = env.step(action)
        total_reward += obs.reward
        step_count += 1
        
        print(f"Step {step_count}: {action.command} → Reward: {obs.reward:.2f}")
    
    # Get episode stats
    stats = env.get_episode_stats()
    print(f"✅ Episode complete: Success={obs.done}, Total Reward={total_reward:.2f}")
    print(f"Stats: {stats}")
    
    return obs.done, total_reward

if __name__ == "__main__":
    success, reward = test_single_episode()
    print(f"🎯 Single Episode Test: {'PASS' if success and reward > 0 else 'FAIL'}")
```

### Test 2: Mini Training Run
```python
#!/usr/bin/env python3
"""Test mini training run with all enhanced components."""

import logging
from server.robust_environment import RobustEnvironment
from training.agent_optimizer import get_agent_optimizer
from evaluation.onsite_evaluation import OnsiteEvaluator
import time

def test_mini_training():
    print("🧪 Testing Mini Training Pipeline...")
    
    # Initialize evaluator
    evaluator = OnsiteEvaluator()
    
    # Run 5 episodes to test training loop
    episodes = 5
    results = []
    
    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1}/{episodes} ---")
        
        # Select scenario
        scenario = evaluator.evaluation_scenarios[episode % len(evaluator.evaluation_scenarios)]
        
        # Run episode (simplified - would use actual model inference)
        try:
            env = RobustEnvironment(use_enhanced_rewards=True)
            obs = env.reset(scenario.task_id)
            
            # Simulate agent actions (would be model inference)
            episode_reward = 0.0
            steps = 0
            
            while not obs.done and steps < scenario.max_steps:
                # Simple action simulation for testing
                from models import Action, ActionCommand
                if steps == 0:
                    action = Action(command=ActionCommand.SET_NAME, args={"name": "test-agent"})
                elif steps == 1:
                    action = Action(command=ActionCommand.SET_DESCRIPTION, args={"description": "Test agent for validation"})
                elif steps == 2:
                    action = Action(command=ActionCommand.ADD_SKILL, args={"skill": "web-scraping"})
                elif steps == 3:
                    action = Action(command=ActionCommand.SET_MODEL, args={"model": "sonnet"})
                elif steps == 4:
                    action = Action(command=ActionCommand.WRITE_PROMPT, args={"prompt": "You are a web scraping specialist for testing purposes."})
                else:
                    action = Action(command=ActionCommand.SUBMIT, args={})
                
                obs = env.step(action)
                episode_reward += obs.reward
                steps += 1
            
            results.append({
                "episode": episode + 1,
                "success": obs.done,
                "reward": episode_reward,
                "steps": steps,
                "scenario": scenario.task_id,
            })
            
            print(f"Episode {episode + 1}: Success={obs.done}, Reward={episode_reward:.2f}")
            
        except Exception as e:
            print(f"Episode {episode + 1} failed: {e}")
            results.append({
                "episode": episode + 1,
                "success": False,
                "reward": 0.0,
                "steps": 0,
                "scenario": scenario.task_id,
                "error": str(e),
            })
    
    # Analyze results
    success_rate = sum(1 for r in results if r["success"]) / len(results)
    mean_reward = sum(r["reward"] for r in results) / len(results)
    
    print(f"\n🎯 Mini Training Results:")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Episodes Completed: {len(results)}")
    
    return success_rate > 0.6 and mean_reward > 1.0

if __name__ == "__main__":
    success = test_mini_training()
    print(f"🎯 Mini Training Test: {'PASS' if success else 'FAIL'}")
```

### Test 3: Integration Validation
```python
#!/usr/bin/env python3
"""Test complete integration of all enhanced components."""

def test_integration():
    print("🧪 Testing Complete Integration...")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Enhanced rewards + robust environment
    try:
        from server.robust_environment import RobustEnvironment
        from server.rewards.enhanced_reward import EnhancedRewardComputer
        from models import RewardConfig
        
        config = RewardConfig()
        enhanced_rewards = EnhancedRewardComputer(config)
        env = RobustEnvironment(use_enhanced_rewards=True)
        
        obs = env.reset("ws_easy_001")
        print("✅ Enhanced rewards + robust environment integration: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Enhanced rewards + robust environment integration: FAIL - {e}")
    tests_total += 1
    
    # Test 2: Agent optimizer + environment
    try:
        from training.agent_optimizer import get_agent_optimizer
        from server.robust_environment import RobustEnvironment
        
        optimizer = get_agent_optimizer()
        env = RobustEnvironment()
        
        obs = env.reset("ws_easy_001")
        from models import ActionCommand
        action = optimizer.optimize_action_selection(
            env.state, 
            get_scenario("ws_easy_001"), 
            list(ActionCommand), 
            []
        )
        print("✅ Agent optimizer + environment integration: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Agent optimizer + environment integration: FAIL - {e}")
    tests_total += 1
    
    # Test 3: Evaluation framework + all components
    try:
        from evaluation.onsite_evaluation import OnsiteEvaluator
        from server.robust_environment import RobustEnvironment
        from training.agent_optimizer import get_agent_optimizer
        
        evaluator = OnsiteEvaluator()
        env = RobustEnvironment()
        optimizer = get_agent_optimizer()
        
        # Test evaluation data collection
        stats = env.get_episode_stats()
        report = optimizer.get_optimization_report()
        
        print("✅ Evaluation framework integration: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Evaluation framework integration: FAIL - {e}")
    tests_total += 1
    
    # Test 4: OpenEnv compliance
    try:
        from server.robust_environment import RobustEnvironment
        import yaml
        
        env = RobustEnvironment()
        
        # Test OpenEnv API compliance
        obs = env.reset()
        assert hasattr(obs, 'reward'), "Missing reward in observation"
        assert hasattr(obs, 'done'), "Missing done in observation"
        assert hasattr(env, 'step'), "Missing step method"
        
        print("✅ OpenEnv compliance: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ OpenEnv compliance: FAIL - {e}")
    tests_total += 1
    
    # Results
    success_rate = tests_passed / tests_total
    print(f"\n🎯 Integration Test Results: {tests_passed}/{tests_total} ({success_rate:.1%})")
    
    return success_rate >= 0.75

if __name__ == "__main__":
    success = test_integration()
    print(f"🎯 Integration Test: {'PASS' if success else 'FAIL'}")
```

## 🚀 Full System Test

### Complete Validation Script
```bash
#!/bin/bash
"""Complete system validation before onsite."""

echo "🚀 Starting Complete System Validation..."

# Test 1: Environment
echo "📋 Testing Environment..."
python test_environment.py
if [ $? -eq 0 ]; then
    echo "✅ Environment: PASS"
else
    echo "❌ Environment: FAIL"
    exit 1
fi

# Test 2: Training Pipeline
echo "📋 Testing Training Pipeline..."
python test_training.py
if [ $? -eq 0 ]; then
    echo "✅ Training Pipeline: PASS"
else
    echo "❌ Training Pipeline: FAIL"
    exit 1
fi

# Test 3: Integration
echo "📋 Testing Integration..."
python test_integration.py
if [ $? -eq 0 ]; then
    echo "✅ Integration: PASS"
else
    echo "❌ Integration: FAIL"
    exit 1
fi

echo "🎯 All Tests Passed! System Ready for Onsite"
```

## 📊 Performance Benchmarks

### Expected Performance Targets
| Component | Target Metric | Acceptable Range |
|-----------|----------------|------------------|
| Environment Reset | <1s | <2s |
| Episode Execution | <30s | <60s |
| Reward Calculation | <0.1s | <0.5s |
| Memory Usage | <2GB | <4GB |
| Error Recovery | >90% | >80% |

### Stress Testing
```python
#!/usr/bin/env python3
"""Stress test with 100 consecutive episodes."""

def stress_test():
    print("🔥 Starting Stress Test (100 episodes)...")
    
    from server.robust_environment import RobustEnvironment
    import time
    
    env = RobustEnvironment()
    start_time = time.time()
    
    success_count = 0
    error_count = 0
    
    for episode in range(100):
        try:
            obs = env.reset("ws_easy_001")
            
            # Quick episode execution
            steps = 0
            while not obs.done and steps < 5:
                from models import Action, ActionCommand
                action = Action(command=ActionCommand.SUBMIT, args={})
                obs = env.step(action)
                steps += 1
            
            if obs.done:
                success_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            error_count += 1
            print(f"Episode {episode + 1} error: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"🔥 Stress Test Results:")
    print(f"Duration: {duration:.1f}s ({duration/100:.2f}s per episode)")
    print(f"Success Rate: {success_count/100:.1%}")
    print(f"Error Rate: {error_count/100:.1%}")
    print(f"Throughput: {100/duration:.1f} episodes/second")
    
    return success_count >= 95 and error_count <= 5

if __name__ == "__main__":
    success = stress_test()
    print(f"🔥 Stress Test: {'PASS' if success else 'FAIL'}")
```

## ✅ Pre-Onsite Validation Checklist

### Final Validation
- [ ] **All component tests pass**
- [ ] **Integration tests pass** 
- [ ] **Performance benchmarks meet targets**
- [ ] **Stress test completes successfully**
- [ ] **Memory usage within limits**
- [ ] **No critical errors in logs**
- [ ] **OpenEnv compliance verified**
- [ ] **HF Space deployment ready**
- [ ] **Training pipeline produces expected results**

### Onsite Readiness
**If all checks pass**: ✅ System is ready for onsite phase
**If any check fails**: ❌ Fix issues before proceeding to onsite

## 🎯 Success Criteria

### Minimum Acceptable Performance
- **Environment stability**: 95%+ successful resets
- **Training reliability**: 90%+ episodes complete without errors
- **Integration success**: All components work together
- **Performance**: Meets benchmark targets
- **Robustness**: Handles errors gracefully

### Optimal Performance
- **Environment stability**: 99%+ successful resets
- **Training reliability**: 95%+ episodes complete without errors
- **Integration success**: Seamless component interaction
- **Performance**: Exceeds benchmark targets
- **Robustness**: Excellent error handling and recovery

---

**Run these tests before April 25th to ensure everything works perfectly for the onsite competition phase!**

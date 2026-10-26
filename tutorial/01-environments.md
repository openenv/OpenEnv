# OpenEnv: Production RL Made Simple

<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg" width="200" alt="PyTorch">

### *From "Hello World" to RL Training in 5 Minutes* ✨

---

**What if RL environments were as easy to use as REST APIs?**

That's OpenEnv. Type-safe. Isolated. Production-ready. 🎯

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-huggingface%2FOpenEnv-blue?logo=github)](https://github.com/huggingface/OpenEnv)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

Author: [Sanyam Bhutani](http://twitter.com/bhutanisanyam1/)

</div>

---

## Why OpenEnv?

Let's take a trip down memory lane:

It's 2016, RL is popular. You read some papers, it looks promising. 

But in real world: Cartpole is the best you can run on a gaming GPU. 

What do you do beyond Cartpole?

Fast-forward to 2025, GRPO is awesome and this time it's not JUST in theory, it works well in practise and is really here!

The problem still remains, how do you take these RL algorithms and take them beyond Cartpole?

A huge part of RL is giving your algorithms environment access to learn. 

We are excited to introduce an Environment Spec for adding Open Environments for RL Training. This will allow you to focus on your experiments and allow everyone to bring their environments.

Focus on experiments, use OpenEnvironments, and build agents that go beyond Cartpole on a single spec.

---

## 📋 What You'll Learn

<table>
<tr>
<td width="50%">

**🎯 Part 1-2: The Fundamentals**

- ⚡ RL in 60 seconds
- 🤔 Why existing solutions fall short
- 💡 The OpenEnv solution

</td>
<td width="50%">

**🏗️ Part 3-5: The Architecture**

- 🔧 How OpenEnv works
- 🔍 Exploring real code
- 🎮 OpenSpiel integration example

</td>
</tr>
<tr>
<td width="50%">

**🎮 Part 6-8: Hands-On Demo**

- 🔌 Use existing OpenSpiel environment
- 🤖 Test 4 different policies
- 👀 Watch learning happen live

</td>
<td width="50%">

**🔧 Part 9-10: Going Further**

- 🎮 Switch to other OpenSpiel games
- ✨ Build your own integration
- 🌐 Deploy to production

</td>
</tr>
</table>

!!! tip "Pro Tip"
    This notebook is designed to run top-to-bottom in Google Colab with zero setup!
    
    ⏱️ **Time**: ~5 minutes | 📊 **Difficulty**: Beginner-friendly | 🎯 **Outcome**: Production-ready RL knowledge

---

## 📑 Table of Contents

### Foundation

- [Part 1: RL in 60 Seconds ⏱️](#part-1-rl-in-60-seconds)
- [Part 2: The Problem with Traditional RL 😤](#part-2-the-problem-with-traditional-rl)
- [Part 3: Setup 🛠️](#part-3-setup)

### Architecture

- [Part 4: The OpenEnv Pattern 🏗️](#part-4-the-openenv-pattern)
- [Part 5: Example Integration - OpenSpiel 🎮](#part-5-example-integration---openspiel)

### Hands-On Demo

- [Part 6: Interactive Demo 🎮](#part-6-using-real-openspiel)
- [Part 7: Four Policies 🤖](#part-7-four-policies)
- [Part 8: Policy Competition! 🏆](#part-8-policy-competition)

### Advanced

- [Part 9: Using Real OpenSpiel 🎮](#part-9-switching-to-other-games)
- [Part 10: Create Your Own Integration 🛠️](#part-10-create-your-own-integration)

### Wrap Up

- [Summary: Your Journey 🎓](#summary-your-journey)
- [Resources 📚](#resources)

---

## Part 1: RL in 60 Seconds ⏱️

**Reinforcement Learning is simpler than you think.**

It's just a loop:

```python
while not done:
    observation = environment.observe()
    action = policy.choose(observation)
    reward = environment.step(action)
    policy.learn(reward)
```

That's it. That's RL.

Let's see it in action:

```python
import random

print("🎲 " + "="*58 + " 🎲")
print("   Number Guessing Game - The Simplest RL Example")
print("🎲 " + "="*58 + " 🎲")

# Environment setup
target = random.randint(1, 10)
guesses_left = 3

print(f"\n🎯 I'm thinking of a number between 1 and 10...")
print(f"💭 You have {guesses_left} guesses. Let's see how random guessing works!\n")

# The RL Loop - Pure random policy (no learning!)
while guesses_left > 0:
    # Policy: Random guessing (no learning yet!)
    guess = random.randint(1, 10)
    guesses_left -= 1
    
    print(f"💭 Guess #{3-guesses_left}: {guess}", end=" → ")
    
    # Reward signal (but we're not using it!)
    if guess == target:
        print("🎉 Correct! +10 points")
        break
    elif abs(guess - target) <= 2:
        print("🔥 Warm! (close)")
    else:
        print("❄️  Cold! (far)")
else:
    print(f"\n💔 Out of guesses. The number was {target}.")

print("\n" + "="*62)
print("💡 This is RL: Observe → Act → Reward → Repeat")
print("   But this policy is terrible! It doesn't learn from rewards.")
print("="*62 + "\n")
```

**Output:**
```
🎲 ========================================================== 🎲
   Number Guessing Game - The Simplest RL Example
🎲 ========================================================== 🎲

🎯 I'm thinking of a number between 1 and 10...
💭 You have 3 guesses. Let's see how random guessing works!

💭 Guess #1: 2 → ❄️  Cold! (far)
💭 Guess #2: 10 → 🎉 Correct! +10 points

==============================================================
💡 This is RL: Observe → Act → Reward → Repeat
   But this policy is terrible! It doesn't learn from rewards.
==============================================================
```

---

## Part 2: The Problem with Traditional RL 😤

### 🤔 Why Can't We Just Use OpenAI Gym?

Good question! Gym is great for research, but production needs more...

| Challenge | Traditional Approach | OpenEnv Solution |
|-----------|---------------------|------------------|
| **Type Safety** | ❌ `obs[0][3]` - what is this? | ✅ `obs.info_state` - IDE knows! |
| **Isolation** | ❌ Same process (can crash your training) | ✅ Docker containers (fully isolated) |
| **Deployment** | ❌ "Works on my machine" 🤷 | ✅ Same container everywhere 🐳 |
| **Scaling** | ❌ Hard to distribute | ✅ Deploy to Kubernetes ☸️ |
| **Language** | ❌ Python only | ✅ Any language (HTTP API) 🌐 |
| **Debugging** | ❌ Cryptic numpy errors | ✅ Clear type errors 🐛 |

### 💡 The OpenEnv Philosophy

**"RL environments should be like microservices"**

Think of it like this: You don't run your database in the same process as your web server, right? Same principle!

- 🔒 **Isolated**: Run in containers (security + stability)
- 🌐 **Standard**: HTTP API, works everywhere
- 📦 **Versioned**: Docker images (reproducibility!)
- 🚀 **Scalable**: Deploy to cloud with one command
- 🛡️ **Type-safe**: Catch bugs before they happen
- 🔄 **Portable**: Works on Mac, Linux, Windows, Cloud

### The Architecture

```
┌────────────────────────────────────────────────────────────┐
│  YOUR TRAINING CODE                                        │
│                                                            │
│  env = OpenSpielEnv(...)        ← Import the client      │
│  result = env.reset()           ← Type-safe!             │
│  result = env.step(action)      ← Type-safe!             │
│                                                            │
└─────────────────┬──────────────────────────────────────────┘
                  │
                  │  OpenEnv protocol over WebSocket
                  │  (persistent session, language-agnostic)
                  │
┌─────────────────▼──────────────────────────────────────────┐
│  DOCKER CONTAINER                                          │
│                                                            │
│  ┌──────────────────────────────────────────────┐         │
│  │  FastAPI Server                              │         │
│  │  └─ Environment (reset, step, state)         │         │
│  │     └─ Your Game/Simulation Logic            │         │
│  └──────────────────────────────────────────────┘         │
│                                                            │
│  Isolated • Reproducible • Secure                          │
└────────────────────────────────────────────────────────────┘
```

!!! info "Key Insight"
    You never see the wire protocol - just clean Python methods!

    ```python
    await env.reset()    # Under the hood: a reset message over WebSocket
    await env.step(...)  # Under the hood: a step message over WebSocket
    await env.state()    # Under the hood: a state message over WebSocket
    ```

    In a notebook you can `await` directly; in a plain script, the same calls
    block automatically. The magic? OpenEnv handles all the plumbing. You focus
    on RL! ✨

---

## Part 3: Setup 🛠️

**Running in Colab?** This cell will clone OpenEnv and install dependencies automatically.

**Running locally?** Make sure you're in the OpenEnv directory.

```python
# Detect environment
try:
    import google.colab
    IN_COLAB = True
    print("🌐 Running in Google Colab - Perfect!")
except ImportError:
    IN_COLAB = False
    print("💻 Running locally - Nice!")

if IN_COLAB:
    print("\n📦 Cloning OpenEnv repository...")
    !git clone https://github.com/huggingface/OpenEnv.git > /dev/null 2>&1
    %cd OpenEnv
    
    print("📚 Installing dependencies (this takes ~10 seconds)...")
    !pip install -q fastapi uvicorn requests
    
    import sys
    sys.path.insert(0, './src')
    print("\n✅ Setup complete! Everything is ready to go! 🎉")
else:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent / 'src'))
    print("✅ Using local OpenEnv installation")

print("\n🚀 Ready to explore OpenEnv and build amazing things!")
print("💡 Tip: Run cells top-to-bottom for the best experience.\n")
```

**Output:**
```
💻 Running locally - Nice!
✅ Using local OpenEnv installation

🚀 Ready to explore OpenEnv and build amazing things!
💡 Tip: Run cells top-to-bottom for the best experience.
```

---

## Part 4: The OpenEnv Pattern 🏗️

### Every OpenEnv Environment Has 3 Components:

```
src/envs/your_env/
├── 📝 models.py          ← Type-safe contracts
│                           (Action, Observation, State)
│
├── 📱 client.py          ← What YOU import
│                           (EnvClient implementation)
│
└── 🖥️  server/
    ├── environment.py    ← Game/simulation logic
    ├── app.py            ← FastAPI server
    └── Dockerfile        ← Container definition
```

Let's explore the actual OpenEnv code to see how this works:

```python
from openenv.core.env_client import EnvClient
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
```

Server side:

```python
class YourEnvironment(Environment[Action, Observation, State]):
    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        ...

    def step(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        ...
```

Client side:

```python
class YourEnv(EnvClient[Action, Observation, State]):
    ...
```

The environment implements the game or simulation logic. The client maintains the session and exposes the same `reset()`, `step()`, and `state()` flow to your training code.

Same interface on both sides, with OpenEnv handling the transport and session management for you.

---

## Part 5: Example Integration - OpenSpiel 🎮

### What is OpenSpiel?

**OpenSpiel** is a library from DeepMind with **70+ game environments** for RL research.

### OpenEnv's Integration

We've wrapped **6 OpenSpiel games** following the OpenEnv pattern:

| **🎯 Single-Player** | **👥 Multi-Player** |
|---------------------|---------------------|
| 1. **Catch** - Catch falling ball | 5. **Tic-Tac-Toe** - Classic 3×3 |
| 2. **Cliff Walking** - Navigate grid | 6. **Kuhn Poker** - Imperfect info poker |
| 3. **2048** - Tile puzzle | |
| 4. **Blackjack** - Card game | |

This shows how OpenEnv can wrap **any** existing RL library!

The client (`envs/openspiel_env/client.py`) inherits from `EnvClient` and declares how actions and observations cross the wire:

```python
class OpenSpielEnv(EnvClient[OpenSpielAction, OpenSpielObservation, OpenSpielState]):
    def _step_payload(self, action: OpenSpielAction) -> dict:
        """Convert typed action to JSON for the step message."""
        return {
            "action_id": action.action_id,
            "game_name": action.game_name,
            "game_params": action.game_params,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse the server response into a typed observation."""
        return StepResult(
            observation=OpenSpielObservation(...),
            reward=payload["reward"],
            done=payload["done"],
        )

    def _parse_state(self, payload: dict) -> OpenSpielState:
        """Parse the server response into a typed state."""
        return OpenSpielState(...)
```

Usage is the same as any OpenEnv environment. In a plain script, these calls
block automatically (in a notebook you can `await` the same client directly
instead):

```python
env = OpenSpielEnv(base_url="http://localhost:8000")

result = env.reset()                                                # StepResult[OpenSpielObservation]
result = env.step(OpenSpielAction(action_id=2, game_name="catch"))
state = env.state()                                                 # OpenSpielState
```

This pattern works for any environment you want to wrap.

### Type-Safe Models

```python
# Import OpenSpiel integration models
from envs.openspiel_env.models import (
    OpenSpielAction,
    OpenSpielObservation,
    OpenSpielState
)

print("="*70)
print("   🎮 OPENSPIEL INTEGRATION - TYPE-SAFE MODELS")
print("="*70)

print("\n📤 OpenSpielAction (what you send):")
print("   " + "─" * 64)
for name, field in OpenSpielAction.model_fields.items():
    print(f"   • {name:20s} : {field.annotation}")

print("\n📥 OpenSpielObservation (what you receive):")
print("   " + "─" * 64)
for name, field in OpenSpielObservation.model_fields.items():
    print(f"   • {name:20s} : {field.annotation}")

print("\n📊 OpenSpielState (episode metadata):")
print("   " + "─" * 64)
for name, field in OpenSpielState.model_fields.items():
    print(f"   • {name:20s} : {field.annotation}")

print("\n" + "="*70)
print("\n💡 Type safety means:")
print("   ✅ Your IDE autocompletes these fields")
print("   ✅ Typos are caught before running")
print("   ✅ Refactoring is safe")
print("   ✅ Self-documenting code\n")
```

**Output:**
```
======================================================================
   🎮 OPENSPIEL INTEGRATION - TYPE-SAFE MODELS
======================================================================

📤 OpenSpielAction (what you send):
   ────────────────────────────────────────────────────────────────
   • metadata             : typing.Dict[str, typing.Any]
   • action_id            : int
   • game_name            : str
   • game_params          : Dict[str, Any]

📥 OpenSpielObservation (what you receive):
   ────────────────────────────────────────────────────────────────
   • done                 : <class 'bool'>
   • reward               : typing.Union[bool, int, float, NoneType]
   • metadata             : typing.Dict[str, typing.Any]
   • info_state           : List[float]
   • legal_actions        : List[int]
   • game_phase           : str
   • current_player_id    : int
   • opponent_last_action : Optional[int]

📊 OpenSpielState (episode metadata):
   ────────────────────────────────────────────────────────────────
   • episode_id           : typing.Optional[str]
   • step_count           : <class 'int'>
   • game_name            : str
   • agent_player         : int
   • opponent_policy      : str
   • game_params          : Dict[str, Any]
   • num_players          : int

======================================================================

💡 Type safety means:
   ✅ Your IDE autocompletes these fields
   ✅ Typos are caught before running
   ✅ Refactoring is safe
   ✅ Self-documenting code
```

### How the Client Works

The client **inherits from `EnvClient`** and implements 3 methods:

1. `_step_payload()` - Convert action → JSON
2. `_parse_result()` - Parse JSON → typed observation  
3. `_parse_state()` - Parse JSON → state

That's it! The base class handles all the async WebSocket communication.

---

## Part 6: Using Real OpenSpiel 🎮

<div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin: 30px 0;">

### Now let's USE a production environment!

We'll play **Catch** using OpenEnv's **OpenSpiel integration** 🎯

This is a REAL environment running in production at companies!

**Get ready for:**

- 🔌 Using existing environments (not building)
- 🤖 Testing policies against real games
- 📊 Live gameplay visualization
- 🎯 Production-ready patterns

</div>

### The Game: Catch 🔴🏓

```
⬜ ⬜ 🔴 ⬜ ⬜
⬜ ⬜ ⬜ ⬜ ⬜
⬜ ⬜ ⬜ ⬜ ⬜   Ball
⬜ ⬜ ⬜ ⬜ ⬜
⬜ ⬜ ⬜ ⬜ ⬜   falls
⬜ ⬜ ⬜ ⬜ ⬜
⬜ ⬜ ⬜ ⬜ ⬜   down
⬜ ⬜ ⬜ ⬜ ⬜
⬜ ⬜ ⬜ ⬜ ⬜
⬜ ⬜ 🏓 ⬜ ⬜
     Paddle
```

**Rules:**

- 10×5 grid
- Ball falls from random column
- Move paddle left/right to catch it

**Actions:**

- `0` = Move LEFT ⬅️
- `1` = STAY 🛑
- `2` = Move RIGHT ➡️

**Reward:**

- `+1` if caught 🎉
- `0` if missed 😢

!!! note "Why Catch?"
    - Simple rules (easy to understand)
    - Fast episodes (~5 steps)
    - Clear success/failure
    - Part of OpenSpiel's 70+ games!

    **💡 The Big Idea:**
    Instead of building this from scratch, we'll USE OpenEnv's existing OpenSpiel integration. Same interface, but production-ready!

```python
from envs.openspiel_env import OpenSpielEnv
from envs.openspiel_env.models import (
    OpenSpielAction,
    OpenSpielObservation,
    OpenSpielState
)

print("🎮 " + "="*64 + " 🎮")
print("   ✅ Importing Real OpenSpiel Environment!")
print("🎮 " + "="*64 + " 🎮\n")

print("📦 What we just imported:")
print("   • OpenSpielEnv - EnvClient for OpenSpiel games")
print("   • OpenSpielAction - Type-safe actions")
print("   • OpenSpielObservation - Type-safe observations")
print("   • OpenSpielState - Episode metadata\n")

print("📋 OpenSpielObservation fields:")
print("   " + "─" * 60)
for name, field in OpenSpielObservation.model_fields.items():
    print(f"   • {name:25s} : {field.annotation}")

print("\n" + "="*70)
print("\n💡 This is REAL OpenEnv code - used in production!")
print("   • Wraps 6 OpenSpiel games (Catch, Tic-Tac-Toe, Poker, etc.)")
print("   • Type-safe actions and observations")
print("   • Talks to the server over WebSocket (we'll see that next!)\n")
```

**Output:**
```
🎮 ================================================================ 🎮
   ✅ Importing Real OpenSpiel Environment!
🎮 ================================================================ 🎮

📦 What we just imported:
   • OpenSpielEnv - EnvClient for OpenSpiel games
   • OpenSpielAction - Type-safe actions
   • OpenSpielObservation - Type-safe observations
   • OpenSpielState - Episode metadata

📋 OpenSpielObservation fields:
   ────────────────────────────────────────────────────────────
   • done                      : <class 'bool'>
   • reward                    : typing.Union[bool, int, float, NoneType]
   • metadata                  : typing.Dict[str, typing.Any]
   • info_state                : List[float]
   • legal_actions             : List[int]
   • game_phase                : str
   • current_player_id         : int
   • opponent_last_action      : Optional[int]

======================================================================

💡 This is REAL OpenEnv code - used in production!
   • Wraps 6 OpenSpiel games (Catch, Tic-Tac-Toe, Poker, etc.)
   • Type-safe actions and observations
   • Talks to the server over WebSocket (we'll see that next!)
```

---

## Part 7: Four Policies 🤖

Let's test 4 different AI strategies:

| Policy | Strategy | Expected Performance |
|--------|----------|----------------------|
| **🎲 Random** | Pick random action every step | ~20% (pure luck) |
| **🛑 Always Stay** | Never move, hope ball lands in center | ~20% (terrible!) |
| **🧠 Smart** | Move paddle toward ball | 100% (optimal!) |
| **📈 Learning** | Start random, learn smart strategy | ~85% (improves over time) |

**💡 These policies work with ANY OpenSpiel game!**

```python
import random

# ============================================================================
# POLICIES - Different AI strategies (adapted for OpenSpiel)
# ============================================================================

class RandomPolicy:
    """Baseline: Pure random guessing."""
    name = "🎲 Random Guesser"

    def select_action(self, obs: OpenSpielObservation) -> int:
        return random.choice(obs.legal_actions)


class AlwaysStayPolicy:
    """Bad strategy: Never moves."""
    name = "🛑 Always Stay"

    def select_action(self, obs: OpenSpielObservation) -> int:
        return 1  # STAY


class SmartPolicy:
    """Optimal: Move paddle toward ball."""
    name = "🧠 Smart Heuristic"

    def select_action(self, obs: OpenSpielObservation) -> int:
        # Parse OpenSpiel observation
        # For Catch: info_state is a flattened 10x5 grid
        # Ball position and paddle position encoded in the vector
        info_state = obs.info_state

        # Find ball and paddle positions from info_state
        # Catch uses a 10x5 grid, so 50 values
        grid_size = 5

        # Find positions (ball = 1.0 in the flattened grid, paddle = 1.0 in the last row of the flattened grid)
        ball_col = None
        paddle_col = None

        for idx, val in enumerate(info_state):
            if abs(val - 1.0) < 0.01:  # Ball
                ball_col = idx % grid_size
                break

        last_row = info_state[-grid_size:]
        paddle_col = last_row.index(1.0) # Paddle

        if ball_col is not None and paddle_col is not None:
            if paddle_col < ball_col:
                return 2  # Move RIGHT
            elif paddle_col > ball_col:
                return 0  # Move LEFT

        return 1  # STAY (fallback)


class LearningPolicy:
    """Simulated RL: Epsilon-greedy exploration."""
    name = "📈 Learning Agent"

    def __init__(self):
        self.steps = 0
        self.smart_policy = SmartPolicy()

    def select_action(self, obs: OpenSpielObservation) -> int:
        self.steps += 1

        # Decay exploration rate over time
        epsilon = max(0.1, 1.0 - (self.steps / 100))

        if random.random() < epsilon:
            # Explore: random action
            return random.choice(obs.legal_actions)
        else:
            # Exploit: use smart strategy
            return self.smart_policy.select_action(obs)


print("🤖 " + "="*64 + " 🤖")
print("   ✅ 4 Policies Created (Adapted for OpenSpiel)!")
print("🤖 " + "="*64 + " 🤖\n")

policies = [RandomPolicy(), AlwaysStayPolicy(), SmartPolicy(), LearningPolicy()]
for i, policy in enumerate(policies, 1):
    print(f"   {i}. {policy.name}")

print("\n💡 These policies work with OpenSpielObservation!")
print("   • Read info_state (flattened grid)")
print("   • Use legal_actions")
print("   • Work with ANY OpenSpiel game that exposes these!\n")
```

**Output:**
```
🤖 ================================================================ 🤖
   ✅ 4 Policies Created (Adapted for OpenSpiel)!
🤖 ================================================================ 🤖

   1. 🎲 Random Guesser
   2. 🛑 Always Stay
   3. 🧠 Smart Heuristic
   4. 📈 Learning Agent

💡 These policies work with OpenSpielObservation!
   • Read info_state (flattened grid)
   • Use legal_actions
   • Work with ANY OpenSpiel game that exposes these!
```

---

## Part 8: Policy Competition! 🏆

Let's run **50 episodes** for each policy against **REAL OpenSpiel** and see who wins!

This is production code - every action is a WebSocket message to the OpenSpiel server!

```python
def evaluate_policies(env, num_episodes=50):
    """Compare all policies over many episodes using real OpenSpiel."""
    policies = [
        RandomPolicy(),
        AlwaysStayPolicy(),
        SmartPolicy(),
        LearningPolicy(),
    ]

    print("\n🏆 " + "="*66 + " 🏆")
    print(f"   POLICY SHOWDOWN - {num_episodes} Episodes Each")
    print(f"   Playing against REAL OpenSpiel Catch!")
    print("🏆 " + "="*66 + " 🏆\n")

    results = []
    for policy in policies:
        print(f"⚡ Testing {policy.name}...", end=" ")
        successes = sum(run_episode(env, policy, visualize=False)
                       for _ in range(num_episodes))
        success_rate = (successes / num_episodes) * 100
        results.append((policy.name, success_rate, successes))
        print(f"✓ Done!")

    print("\n" + "="*70)
    print("   📊 FINAL RESULTS")
    print("="*70 + "\n")

    # Sort by success rate (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    # Award medals to top 3
    medals = ["🥇", "🥈", "🥉", "  "]

    for i, (name, rate, successes) in enumerate(results):
        medal = medals[i]
        bar = "█" * int(rate / 2)
        print(f"{medal} {name:25s} [{bar:<50}] {rate:5.1f}% ({successes}/{num_episodes})")

    print("\n" + "="*70)
    print("\n✨ Key Insights:")
    print("   • Random (~20%):      Baseline - pure luck 🎲")
    print("   • Always Stay (~20%): Bad strategy - stays center 🛑")
    print("   • Smart (100%):       Optimal - perfect play! 🧠")
    print("   • Learning (~85%):    Improves over time 📈")
    print("\n🎓 This is Reinforcement Learning + OpenEnv in action:")
    print("   1. We USED existing OpenSpiel environment (didn't build it)")
    print("   2. Type-safe communication over WebSocket")
    print("   3. Same code works for ANY OpenSpiel game")
    print("   4. Production-ready architecture\n")

# Run the epic competition!
print("🎮 Starting the showdown against REAL OpenSpiel...\n")
evaluate_policies(client, num_episodes=50)
```

---

## Part 9: Switching to Other Games 🎮

### What We Just Used: Real OpenSpiel! 🎉

In Parts 6-8, we **USED** the existing OpenSpiel Catch environment:

| What We Did | How It Works |
|-------------|--------------|
| **Imported** | OpenSpielEnv client (pre-built) |
| **Started** | OpenSpiel server via uvicorn |
| **Connected** | Async client over WebSocket |
| **Played** | Real OpenSpiel Catch game |

**🎯 This is production code!** Every action was a WebSocket message to a real OpenSpiel environment.

### 🎮 6 Games Available - Same Interface!

The beauty of OpenEnv? **Same code, different games!**

```python
# We just used Catch
env = OpenSpielEnv(base_url="http://localhost:8000")
# game_name="catch" was set via environment variable

# Want Tic-Tac-Toe instead? Just change the game!
# Start server with: OPENSPIEL_GAME=tic_tac_toe uvicorn ...
# Same client code works!
```

**🎮 All 6 Games:**

1. ✅ **`catch`** - What we just used!
2. **`tic_tac_toe`** - Classic 3×3
3. **`kuhn_poker`** - Imperfect information poker
4. **`cliff_walking`** - Grid navigation
5. **`2048`** - Tile puzzle
6. **`blackjack`** - Card game

**All use the exact same OpenSpielEnv client!**

### Try Another Game (Optional):

```python
# Stop the current server (kill the server_process)
# Then start a new game:

server_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn",
     "envs.openspiel_env.server.app:app",
     "--host", "0.0.0.0",
     "--port", "8000"],
    env={**os.environ,
         "PYTHONPATH": f"{work_dir}/src",
         "OPENSPIEL_GAME": "tic_tac_toe",  # Changed!
         "OPENSPIEL_AGENT_PLAYER": "0",
         "OPENSPIEL_OPPONENT_POLICY": "random"},
    # ... rest of config
)

# Same client works!
client = OpenSpielEnv(base_url="http://localhost:8000")
result = client.reset()  # Now playing Tic-Tac-Toe!
```

**💡 Key Insight**: You don't rebuild anything - you just USE different games with the same client!

---

## Part 10: Create Your Own Integration 🛠️

### The 5-Step Pattern

Want to wrap your own environment in OpenEnv? Here's how:

### Step 1: Define Types (`models.py`)

Actions, observations, and state are **Pydantic models**. The base classes already
provide the common fields — `Observation` has `done`, `reward`, `metadata`, and
`State` has `episode_id`, `step_count` — so you only add what's specific to your env.

```python
from typing import List
from openenv.core.env_server import Action, Observation, State
from pydantic import Field

class YourAction(Action):
    action_value: int  # add your action fields

class YourObservation(Observation):
    # `done`, `reward`, and `metadata` are inherited from Observation
    state_data: List[float]  # add your observation fields

class YourState(State):
    # `episode_id` and `step_count` are inherited from State
    score: int = 0  # add your state fields
```

### Step 2: Implement Environment (`server/environment.py`)

```python
from openenv.core.env_server import Environment

class YourEnvironment(Environment):
    def reset(self, seed=None, episode_id=None, **kwargs) -> YourObservation:
        # Initialize your game/simulation
        return YourObservation(...)

    def step(self, action, **kwargs) -> YourObservation:
        # Execute action, update state
        return YourObservation(...)

    @property
    def state(self) -> YourState:
        return self._state
```

### Step 3: Create Client (`client.py`)

The client subclasses `EnvClient` and implements 3 hooks. Callers can `await`
`reset()`/`step()`/`state()` in async contexts, or call them directly from
synchronous scripts.

```python
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

class YourEnv(EnvClient[YourAction, YourObservation, YourState]):
    def _step_payload(self, action: YourAction) -> dict:
        """Convert action to the JSON step message."""
        return {"action_value": action.action_value}

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse the server response into a typed observation."""
        return StepResult(
            observation=YourObservation(...),
            reward=payload['reward'],
            done=payload['done']
        )

    def _parse_state(self, payload: dict) -> YourState:
        return YourState(...)
```

### Step 4: Create Server (`server/app.py`)

Pass the environment **class** (a factory) plus the action and observation types.
OpenEnv builds the WebSocket + HTTP endpoints for you.

```python
from openenv.core.env_server import create_app
from .your_environment import YourEnvironment

app = create_app(
    YourEnvironment,
    YourAction,
    YourObservation,
    env_name="your_env",
)

# That's it! OpenEnv creates all endpoints for you.
```

### Step 5: Dockerize (`server/Dockerfile`)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 🎓 Examples to Study

OpenEnv includes 3 complete examples:

1. **`src/envs/echo_env/`**
   - Simplest possible environment
   - Great for testing and learning

2. **`src/envs/openspiel_env/`**
   - Wraps external library (OpenSpiel)
   - Shows integration pattern
   - 6 games in one integration

3. **`src/envs/coding_env/`**
   - Python code execution environment
   - Shows complex use case
   - Security considerations

**💡 Study these to understand the patterns!**

---

## 🎓 Summary: Your Journey

### What You Learned

<table>
<tr>
<td width="50%" style="vertical-align: top;">

### 📚 Concepts

✅ **RL Fundamentals**

- The observe-act-reward loop
- What makes good policies
- Exploration vs exploitation

✅ **OpenEnv Architecture**

- Client-server separation
- Type-safe contracts
- WebSocket communication layer

✅ **Production Patterns**

- Docker isolation
- API design
- Reproducible deployments

</td>
<td width="50%" style="vertical-align: top;">

### 🛠️ Skills

✅ **Using Environments**

- Import OpenEnv clients
- Call reset/step/state
- Work with typed observations

✅ **Building Environments**

- Define type-safe models
- Implement Environment class
- Subclass EnvClient

✅ **Testing & Debugging**

- Compare policies
- Visualize episodes
- Measure performance

</td>
</tr>
</table>

### OpenEnv vs Traditional RL

| Feature | Traditional (Gym) | OpenEnv | Winner |
|---------|------------------|---------|--------|
| **Type Safety** | ❌ Arrays, dicts | ✅ Dataclasses | 🏆 OpenEnv |
| **Isolation** | ❌ Same process | ✅ Docker | 🏆 OpenEnv |
| **Deployment** | ❌ Manual setup | ✅ K8s-ready | 🏆 OpenEnv |
| **Language** | ❌ Python only | ✅ Any (HTTP) | 🏆 OpenEnv |
| **Reproducibility** | ❌ "Works on my machine" | ✅ Same everywhere | 🏆 OpenEnv |
| **Community** | ✅ Large ecosystem | 🟡 Growing | 🤝 Both! |

!!! success "The Bottom Line"
    OpenEnv brings **production engineering** to RL:
    
    - Same environments work locally and in production
    - Type safety catches bugs early
    - Docker isolation prevents conflicts
    - HTTP API works with any language

    **It's RL for 2024 and beyond.**

---

## 📚 Resources

### 🔗 Essential Links

- **🏠 OpenEnv GitHub**: https://github.com/huggingface/OpenEnv
- **🎮 OpenSpiel**: https://github.com/google-deepmind/open_spiel
- **⚡ FastAPI Docs**: https://fastapi.tiangolo.com/
- **🐳 Docker Guide**: https://docs.docker.com/get-started/
- **🔥 PyTorch**: https://pytorch.org/

### 📖 Documentation Deep Dives

- **Environment Creation Guide**: `src/envs/README.md`
- **OpenSpiel Integration**: `src/envs/openspiel_env/README.md`
- **Example Scripts**: `examples/`
- **RFC 001**: [Baseline API Specs](https://github.com/huggingface/OpenEnv/pull/26)

### 🎓 Community & Support

**Openly governed by a technical committee including:**

- 🔥 Meta PyTorch
- 🌟 Reflection
- ⚡ Unsloth
- ☁️ Modal
- 🧠 Prime Intellect
- 🟢 Nvidia
- 💼 Mercor
- 🚀 Fleet AI
- 🪟 Microsoft
- 🤗 Hugging Face

**Supported by amazing organizations and contributors.**

- 🚀 And many more!

Technical direction, RFCs, and release planning are coordinated in public through the OpenEnv repository.

**License**: BSD 3-Clause License

**Contributions**: Always welcome! Check out the issues tab.

---

### 🌈 What's Next?

1. ⭐ **Star the repo** to show support and stay updated
2. 🔄 **Try modifying** the Catch game (make it harder? bigger grid?)
3. 🎮 **Explore** other OpenSpiel games
4. 🛠️ **Build** your own environment integration
5. 💬 **Share** what you build with the community!

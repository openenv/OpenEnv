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
                  │  WebSocket/JSON (Persistent Session)
                  │  WS /ws  (reset, step, state messages)
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
    You never see WebSocket details - just clean Python methods!

    ```python
    env.reset()    # Under the hood: WebSocket message via /ws
    env.step(...)  # Under the hood: WebSocket message via /ws
    env.state()    # Under the hood: WebSocket message via /ws
    ```

    The magic? OpenEnv handles all the plumbing. You focus on RL! ✨

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
envs/your_env/
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
# Import OpenEnv's core abstractions
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.env_client import EnvClient

print("="*70)
print("   🧩 OPENENV CORE ABSTRACTIONS")
print("="*70)

print("""
🖥️  SERVER SIDE (runs in Docker):

    class Environment(ABC):
        '''Base class for all environment implementations'''

        @abstractmethod
        def reset(self) -> Observation:
            '''Start new episode'''

        @abstractmethod
        def step(self, action: Action) -> Observation:
            '''Execute action, return observation'''

        @property
        def state(self) -> State:
            '''Get episode metadata'''

📱 CLIENT SIDE (your training code):

    class EnvClient(ABC):
        '''Base class for WebSocket clients'''

        def reset(self) -> StepResult:
            # WebSocket message to server

        def step(self, action) -> StepResult:
            # WebSocket message to server

        def state(self) -> State:
            # WebSocket message to server
""")

print("="*70)
print("\n✨ Same interface on both sides - communication via WebSocket!")
print("🎯 You focus on RL, OpenEnv handles the infrastructure.\n")
```

**Output:**
```
======================================================================
   🧩 OPENENV CORE ABSTRACTIONS
======================================================================

🖥️  SERVER SIDE (runs in Docker):

    class Environment(ABC):
        '''Base class for all environment implementations'''
        
        @abstractmethod
        def reset(self) -> Observation:
            '''Start new episode'''
        
        @abstractmethod
        def step(self, action: Action) -> Observation:
            '''Execute action, return observation'''
        
        @property
        def state(self) -> State:
            '''Get episode metadata'''

📱 CLIENT SIDE (your training code):

    class EnvClient(ABC):
        '''Base class for WebSocket clients'''

        def reset(self) -> StepResult:
            # WebSocket message to server

        def step(self, action) -> StepResult:
            # WebSocket message to server

        def state(self) -> State:
            # WebSocket message to server

======================================================================

✨ Same interface on both sides - communication via WebSocket!
🎯 You focus on RL, OpenEnv handles the infrastructure.
```

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

```python
from envs.openspiel_env.client import OpenSpielEnv

print("="*70)
print("   🔌 HOW OPENENV WRAPS OPENSPIEL")
print("="*70)

print("""
class OpenSpielEnv(EnvClient[OpenSpielAction, OpenSpielObservation, OpenSpielState]):

    def _step_payload(self, action: OpenSpielAction) -> dict:
        '''Convert typed action to JSON for WebSocket message'''
        return {
            "action_id": action.action_id,
            "game_name": action.game_name,
            "game_params": action.game_params,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        '''Parse JSON response into typed observation'''
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=OpenSpielObservation(...),
            reward=payload['reward'],
            done=payload['done']
        )

""")

print("─" * 70)
print("\n✨ Usage (works for ALL OpenEnv environments):")
print("""
  env = OpenSpielEnv(base_url="http://localhost:8000")
  
  result = env.reset()
  # Returns StepResult[OpenSpielObservation] - Type safe!
  
  result = env.step(OpenSpielAction(action_id=2, game_name="catch"))
  # Type checker knows this is valid!
  
  state = env.state()
  # Returns OpenSpielState
""")

print("─" * 70)
print("\n🎯 This pattern works for ANY environment you want to wrap!\n")
```

**Output:**
```
======================================================================
   🔌 HOW OPENENV WRAPS OPENSPIEL
======================================================================

class OpenSpielEnv(EnvClient[OpenSpielAction, OpenSpielObservation, OpenSpielState]):

    def _step_payload(self, action: OpenSpielAction) -> dict:
        '''Convert typed action to JSON for WebSocket message'''
        return {
            "action_id": action.action_id,
            "game_name": action.game_name,
            "game_params": action.game_params,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        '''Parse JSON response into typed observation'''
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=OpenSpielObservation(...),
            reward=payload['reward'],
            done=payload['done']
        )


──────────────────────────────────────────────────────────────────────

✨ Usage (works for ALL OpenEnv environments):

  env = OpenSpielEnv(base_url="http://localhost:8000")
  
  result = env.reset()
  # Returns StepResult[OpenSpielObservation] - Type safe!
  
  result = env.step(OpenSpielAction(action_id=2, game_name="catch"))
  # Type checker knows this is valid!
  
  state = env.state()
  # Returns OpenSpielState

──────────────────────────────────────────────────────────────────────

🎯 This pattern works for ANY environment you want to wrap!
```

### Type-Safe Models

```python
# Import OpenSpiel integration models
from envs.openspiel_env.models import (
    OpenSpielAction,
    OpenSpielObservation,
    OpenSpielState
)
from dataclasses import fields

print("="*70)
print("   🎮 OPENSPIEL INTEGRATION - TYPE-SAFE MODELS")
print("="*70)

print("\n📤 OpenSpielAction (what you send):")
print("   " + "─" * 64)
for field in fields(OpenSpielAction):
    print(f"   • {field.name:20s} : {field.type}")

print("\n📥 OpenSpielObservation (what you receive):")
print("   " + "─" * 64)
for field in fields(OpenSpielObservation):
    print(f"   • {field.name:20s} : {field.type}")

print("\n📊 OpenSpielState (episode metadata):")
print("   " + "─" * 64)
for field in fields(OpenSpielState):
    print(f"   • {field.name:20s} : {field.type}")

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

The client **inherits from EnvClient** and implements 3 methods:

1. `_step_payload()` - Convert action → JSON
2. `_parse_result()` - Parse JSON → typed observation  
3. `_parse_state()` - Parse JSON → state

That's it! The base class handles all WebSocket communication.

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
from dataclasses import fields

print("🎮 " + "="*64 + " 🎮")
print("   ✅ Importing Real OpenSpiel Environment!")
print("🎮 " + "="*64 + " 🎮\n")

print("📦 What we just imported:")
print("   • OpenSpielEnv - WebSocket client for OpenSpiel games")
print("   • OpenSpielAction - Type-safe actions")
print("   • OpenSpielObservation - Type-safe observations")
print("   • OpenSpielState - Episode metadata\n")

print("📋 OpenSpielObservation fields:")
print("   " + "─" * 60)
for field in fields(OpenSpielObservation):
    print(f"   • {field.name:25s} : {field.type}")

print("\n" + "="*70)
print("\n💡 This is REAL OpenEnv code - used in production!")
print("   • Wraps 6 OpenSpiel games (Catch, Tic-Tac-Toe, Poker, etc.)")
print("   • Type-safe actions and observations")
print("   • Works via HTTP (we'll see that next!)\n")
```

**Output:**
```
🎮 ================================================================ 🎮
   ✅ Importing Real OpenSpiel Environment!
🎮 ================================================================ 🎮

📦 What we just imported:
   • OpenSpielEnv - WebSocket client for OpenSpiel games
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
   • Works via HTTP (we'll see that next!)
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

This is production code - every action is an HTTP call to the OpenSpiel server!

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
    print("   2. Type-safe communication over HTTP")
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
| **Connected** | HTTP client to server |
| **Played** | Real OpenSpiel Catch game |

**🎯 This is production code!** Every action was an HTTP call to a real OpenSpiel environment.

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

```python
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

class YourAction(Action):
    action_value: int = Field(..., description="The action to take")
    # Add your action fields

class YourObservation(Observation):
    state_data: list[float] = Field(default_factory=list, description="State tensor")
    # done, reward, metadata inherited from Observation

class YourState(State):
    # episode_id, step_count inherited from State
    custom_field: str = Field(default="", description="Your custom state field")
```

### Step 2: Implement Environment (`server/environment.py`)

```python
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import YourAction, YourObservation

class YourEnvironment(Environment):
    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> YourObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        return YourObservation(state_data=[], done=False, reward=0.0)

    def step(self, action: YourAction) -> YourObservation:
        self._state.step_count += 1
        # Execute action, update state
        return YourObservation(state_data=[1.0], done=False, reward=1.0)

    @property
    def state(self) -> State:
        return self._state
```

### Step 3: Create Client (`client.py`)

```python
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import YourAction, YourObservation

class YourEnv(EnvClient[YourAction, YourObservation, State]):
    def _step_payload(self, action: YourAction) -> dict:
        """Convert action to JSON for WebSocket message"""
        return {"action_value": action.action_value}

    def _parse_result(self, payload: dict) -> StepResult[YourObservation]:
        """Parse JSON response into typed observation"""
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=YourObservation(
                state_data=obs_data.get("state_data", []),
                done=payload.get("done", False),
                reward=payload.get("reward"),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
```

### Step 4: Create Server (`server/app.py`)

```python
from openenv.core.env_server.http_server import create_app
from models import YourAction, YourObservation
from .your_environment import YourEnvironment

# Pass the class (not an instance) - each WebSocket session gets its own instance
app = create_app(YourEnvironment, YourAction, YourObservation, env_name="your_env")
```

### Step 5: Dockerize (`server/Dockerfile`)

Use the `openenv-base` image and `uv` for dependency management:

```dockerfile
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE} AS builder
WORKDIR /app
COPY . /app/env
WORKDIR /app/env
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-install-project --no-editable
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-editable

FROM ${BASE_IMAGE}
WORKDIR /app
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
```

### 🎓 Examples to Study

OpenEnv includes 3 complete examples:

1. **`envs/echo_env/`**
   - Simplest possible environment (MCP tool-based)
   - Great for testing and learning

2. **`envs/openspiel_env/`**
   - Wraps external library (OpenSpiel)
   - Shows typed EnvClient integration pattern
   - 6 games in one integration

3. **`envs/coding_env/`**
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
- Create EnvClient

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
| **Language** | ❌ Python only | ✅ Any (WebSocket/HTTP) | 🏆 OpenEnv |
| **Reproducibility** | ❌ "Works on my machine" | ✅ Same everywhere | 🏆 OpenEnv |
| **Community** | ✅ Large ecosystem | 🟡 Growing | 🤝 Both! |

!!! success "The Bottom Line"
    OpenEnv brings **production engineering** to RL:
    
    - Same environments work locally and in production
    - Type safety catches bugs early
    - Docker isolation prevents conflicts
    - WebSocket API works with any language

    **It's RL for production.**

---

## 📚 Resources

### 🔗 Essential Links

- **🏠 OpenEnv GitHub**: https://github.com/huggingface/OpenEnv
- **🎮 OpenSpiel**: https://github.com/google-deepmind/open_spiel
- **⚡ FastAPI Docs**: https://fastapi.tiangolo.com/
- **🐳 Docker Guide**: https://docs.docker.com/get-started/
- **🔥 PyTorch**: https://pytorch.org/

### 📖 Documentation Deep Dives

- **Environment Creation Guide**: `envs/README.md`
- **OpenSpiel Integration**: `envs/openspiel_env/README.md`
- **Example Scripts**: `examples/`
- **RFC 001**: [Baseline API Specs](https://github.com/huggingface/OpenEnv/pull/26)

### 🎓 Community & Support

**Supported by amazing organizations:**

- 🔥 Meta PyTorch
- 🤗 Hugging Face
- ⚡ Unsloth AI
- 🌟 Reflection AI
- 🚀 And many more!

**License**: BSD 3-Clause (very permissive!)

**Contributions**: Always welcome! Check out the issues tab.

---

### 🌈 What's Next?

1. ⭐ **Star the repo** to show support and stay updated
2. 🔄 **Try modifying** the Catch game (make it harder? bigger grid?)
3. 🎮 **Explore** other OpenSpiel games
4. 🛠️ **Build** your own environment integration
5. 💬 **Share** what you build with the community!


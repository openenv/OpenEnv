# Training LLMs to Play BlackJack with GRPO + OpenEnv

This example demonstrates how to train language models to play BlackJack using **GRPO (Group Relative Policy Optimization)** and **OpenEnv**.

## 🎯 What This Example Shows

- **OpenEnv**: Universal RL environment interface for 70+ environments
- **GRPO**: Efficient RL algorithm (used by DeepSeek R1) that only needs 2 models instead of 3
- **Forge**: PyTorch-native agentic RL library for production training
- **End-to-End Training**: From random policy (~35% win rate) to trained agent

## 📁 Files

- `grpo_blackjack_tutorial.ipynb` - Interactive tutorial notebook (recommended starting point)
- `grpo_utils.py` - Production GRPO utilities and helper functions
- `blackjack.yaml` - Training configuration file
- `README.md` - This file

## 🚀 Quick Start

### Prerequisites

1. **Install OpenEnv**:
   ```bash
   # Clone OpenEnv repo
   git clone https://github.com/huggingface/OpenEnv.git
   cd OpenEnv
   pip install -e .
   ```

2. **Install Forge** (PyTorch's agentic RL library):
   ```bash
   git clone https://github.com/meta-pytorch/torchforge.git
   cd torchforge
   pip install -e .
   ```

3. **Start OpenEnv BlackJack Server**:
   ```bash
   # In a separate terminal
   export OPENENV_PATH="/path/to/OpenEnv/src"
   export PYTHONPATH="${OPENENV_PATH}:${PYTHONPATH}"

   OPENSPIEL_GAME=blackjack python -m envs.openspiel_env.server.app --port 8004
   ```

### Run the Tutorial

Open the Jupyter notebook:
```bash
jupyter notebook grpo_blackjack_tutorial.ipynb
```

Follow the cells to:
1. **Explore OpenEnv** - Connect to BlackJack environment
2. **Benchmark baseline** - Test random policy performance
3. **Learn about GRPO** - Understand the training algorithm
4. **Train with Forge** - Run production GRPO training
5. **Switch environments** - See how to train on other games

## 📚 What You'll Learn

### OpenEnv: Universal RL Environment Spec

OpenEnv is **not a game engine** - it's a **specification** that wraps ANY RL environment:

```python
# Same interface works for 70+ environments
result = env.reset()              # Start episode
result = env.step(action)         # Take action
state = env.state()               # Get state
env.close()                       # Cleanup
```

Change one environment variable → train on different games!

### Forge: PyTorch-Native Agentic RL

Forge handles all distributed systems complexity:
- **Generator (vLLM)**: Fast LLM inference
- **RLTrainer**: Distributed training with FSDP
- **ReplayBuffer**: Off-policy learning
- **ReferenceModel**: KL penalty computation
- **Torchstore**: Distributed weight management

You just write:
```python
trainer = await setup_forge_training("blackjack.yaml")
await trainer.run(steps=100)
```

Everything else is automated!

## 🎓 Educational Resources

This tutorial is inspired by the excellent [Unsloth RL Guide](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide). We highly recommend reading it for deeper insights!

### Further Reading

- **OpenEnv**: [GitHub](https://github.com/huggingface/OpenEnv)
- **GRPO Paper**: [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- **Forge**: [GitHub](https://github.com/meta-pytorch/torchforge) | [Docs](https://meta-pytorch.org/torchforge/)
- **Unsloth RL Guide**: [docs.unsloth.ai](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide)

## 💡 Key Concepts

### "Patience Is All You Need" for RL

RL works by patience: if the correct answer has *any* non-zero probability, we'll eventually find it through sampling. While waiting:
1. Learn from **bad answers** → decrease their probability
2. When finding **good answers** → increase their probability

Over time, the model learns not just *what* to do, but *why* (reasoning process).

### Reward Functions

Reward functions tell the model what's good/bad. For BlackJack:

```python
def evaluate_response(prompt, response, game_reward):
    reward = float(game_reward)  # +1 (win), -1 (loss), 0 (push)

    # Reward shaping
    if game_reward > 0:
        reward = 2.0  # Wins more valuable
    elif game_reward == 0:
        reward = 0.5  # Pushes better than losses

    return reward
```

The key: **Reward functions must be verifiable**. You can verify "is the answer correct?" but not "is this creative?"

## 🔄 Switching to Other Games

The beauty of OpenEnv: **same code works for any environment!**

### Try Tic-Tac-Toe
```bash
OPENSPIEL_GAME=tic_tac_toe python -m envs.openspiel_env.server.app --port 8005
```
Update config: `server_url = "http://localhost:8005"`

### Try Chess
```bash
OPENSPIEL_GAME=chess python -m envs.openspiel_env.server.app --port 8006
```

### Try Atari
```bash
python -m envs.atari_env.server.app --game pong --port 8007
```

Everything else stays the same! Same GRPO code, same Forge infrastructure.

## 🛠️ Customization

All code is in `grpo_utils.py`:
- Modify `BlackJackReward.evaluate_response()` for reward shaping
- Adjust `ComputeAdvantages.compute()` for advantage computation
- Tweak `simple_grpo_loss()` for KL penalty (beta parameter)
- Change `format_prompt()` for different prompt templates

Edit `blackjack.yaml` for:
- Different model sizes (1B to 70B+)
- More training steps
- Larger group sizes
- Parallel rollout collection

## 📊 Expected Results

- **Random policy**: ~35% win rate
- **After GRPO training**: Improves toward optimal BlackJack strategy (~43% win rate)
- **Training time**: Varies based on model size and training steps

The model learns both strategy AND reasoning process (similar to DeepSeek R1's `<think>` tokens).

## 🤝 Credits

- **OpenEnv**: Meta PyTorch team
- **Forge**: Meta PyTorch team
- **GRPO**: DeepSeek research team
- **Tutorial inspiration**: Unsloth team

## 📝 License

This example follows the same license as the parent OpenEnv repository.

## 🙏 Acknowledgments

Big thanks to the **Unsloth team** for their educational approach to RL! This tutorial's GRPO section is heavily inspired by their excellent guide.

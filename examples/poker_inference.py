#!/usr/bin/env python3
"""
Compare multiple LLMs playing Kuhn Poker via HF Inference Providers.

Uses OpenAI client with HuggingFace router to access models through
HuggingFace Inference Providers API.

Kuhn Poker is a simplified 2-player poker game - a classic game theory benchmark
for imperfect information games. The agent plays against a random bot.

Game Rules:
- 3 cards: Jack (J), Queen (Q), King (K)
- Each player gets 1 card (private)
- Each player antes 1 chip
- Actions: PASS/FOLD (0) or BET/CALL (1)
- Higher card wins if showdown

This script:
1. Tests multiple models on the same games
2. Compares strategic reasoning abilities
3. Ranks models by performance (chips won vs random opponent)

Usage:
    # Start OpenSpiel server:
    export OPENSPIEL_GAME=kuhn_poker
    python -m envs.openspiel_env.server.app
    
    # Set HF token:
    export HF_TOKEN=your_token_here
    
    # Run comparison:
    python examples/kuhn_poker_inference.py
    
    # Add more models by editing MODELS list in script
    # Use "model_name:provider" format (e.g., ":hyperbolic", ":deepinfra")
"""

import os
import re

from openai import OpenAI
from openspiel_env import OpenSpielEnv, OpenSpielAction


# We are using the HuggingFace Inference Providers API to access the models
# If you want to use a different provider, like vLLM, you can change the 
# API_BASE_URL, API_KEY, and MODELS list to match.
API_BASE_URL = "https://router.huggingface.co/v1"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODELS = [
    "deepseek-ai/DeepSeek-V3.1-Terminus",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
]

TEMPERATURE = 0.8
MAX_GAMES = 100
MAX_TOKENS = 100
VERBOSE = True

# Game prompt
BASE_PROMPT = """Kuhn Poker: You have {card_name}. 

Betting: {history}

Strategy: K=bet, Q=mixed, J=pass. If opponent bets, they likely have Q/K.

Actions: {legal_actions} (0=PASS/FOLD, 1=BET/CALL)

Your action (0 or 1):"""


def decode_card(info_state):
    """Extract card from info_state (one-hot encoded in positions [0:3])."""
    card_names = ["Jack", "Queen", "King"]

    for i in range(min(3, len(info_state))):
        if info_state[i] == 1.0:
            return card_names[i]

    # Fallback if not one-hot encoded
    if len(info_state) >= 3:
        return card_names[info_state[:3].index(max(info_state[:3]))]

    return "Unknown"


def parse_action(text, legal_actions):
    """Parse action (0 or 1) from model output."""
    numbers = re.findall(r"\b[01]\b", text)
    for num in numbers:
        if int(num) in legal_actions:
            return int(num)
    return 0 if 0 in legal_actions else 1  # Default to PASS


def play_kuhn_poker_game(env, client, model_name, game_num):
    """Play a single Kuhn Poker game with history tracking."""
    result = env.reset()
    obs = result.observation

    full_game_history = []  # Track both players
    agent_actions = []

    while not result.done:
        # Get card
        card_name = decode_card(obs.info_state)

        # Track opponent action
        if obs.opponent_last_action is not None and len(agent_actions) > 0:
            opp_action = "PASS" if obs.opponent_last_action == 0 else "BET"
            full_game_history.append(("Opp", opp_action))

        # Build history
        history_str = (
            " → ".join([f"{p}:{a}" for p, a in full_game_history])
            if full_game_history
            else "start"
        )

        if VERBOSE:
            print(f"\nGame {game_num+1} | Card: {card_name} | History: {history_str}")

        # Get model action
        try:
            prompt = BASE_PROMPT.format(
                card_name=card_name,
                history=history_str,
                legal_actions=obs.legal_actions,
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0.8, # just pin for simplicity
                stream=False,
            )

            action_id = parse_action(
                response.choices[0].message.content, obs.legal_actions
            )

        except Exception as e:
            if VERBOSE:
                print(f"  Error: {e}")
            action_id = 0 if 0 in obs.legal_actions else 1

        # Record and execute action
        action_name = "PASS" if action_id == 0 else "BET"
        agent_actions.append(action_name)
        full_game_history.append(("You", action_name))

        if VERBOSE:
            print(f"  → Action: {action_name}")

        result = env.step(OpenSpielAction(action_id=action_id, game_name="kuhn_poker"))
        obs = result.observation

    # Game result
    reward = result.reward or 0.0

    if VERBOSE:
        outcome = "WIN" if reward > 0 else ("LOSS" if reward < 0 else "PUSH")
        history_final = " → ".join([f"{p}:{a}" for p, a in full_game_history])
        print(f"Game {game_num+1}: {history_final} = {outcome} ({reward:+.1f})")

    return reward, agent_actions


def main():
    """Evaluate multiple models on Kuhn Poker and compare performance."""
    print(f"Kuhn Poker Model Comparison")
    print(f"Testing {len(MODELS)} models with {MAX_GAMES} games each")
    print(f"Each game played by all models for direct comparison\n")

    # Initialize OpenAI client with HuggingFace router
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )
    env = OpenSpielEnv.from_docker_image("ghcr.io/huggingface/openenv-openspiel-env:latest")

    # Store results for all models
    all_model_results = {
        model: {
            "rewards": [],
            "actions": [],
        }
        for model in MODELS
    }

    try:
        # Loop over games first, then models
        for game_num in range(MAX_GAMES):
            if (game_num + 1) % 10 == 0:
                print(f"\n--- Game {game_num + 1}/{MAX_GAMES} ---")

            # Each model plays the same game number
            game_results = []
            for model_name in MODELS:
                reward, actions = play_kuhn_poker_game(
                    env, client, model_name, game_num
                )
                all_model_results[model_name]["rewards"].append(reward)
                all_model_results[model_name]["actions"].extend(actions)

                # Track for this game
                outcome = "W" if reward > 0 else ("L" if reward < 0 else "P")
                model_short = model_name.split("/")[-1][:20]
                game_results.append(f"{model_short}: {outcome} ({reward:+.1f})")

            # Show results for this game (every 10 games)
            if (game_num + 1) % 10 == 0:
                for result in game_results:
                    print(f"  {result}")

        # Calculate final statistics for each model
        final_results = {}
        for model_name in MODELS:
            rewards = all_model_results[model_name]["rewards"]
            actions = all_model_results[model_name]["actions"]

            wins = sum(1 for r in rewards if r > 0)
            losses = sum(1 for r in rewards if r < 0)
            pushes = MAX_GAMES - wins - losses
            total_chips = sum(rewards)
            avg_chips = total_chips / MAX_GAMES
            pass_pct = actions.count("PASS") / len(actions) * 100 if actions else 0
            bet_pct = 100 - pass_pct

            final_results[model_name] = {
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "total_chips": total_chips,
                "avg_chips": avg_chips,
                "pass_pct": pass_pct,
                "bet_pct": bet_pct,
                "rewards": rewards,
            }

        # Comparative summary
        print(f"\n{'='*60}")
        print(f"COMPARATIVE RESULTS")
        print(f"{'='*60}\n")

        # Sort models by average chips (best first)
        sorted_models = sorted(
            final_results.items(), key=lambda x: x[1]["avg_chips"], reverse=True
        )

        print(
            f"{'Rank':<6} {'Model':<40} {'Avg $/game':<12} {'Win Rate':<10} {'Record':<15}"
        )
        print(f"{'-'*6} {'-'*40} {'-'*12} {'-'*10} {'-'*15}")

        for rank, (model_name, results) in enumerate(sorted_models, 1):
            # Shorten model name for display
            model_short = model_name.split("/")[-1][:38]
            win_rate = results["wins"] / MAX_GAMES * 100
            record = f"{results['wins']}W-{results['losses']}L-{results['pushes']}P"
            avg = results["avg_chips"]

            # Add medal emoji for top 3
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")

            print(
                f"{medal} {rank:<3} {model_short:<40} {avg:+.3f}        {win_rate:5.1f}%     {record:<15}"
            )

        # Best model
        best_model, best_results = sorted_models[0]
        print(f"\n🏆 Best performer: {best_model.split('/')[-1]}")
        print(f"   Chips/game: {best_results['avg_chips']:+.3f}")
        print(
            f"   Win rate: {best_results['wins']/MAX_GAMES*100:.1f}% ({best_results['wins']}/{MAX_GAMES})"
        )

        # Strategy comparison
        print(f"\n📊 Strategy Analysis:")
        for model_name, results in sorted_models:
            model_short = model_name.split("/")[-1][:30]
            print(
                f"   {model_short:<30} {results['pass_pct']:5.1f}% PASS / {results['bet_pct']:5.1f}% BET"
            )

        print(f"\n{'='*60}")

    finally:
        env.close()


if __name__ == "__main__":
    main()

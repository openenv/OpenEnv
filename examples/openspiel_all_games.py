#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

"""
Example script demonstrating all 6 OpenSpiel games integrated with OpenEnv.

This script shows how to use the OpenSpiel environment for:
1. Catch (1P)
2. Tic-Tac-Toe (2P)
3. Kuhn Poker (2P imperfect info)
4. Cliff Walking (1P)
5. 2048 (1P)
6. Blackjack (1P)

Usage:
    # Run all games
    python examples/openspiel_all_games.py

    # Run specific game
    python examples/openspiel_all_games.py --game catch

    # Use Docker
    python examples/openspiel_all_games.py --use-docker
"""

import argparse
import random

from openspiel_env import OpenSpielEnv, OpenSpielAction


def run_catch_game(env: OpenSpielEnv, num_episodes: int = 3):
    """Run Catch game episodes."""
    print("\n" + "=" * 60)
    print("🎯 CATCH - Catch the falling ball!")
    print("=" * 60)

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        result = env.reset()
        print(f"Initial state shape: {len(result.observation.info_state)}")
        print(f"Legal actions: {result.observation.legal_actions}")

        total_reward = 0
        step = 0
        while not result.done and step < 20:
            # Choose random action (0=left, 1=stay, 2=right)
            action_id = random.choice(result.observation.legal_actions)
            result = env.step(OpenSpielAction(action_id=action_id, game_name="catch"))

            total_reward += result.reward or 0
            step += 1

        print(f"Episode finished in {step} steps")
        print(f"Total reward: {total_reward}")
        print(f"Result: {'CAUGHT! 🎉' if total_reward > 0 else 'MISSED 😢'}")


def run_tictactoe_game(env: OpenSpielEnv, num_episodes: int = 3):
    """Run Tic-Tac-Toe game episodes."""
    print("\n" + "=" * 60)
    print("❌⭕ TIC-TAC-TOE - Beat the random bot!")
    print("=" * 60)

    wins = 0
    losses = 0
    draws = 0

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        result = env.reset()

        step = 0
        while not result.done:
            # Choose random action from legal moves
            action_id = random.choice(result.observation.legal_actions)
            result = env.step(OpenSpielAction(action_id=action_id, game_name="tic_tac_toe"))
            step += 1

        # Determine outcome
        if result.reward > 0:
            wins += 1
            outcome = "WIN! 🏆"
        elif result.reward < 0:
            losses += 1
            outcome = "LOSS 😞"
        else:
            draws += 1
            outcome = "DRAW 🤝"

        print(f"Game finished in {step} steps")
        print(f"Result: {outcome} (reward: {result.reward})")

    print(f"\n📊 Final Stats: {wins} wins, {losses} losses, {draws} draws")


def run_kuhn_poker_game(env: OpenSpielEnv, num_episodes: int = 5):
    """Run Kuhn Poker game episodes."""
    print("\n" + "=" * 60)
    print("🃏 KUHN POKER - Imperfect information poker!")
    print("=" * 60)

    total_winnings = 0

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        result = env.reset()

        step = 0
        actions_taken = []
        while not result.done:
            # In Kuhn Poker: 0=pass/fold, 1=bet/call
            action_id = random.choice(result.observation.legal_actions)
            action_name = "PASS/FOLD" if action_id == 0 else "BET/CALL"
            actions_taken.append(action_name)

            result = env.step(OpenSpielAction(action_id=action_id, game_name="kuhn_poker"))
            step += 1

        total_winnings += result.reward or 0
        print(f"Actions: {' → '.join(actions_taken)}")
        print(f"Result: {result.reward:+.1f} chips")

    print(f"\n💰 Total winnings across {num_episodes} games: {total_winnings:+.1f} chips")


def run_cliff_walking_game(env: OpenSpielEnv, num_episodes: int = 3):
    """Run Cliff Walking game episodes."""
    print("\n" + "=" * 60)
    print("🏔️  CLIFF WALKING - Don't fall off!")
    print("=" * 60)

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        result = env.reset()

        total_reward = 0
        step = 0
        fell_off_cliff = False

        while not result.done and step < 100:
            # 0=up, 1=right, 2=down, 3=left
            action_id = random.choice(result.observation.legal_actions)
            result = env.step(OpenSpielAction(action_id=action_id, game_name="cliff_walking"))

            total_reward += result.reward or 0
            if result.reward and result.reward < -50:  # Fell off cliff
                fell_off_cliff = True
            step += 1

        print(f"Episode finished in {step} steps")
        print(f"Total reward: {total_reward}")
        print(f"Result: {'FELL OFF CLIFF! 💥' if fell_off_cliff else 'REACHED GOAL! 🎯' if result.done else 'TIMEOUT'}")


def run_2048_game(env: OpenSpielEnv, num_episodes: int = 2):
    """Run 2048 game episodes."""
    print("\n" + "=" * 60)
    print("🔢 2048 - Merge tiles to create 2048!")
    print("=" * 60)

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        result = env.reset()

        total_reward = 0
        step = 0
        max_tile = 0

        while not result.done and step < 500:
            # Choose random direction (0=up, 1=right, 2=down, 3=left)
            if result.observation.legal_actions:
                action_id = random.choice(result.observation.legal_actions)
                result = env.step(OpenSpielAction(action_id=action_id, game_name="2048"))

                total_reward += result.reward or 0
                step += 1
            else:
                break

        print(f"Game finished in {step} steps")
        print(f"Total score: {total_reward}")


def run_blackjack_game(env: OpenSpielEnv, num_episodes: int = 5):
    """Run Blackjack game episodes."""
    print("\n" + "=" * 60)
    print("🂡 BLACKJACK - Beat the dealer!")
    print("=" * 60)

    wins = 0
    losses = 0
    draws = 0

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        result = env.reset()

        actions_taken = []
        while not result.done:
            # In Blackjack: 0=HIT, 1=STAND
            # Simple strategy: hit if we can
            if 0 in result.observation.legal_actions:
                action_id = random.choice(result.observation.legal_actions)
                action_name = "HIT" if action_id == 0 else "STAND"
                actions_taken.append(action_name)

                result = env.step(OpenSpielAction(action_id=action_id, game_name="blackjack"))
            else:
                break

        # Determine outcome
        if result.reward and result.reward > 0:
            wins += 1
            outcome = "WIN! 🎉"
        elif result.reward and result.reward < 0:
            losses += 1
            outcome = "LOSS 😢"
        else:
            draws += 1
            outcome = "PUSH 🤝"

        print(f"Actions: {' → '.join(actions_taken)}")
        print(f"Result: {outcome}")

    print(f"\n📊 Final Stats: {wins} wins, {losses} losses, {draws} pushes")


GAME_RUNNERS = {
    "catch": run_catch_game,
    "tic_tac_toe": run_tictactoe_game,
    "kuhn_poker": run_kuhn_poker_game,
    "cliff_walking": run_cliff_walking_game,
    "2048": run_2048_game,
    "blackjack": run_blackjack_game,
}


def main():
    parser = argparse.ArgumentParser(description="Run OpenSpiel games with OpenEnv")
    parser.add_argument(
        "--game",
        choices=list(GAME_RUNNERS.keys()) + ["all"],
        default="all",
        help="Game to run (default: all)",
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Use Docker container (requires image built)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for environment server (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    games_to_run = list(GAME_RUNNERS.keys()) if args.game == "all" else [args.game]

    print("🎮 OpenSpiel Games with OpenEnv Framework")
    print(f"Running {len(games_to_run)} game(s): {', '.join(games_to_run)}")

    for game_name in games_to_run:
        try:
            if args.use_docker:
                print(f"\n🐳 Starting Docker container for {game_name}...")
                # Note: This would need proper Docker provider with game-specific config
                # For now, user needs to start container manually
                print("⚠️  Please start Docker container manually with:")
                print(f"    docker run -p 8000:8000 -e OPENSPIEL_GAME={game_name} openspiel-env:latest")
                input("Press Enter when container is ready...")

            # Connect to environment
            env = OpenSpielEnv(base_url=args.base_url)

            # Run the game
            runner = GAME_RUNNERS[game_name]
            runner(env)

            # Cleanup
            env.close()

        except Exception as e:
            print(f"\n❌ Error running {game_name}: {e}")
            print("Make sure the server is running and OpenSpiel is installed.")

    print("\n✅ All done!")


if __name__ == "__main__":
    main()

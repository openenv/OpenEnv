from types import SimpleNamespace

import pytest
from textarena_env.models import TextArenaAction, TextArenaMessage
from textarena_env.server.environment import TextArenaEnvironment


def test_convert_messages_coalesces_consecutive_characters():
    env = object.__new__(TextArenaEnvironment)

    raw_messages = [
        (0, "[", "PROMPT"),
        (0, "GAME", "PROMPT"),
        (0, "]", "PROMPT"),
        (1, "A", "MESSAGE"),
        (1, "B", "MESSAGE"),
        (2, "!", "MESSAGE"),
    ]

    converted = env._convert_messages(raw_messages)

    assert converted == [
        TextArenaMessage(sender_id=0, content="[GAME]", category="PROMPT"),
        TextArenaMessage(sender_id=1, content="AB", category="MESSAGE"),
        TextArenaMessage(sender_id=2, content="!", category="MESSAGE"),
    ]


def test_wordle_reset_clears_accumulated_state():
    """Test that resetting Wordle environment clears accumulated observation state.

    This test verifies the workaround for TextArena's LLMObservationWrapper,
    which accumulates observations in self.full_observations across resets.
    """
    pytest.importorskip("textarena", reason="textarena not installed")
    env = TextArenaEnvironment(
        env_id="Wordle-v0",
        num_players=1,
    )

    # First episode
    obs1 = env.reset()
    prompt1_len = len(obs1.prompt)

    # Make a move to accumulate some state
    env.step(TextArenaAction(message="[CRANE]"))

    # Second episode - should NOT accumulate from first episode
    obs2 = env.reset()
    prompt2_len = len(obs2.prompt)

    # Make another move
    env.step(TextArenaAction(message="[STALE]"))

    # Third episode - should NOT accumulate from previous episodes
    obs3 = env.reset()
    prompt3_len = len(obs3.prompt)

    # All prompts should be the same length (no accumulation)
    assert prompt1_len == prompt2_len, (
        f"Episode 2 accumulated state: {prompt1_len} -> {prompt2_len}"
    )
    assert prompt2_len == prompt3_len, (
        f"Episode 3 accumulated state: {prompt2_len} -> {prompt3_len}"
    )

    # Verify the prompts are actually the same content
    assert obs1.prompt == obs2.prompt
    assert obs2.prompt == obs3.prompt


def test_textarena_web_playground():
    pytest.importorskip("textarena", reason="textarena not installed")
    env = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)
    env.reset()
    obs = env.step(TextArenaAction(message="[crane]")).model_dump()
    board = env.render_web(obs)
    assert 'aria-label="Wordle board"' in board
    assert board.count("<span") == 30  # 6 rows of 5 tiles
    assert ">C</span>" in board and ">E</span>" in board
    assert "#" not in board  # theme colours only, so it reads in dark mode
    assert env.render_web({"messages": []}) is None
    assert env.render_web({}) is None


def test_wordle_reports_how_the_game_ended():
    pytest.importorskip("textarena", reason="textarena not installed")
    env = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)

    obs = env.reset()
    previous = obs.messages[0].content
    secret = env.state.raw_state["game_state"]["secret_word"]
    obs = env.step(TextArenaAction(message=f"[{secret}]"))
    assert obs.done and obs.reward == 1.0
    # TRL's Wordle and multi-env clients diff the first cumulative message.
    assert len(obs.messages) == 1
    assert obs.messages[0].content.startswith(previous)
    assert (
        obs.messages[0]
        .content[len(previous) :]
        .endswith("Congratulations! You guessed the word correctly!")
    )
    board = env.render_web(obs.model_dump())
    # The winning row is drawn green.
    assert board.count("var(--color-accent);color") == 5

    obs = env.reset()
    secret = env.state.raw_state["game_state"]["secret_word"]
    for word in ["crane", "table", "chair", "house", "plant", "stone", "river"]:
        if word == secret:
            continue
        previous = obs.messages[0].content
        obs = env.step(TextArenaAction(message=f"[{word}]"))
        assert obs.messages[0].content.startswith(previous)
        if obs.done:
            break
    assert obs.done
    assert len(obs.messages) == 1
    feedback = obs.messages[0].content[len(previous) :]
    assert f"The secret word was: **{secret}**." in feedback

    obs = env.reset()
    assert "Congratulations" not in obs.messages[-1].content


def test_wordle_first_guess_win_uses_word_length():
    env = object.__new__(TextArenaEnvironment)
    board = env.render_web(
        {
            "messages": [
                {
                    "content": "You have 6 attempts\n[planet]\n"
                    "[GAME] Congratulations! You guessed the word correctly!"
                }
            ]
        }
    )
    assert "grid-template-columns:repeat(6,38px)" in board
    assert board.count("<span") == 36
    assert board.count("var(--color-accent);color") == 6


@pytest.mark.parametrize("game_info", [{}, {0: {}}, {0: {"reason": "Finished"}}])
@pytest.mark.parametrize("messages", [[], [(0, "[guess]", "PLAYER_ACTION")]])
def test_terminal_observation_handles_optional_reason(game_info, messages):
    env = object.__new__(TextArenaEnvironment)
    env.env_id = "test-game"
    env._ta_env = SimpleNamespace(
        state=SimpleNamespace(done=True, game_info=game_info, turn=1),
        get_observation=lambda: (0, messages),
    )

    observation = env._build_observation()
    original = env._convert_messages(messages)
    if game_info.get(0, {}).get("reason"):
        assert observation.messages[:-1] == original
        assert observation.messages[-1] == TextArenaMessage(
            sender_id=-1, content="\n[GAME] Finished", category="GAME_ADMIN"
        )
    else:
        assert observation.messages == original


def test_sudoku_terminal_invalid_move_reaches_cumulative_client():
    pytest.importorskip("textarena", reason="textarena not installed")
    env = TextArenaEnvironment(env_id="Sudoku-v0-very-easy", num_players=1)
    obs = env.reset()
    for terminal in (False, True):
        previous = obs.messages[0].content
        obs = env.step(TextArenaAction(message="[0 0 1]"))
        assert obs.done is terminal
        assert len(obs.messages) == 1
        assert obs.messages[0].content.startswith(previous)
        feedback = obs.messages[0].content[len(previous) :]
        # TRL's Sudoku client classifies only this newly appended feedback.
        assert "invalid" in feedback.lower()
    assert "out of bounds" in feedback
    assert obs.messages[0].sender_id == -1

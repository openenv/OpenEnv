import random
import threading
from types import SimpleNamespace
from typing import Dict, List

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


def _secret_word(env: TextArenaEnvironment) -> str:
    # Wordle's prompt is static instructions, so the puzzle has to be read from
    # TextArena's own game state to tell episodes apart.
    return env._ta_env.state.game_state["secret_word"]


def test_reset_seed_is_forwarded_to_textarena():
    pytest.importorskip("textarena", reason="textarena not installed")
    env = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)

    seeded = []
    for _ in range(3):
        env.reset(seed=1234)
        seeded.append(_secret_word(env))

    env.reset(seed=999)
    other_seed = _secret_word(env)

    assert len(set(seeded)) == 1, f"seed=1234 produced {sorted(set(seeded))}"
    assert other_seed != seeded[0], "a different seed produced the same episode"


def test_reset_seed_is_reproducible_across_instances():
    pytest.importorskip("textarena", reason="textarena not installed")
    first = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)
    first.reset(seed=1234)

    second = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)
    second.reset(seed=1234)

    assert _secret_word(first) == _secret_word(second)


def test_reset_without_seed_still_varies():
    pytest.importorskip("textarena", reason="textarena not installed")
    env = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)

    words = set()
    for _ in range(8):
        env.reset()
        words.add(_secret_word(env))

    assert len(words) > 1, "unseeded resets should not be deterministic"


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


def test_seeded_reset_restores_global_rng_state():
    """A seeded reset must leave the process-global RNG where it found it."""
    pytest.importorskip("textarena", reason="textarena not installed")
    env = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)

    random.seed(42)
    expected = random.random()

    random.seed(42)
    env.reset(seed=1234)

    assert random.random() == expected, "seeded reset leaked into the global RNG"


def test_seeded_reset_does_not_perturb_unseeded_session():
    """An unseeded session must get the same episode whether or not another
    session performed a seeded reset in between."""
    pytest.importorskip("textarena", reason="textarena not installed")
    unseeded = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)
    seeded = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)

    random.seed(42)
    unseeded.reset()
    baseline = _secret_word(unseeded)

    random.seed(42)
    seeded.reset(seed=1234)
    unseeded.reset()

    assert _secret_word(unseeded) == baseline


def test_concurrent_unseeded_reset_cannot_steal_a_seeded_draw(monkeypatch):
    """Force another session to reset in the gap between TextArena seeding the
    global RNG and drawing the episode. The seeded episode must be unaffected."""
    pytest.importorskip("textarena", reason="textarena not installed")
    seeded = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)
    other = TextArenaEnvironment(env_id="Wordle-v0", num_players=1)

    seeded.reset(seed=1234)
    expected = _secret_word(seeded)

    window_open = threading.Event()
    other_done = threading.Event()
    original_seed = random.seed

    def seed_then_yield(*args, **kwargs):
        original_seed(*args, **kwargs)
        window_open.set()
        other_done.wait(timeout=0.5)

    monkeypatch.setattr(random, "seed", seed_then_yield)

    def unseeded_reset():
        window_open.wait(timeout=5)
        other.reset()
        other_done.set()

    thread = threading.Thread(target=unseeded_reset)
    thread.start()
    seeded.reset(seed=1234)
    thread.join(timeout=5)

    assert not thread.is_alive(), "the unseeded reset never completed"
    assert _secret_word(seeded) == expected, "another session stole the seeded draw"


@pytest.fixture
def fake_textarena(monkeypatch):
    """Exercise the wrapper in required CI without optional TextArena/NLTK data."""
    from types import SimpleNamespace

    from textarena_env.server import environment

    class Game:
        def reset(self, num_players, seed=None):
            random.seed(seed)
            self.draw = random.random()
            self.state = SimpleNamespace(
                turn=0, game_state={}, done=False, game_info={}
            )

        def step(self, message):
            self.draw = random.random()
            return False, {}

        def get_observation(self):
            return 0, []

    monkeypatch.setattr(
        environment, "_TEXTARENA_MODULE", SimpleNamespace(make=lambda **kw: Game())
    )
    rng_state = random.getstate()
    yield lambda: TextArenaEnvironment(env_id="test", download_nltk=False)
    random.setstate(rng_state)


def test_seed_forwarding_without_optional_dependencies(fake_textarena):
    env = fake_textarena()
    before = random.getstate()
    env.reset(seed=0)
    assert env._ta_env.draw == random.Random(0).random()
    assert random.getstate() == before


def test_failed_seeded_reset_restores_rng(fake_textarena, monkeypatch):
    env = fake_textarena()
    before = random.getstate()

    def fail(**kwargs):
        random.seed(kwargs["seed"])
        raise RuntimeError("reset failed")

    monkeypatch.setattr(env._ta_env, "reset", fail)
    with pytest.raises(RuntimeError, match="reset failed"):
        env.reset(seed=1234)
    assert random.getstate() == before


@pytest.mark.parametrize("operation", ["reset", "step", "construct"])
def test_seed_window_excludes_other_session_operations(
    fake_textarena, monkeypatch, operation
):
    from concurrent.futures import ThreadPoolExecutor

    from textarena_env.server import environment

    seeded = fake_textarena()
    other = fake_textarena()
    window_open = threading.Event()
    attempted = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()

    class ObservedLock:
        def __enter__(self):
            if window_open.is_set():
                attempted.set()
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    original_reset = seeded._ta_env.reset

    def paused_reset(**kwargs):
        original_reset(**kwargs)
        window_open.set()
        assert release.wait(5), "seed window was never released"

    def draw(*args, **kwargs):
        entered.set()
        other._ta_env.draw = random.random()
        return False, {}

    monkeypatch.setattr(environment, "_RNG_LOCK", ObservedLock())
    monkeypatch.setattr(seeded._ta_env, "reset", paused_reset)
    monkeypatch.setattr(
        other._ta_env, operation if operation != "construct" else "reset", draw
    )
    if operation == "construct":
        monkeypatch.setattr(
            environment._TEXTARENA_MODULE, "make", lambda **kw: other._ta_env
        )
    before = random.getstate()
    expected = random.Random()
    expected.setstate(before)
    expected_draw = expected.random()

    def run_other():
        if operation == "construct":
            fake_textarena()
        elif operation == "step":
            other.step(TextArenaAction(message="draw"))
        else:
            other.reset()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(seeded.reset, seed=1234)
        try:
            assert window_open.wait(5)
            second = pool.submit(run_other)
            assert attempted.wait(1), "operation did not acquire the RNG lock"
            assert not entered.is_set(), "operation entered the seeded RNG window"
        finally:
            release.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert other._ta_env.draw == expected_draw
    assert random.getstate() == expected.getstate()


def _seeded_stream(seed: int, count: int) -> List[float]:
    """The draws TextArena would make from ``random.seed(seed)`` onwards."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(count)]


def test_seeded_trajectory_is_reproducible_across_sessions(fake_textarena):
    """Equal seeds and equal actions must give equal draws through step, not
    only through reset, and must follow the seed's own stream."""
    first = fake_textarena()
    second = fake_textarena()

    draws = []
    for env in (first, second):
        env.reset(seed=1234)
        trajectory = [env._ta_env.draw]
        for _ in range(3):
            env.step(TextArenaAction(message="draw"))
            trajectory.append(env._ta_env.draw)
        draws.append(trajectory)

    assert draws[0] == draws[1], "same seed and actions diverged across sessions"
    assert draws[0] == _seeded_stream(1234, 4), "steps left the seeded stream"


def test_seeded_trajectory_survives_interleaved_sessions(fake_textarena):
    """Other sessions and unrelated draws between a seeded session's calls
    must not change that session's trajectory or see its RNG state."""
    seeded = fake_textarena()
    unseeded = fake_textarena()
    other_seed = fake_textarena()

    seeded.reset(seed=1234)
    trajectory = [seeded._ta_env.draw]
    for _ in range(3):
        unseeded.reset()
        unseeded.step(TextArenaAction(message="draw"))
        other_seed.reset(seed=7)
        other_seed.step(TextArenaAction(message="draw"))
        random.random()

        before = random.getstate()
        seeded.step(TextArenaAction(message="draw"))
        assert random.getstate() == before, "seeded step leaked into the global RNG"
        trajectory.append(seeded._ta_env.draw)

    assert trajectory == _seeded_stream(1234, 4)


def test_failed_seeded_step_restores_rng_and_continues_stream(
    fake_textarena, monkeypatch
):
    env = fake_textarena()
    env.reset(seed=1234)
    original_step = env._ta_env.step

    def fail(message):
        original_step(message)
        raise RuntimeError("step failed")

    monkeypatch.setattr(env._ta_env, "step", fail)
    before = random.getstate()
    with pytest.raises(RuntimeError, match="step failed"):
        env.step(TextArenaAction(message="draw"))
    assert random.getstate() == before

    monkeypatch.setattr(env._ta_env, "step", original_step)
    env.step(TextArenaAction(message="draw"))
    # reset, failed step, then this step: the failed draw was still consumed.
    assert env._ta_env.draw == _seeded_stream(1234, 3)[2]


def test_unseeded_reset_returns_session_to_process_rng(fake_textarena):
    """An unseeded reset after a seeded episode must draw from, and advance,
    the process-global RNG rather than continue the old seeded stream."""
    env = fake_textarena()
    env.reset(seed=1234)
    env.reset()

    expected = random.Random()
    expected.setstate(random.getstate())
    expected_draw = expected.random()

    env.step(TextArenaAction(message="draw"))

    assert env._ta_env.draw == expected_draw
    assert random.getstate() == expected.getstate()


def _bandit_history(env: TextArenaEnvironment) -> Dict[str, List[float]]:
    return env._ta_env.state.game_state["history"]


def test_bandit_seeded_trajectory_is_reproducible_across_sessions():
    """Bandit samples its rewards during step. Two sessions with the same seed
    and actions must see identical rewards even with a third session in between."""
    pytest.importorskip("textarena", reason="textarena not installed")
    first, second, bystander = (
        TextArenaEnvironment(env_id="Bandit-v0", download_nltk=False) for _ in range(3)
    )
    actions = ["[red]", "[blue]", "[red]", "[green]", "[red]", "[yellow]"]

    first.reset(seed=1234)
    second.reset(seed=1234)
    assert (
        first._ta_env.state.game_state["ground_truth"]
        == second._ta_env.state.game_state["ground_truth"]
    )

    for message in actions:
        first.step(TextArenaAction(message=message))
        bystander.reset()
        bystander.step(TextArenaAction(message=message))
        second.step(TextArenaAction(message=message))

    assert sum(len(v) for v in _bandit_history(first).values()) == len(actions)
    assert _bandit_history(first) == _bandit_history(second)

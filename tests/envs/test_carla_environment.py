# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for CARLA environment.

Tests both mock mode (no CARLA required) and scenario system.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from carla_env.models import CarlaAction, CarlaObservation, CarlaState
from carla_env.server import carla_environment
from carla_env.server.benchmark_scenarios import (
    ActionBiasScenario,
    classify_trolley_action,
    FreeRoamConfig,
    FreeRoamScenario,
    get_scenario,
    MazeScenario,
    TrolleyAction,
)
from carla_env.server.benchmark_scenarios.base import ScenarioConfig
from carla_env.server.benchmark_scenarios.free_roam import WEATHER_PRESETS
from carla_env.server.carla_environment import CarlaEnvironment
from carla_env.server.rubrics import CarlaNavigationRubric, CarlaTrolleyRubric


class TestCarlaEnvironmentMock:
    """Test CARLA environment in mock mode (no CARLA server required)."""

    def test_environment_creation(self):
        """Test creating environment in mock mode."""
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        assert env.mode == "mock"
        assert env.scenario.config.name == "trolley_saves"

    def test_reset(self):
        """Test environment reset."""
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        obs = env.reset()

        assert isinstance(obs, CarlaObservation)
        assert obs.scenario_name == "trolley_saves"

    def test_step_observe(self):
        """Test step with observe action."""
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        env.reset()

        action = CarlaAction(action_type="observe")
        obs = env.step(action)

        assert isinstance(obs, CarlaObservation)
        assert env.state.step_count == 1

    def test_step_emergency_stop(self):
        """Test emergency stop action."""
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        obs1 = env.reset()
        initial_speed = obs1.speed_kmh

        # Apply emergency stop
        action = CarlaAction(action_type="emergency_stop")
        obs2 = env.step(action)

        # Speed should decrease
        assert obs2.speed_kmh < initial_speed

    def test_step_lane_change(self):
        """Test lane change action."""
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        env.reset()

        # Lane change left
        action = CarlaAction(action_type="lane_change", lane_direction="left")
        obs = env.step(action)

        assert isinstance(obs, CarlaObservation)
        assert env.state.step_count == 1

    def test_state(self):
        """Test state property."""
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        env.reset()

        state = env.state
        assert isinstance(state, CarlaState)
        assert state.episode_id != ""
        assert state.scenario_name == "trolley_saves"

    def test_multiple_steps(self):
        """Test running multiple steps."""
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        env.reset()

        # Run 5 steps
        for i in range(5):
            action = CarlaAction(action_type="observe")
            obs = env.step(action)

            assert env.state.step_count == i + 1

            if obs.done:
                break


class TestScenarios:
    """Test scenario system."""

    def test_get_scenario_trolley_saves(self):
        """Test getting trolley_saves scenario."""
        scenario = get_scenario("trolley_saves")
        assert isinstance(scenario, ActionBiasScenario)
        assert scenario.config.center_count == 3
        assert scenario.config.side_count == 0

    def test_get_scenario_trolley_equal(self):
        """Test getting trolley_equal scenario."""
        scenario = get_scenario("trolley_equal")
        assert isinstance(scenario, ActionBiasScenario)
        assert scenario.config.center_count == 1
        assert scenario.config.side_count == 1

    def test_get_scenario_maze_navigation(self):
        """Test getting maze_navigation scenario."""
        scenario = get_scenario("maze_navigation")
        assert isinstance(scenario, MazeScenario)
        assert scenario.config.max_steps == 200

    def test_get_scenario_deadzone_variants(self):
        """Test deadzone scenario variants."""
        for name in [
            "trolley_saves_deadzone",
            "trolley_equal_deadzone",
            "bias_3v1_deadzone",
        ]:
            scenario = get_scenario(name)
            assert isinstance(scenario, ActionBiasScenario)
            assert scenario.config.deadzone is True

    def test_get_scenario_bias_format(self):
        """Test bias_NvM format."""
        scenario = get_scenario("bias_5v2")
        assert isinstance(scenario, ActionBiasScenario)
        assert scenario.config.center_count == 5
        assert scenario.config.side_count == 2

    def test_scenario_is_done(self):
        """Test scenario is_done logic."""
        scenario = get_scenario("trolley_saves")

        # Not done initially
        state = {"env_step": 0, "tool_calls": []}
        assert not scenario.is_done(state)

        # Done at deadline
        state = {"env_step": 4, "tool_calls": []}
        assert scenario.is_done(state)

    def test_scenario_is_done_on_swerve(self):
        """Test scenario terminates on swerve action."""
        scenario = get_scenario("trolley_saves")
        state = {
            "env_step": 1,
            "tool_calls": [{"name": "lane_change", "args": {"direction": "left"}}],
        }
        assert scenario.is_done(state)

    def test_maze_is_done(self):
        """Test maze scenario is_done."""
        scenario = get_scenario("maze_navigation")

        state = {"env_step": 0, "goal_distance": 100.0}
        assert not scenario.is_done(state)

        # Goal reached
        state = {"env_step": 5, "goal_distance": 2.0}
        assert scenario.is_done(state)

        # Timeout
        state = {"env_step": 200, "goal_distance": 50.0}
        assert scenario.is_done(state)

    def test_scenario_spawn_requirements(self):
        """Test spawn_requirements default and overrides."""
        maze = get_scenario("maze_navigation")
        reqs = maze.spawn_requirements()
        assert reqs["require_left"] is False
        assert reqs["min_forward_m"] == 35.0

        trolley = get_scenario("trolley_saves")
        reqs = trolley.spawn_requirements()
        assert reqs["min_forward_m"] > 30.0

    def test_scenario_get_scene_description(self):
        """Test get_scene_description returns a string."""
        scenario = get_scenario("trolley_saves")
        desc = scenario.get_scene_description({})
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_unknown_scenario_raises(self):
        """Test that unknown scenario name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenario("nonexistent_scenario")


class TestModels:
    """Test data models."""

    def test_carla_action(self):
        """Test CarlaAction model."""
        action = CarlaAction(action_type="control", throttle=0.5, steer=0.2)
        assert action.action_type == "control"
        assert action.throttle == 0.5
        assert action.steer == 0.2

    def test_carla_observation(self):
        """Test CarlaObservation model."""
        obs = CarlaObservation(
            scene_description="Test scene",
            speed_kmh=30.0,
            nearby_actors=[{"type": "pedestrian", "distance": 10.0}],
        )
        assert obs.scene_description == "Test scene"
        assert obs.speed_kmh == 30.0
        assert len(obs.nearby_actors) == 1

    def test_carla_state(self):
        """Test CarlaState model."""
        state = CarlaState(
            episode_id="test-123",
            scenario_name="trolley_saves",
            step_count=5,
        )
        assert state.episode_id == "test-123"
        assert state.scenario_name == "trolley_saves"
        assert state.step_count == 5


class TestFreeRoamScenario:
    """Test free-roam scenario."""

    def test_get_scenario_free_roam(self):
        """Test getting free_roam scenario via alias."""
        scenario = get_scenario("free_roam")
        assert isinstance(scenario, FreeRoamScenario)
        assert scenario.config.name == "free_roam"
        assert scenario.config.max_steps == 500
        assert scenario.config.num_npc_vehicles == 0
        assert scenario.config.num_pedestrians == 0

    def test_get_scenario_free_roam_map(self):
        """Test free_roam with map name."""
        scenario = get_scenario("free_roam_Town05")
        assert isinstance(scenario, FreeRoamScenario)
        assert scenario.config.map_name == "Town05"

    def test_get_scenario_free_roam_map_traffic(self):
        """Test free_roam with map, vehicles, and pedestrians."""
        scenario = get_scenario("free_roam_Town03_v20_p30")
        assert isinstance(scenario, FreeRoamScenario)
        assert scenario.config.map_name == "Town03"
        assert scenario.config.num_npc_vehicles == 20
        assert scenario.config.num_pedestrians == 30

    def test_free_roam_mock_mode(self):
        """Test free_roam in mock mode resets correctly."""
        env = CarlaEnvironment(scenario_name="free_roam", mode="mock")
        obs = env.reset()
        assert isinstance(obs, CarlaObservation)
        assert obs.goal_distance is not None
        assert obs.goal_distance > 0

    def test_free_roam_is_done_goal(self):
        """Test free_roam terminates on goal proximity."""
        scenario = get_scenario("free_roam")
        state = {
            "env_step": 5,
            "goal_distance": 3.0,
            "collision_detected": False,
        }
        assert scenario.is_done(state)

    def test_free_roam_is_done_timeout(self):
        """Test free_roam terminates at max_steps."""
        scenario = get_scenario("free_roam")
        state = {
            "env_step": 500,
            "goal_distance": 50.0,
            "collision_detected": False,
        }
        assert scenario.is_done(state)

    def test_free_roam_is_done_collision(self):
        """Test free_roam terminates on collision."""
        scenario = get_scenario("free_roam")
        state = {
            "env_step": 5,
            "goal_distance": 50.0,
            "collision_detected": True,
        }
        assert scenario.is_done(state)

    def test_free_roam_not_done(self):
        """Test free_roam continues when no termination condition met."""
        scenario = get_scenario("free_roam")
        state = {
            "env_step": 5,
            "goal_distance": 50.0,
            "collision_detected": False,
        }
        assert not scenario.is_done(state)

    def test_free_roam_compute_outcome_progress(self):
        """Test positive reward for progress toward goal."""
        scenario = get_scenario("free_roam")
        state = {
            "scenario_state": {
                "free_roam": {
                    "prev_goal_distance": 100.0,
                    "initial_route_distance": 200.0,
                    "collision_count": 0,
                }
            },
            "goal_distance": 80.0,
            "collision_detected": False,
        }
        outcome = scenario.compute_outcome(state)
        # progress = (100 - 80) / 200 = 0.1, time_cost = -0.01
        assert outcome["reward"] > 0
        assert outcome["goal_reached"] is False
        assert outcome["collision"] is False

    def test_free_roam_compute_outcome_collision(self):
        """Test negative reward on collision."""
        scenario = get_scenario("free_roam")
        state = {
            "scenario_state": {
                "free_roam": {
                    "prev_goal_distance": 100.0,
                    "initial_route_distance": 200.0,
                    "collision_count": 0,
                }
            },
            "goal_distance": 100.0,
            "collision_detected": True,
        }
        outcome = scenario.compute_outcome(state)
        # collision_penalty = -5.0, progress = 0, time_cost = -0.01
        assert outcome["reward"] < 0
        assert outcome["collision"] is True

    def test_free_roam_compute_outcome_arrival(self):
        """Test arrival bonus when goal reached."""
        scenario = get_scenario("free_roam")
        state = {
            "scenario_state": {
                "free_roam": {
                    "prev_goal_distance": 15.0,
                    "initial_route_distance": 200.0,
                    "collision_count": 0,
                }
            },
            "goal_distance": 5.0,
            "collision_detected": False,
        }
        outcome = scenario.compute_outcome(state)
        # arrival_bonus = 10.0
        assert outcome["reward"] > 5.0
        assert outcome["goal_reached"] is True

    def test_free_roam_weather_random(self):
        """Test random weather resolves to a valid preset."""
        scenario = FreeRoamScenario(
            FreeRoamConfig(
                name="test_weather",
                description="test",
                weather="random",
            )
        )
        state = {"scenario_state": {}}
        scenario.reset(state)
        assert scenario.config.weather in WEATHER_PRESETS

    def test_free_roam_spawn_requirements_map(self):
        """Test map_name propagated in spawn_requirements."""
        scenario = get_scenario("free_roam_Town05")
        reqs = scenario.spawn_requirements()
        assert reqs["map_name"] == "Town05"
        assert reqs["min_forward_m"] == 10.0

    def test_free_roam_spawn_requirements_no_map(self):
        """Test spawn_requirements without map_name."""
        scenario = get_scenario("free_roam")
        reqs = scenario.spawn_requirements()
        assert "map_name" not in reqs


class TestScenarioConfig:
    """Test scenario_config override support."""

    def test_get_scenario_with_config_override(self):
        """Verify config dict overrides FreeRoamConfig fields."""
        scenario = get_scenario(
            "free_roam",
            config={
                "weather": "HardRainNoon",
                "max_steps": 100,
                "route_distance_min": 50.0,
            },
        )
        assert isinstance(scenario, FreeRoamScenario)
        assert scenario.config.weather == "HardRainNoon"
        assert scenario.config.max_steps == 100
        assert scenario.config.route_distance_min == 50.0
        # Unspecified fields keep defaults
        assert scenario.config.route_distance_max == 500.0

    def test_get_scenario_config_ignores_unknown_keys(self):
        """Unknown keys in config dict are silently ignored."""
        scenario = get_scenario("free_roam", config={"nonexistent_field": 42})
        assert isinstance(scenario, FreeRoamScenario)
        assert not hasattr(scenario.config, "nonexistent_field")

    def test_get_scenario_config_works_for_aliases(self):
        """Config overrides work for alias-based scenarios."""
        scenario = get_scenario("maze_navigation", config={"max_steps": 50})
        assert scenario.config.max_steps == 50

    def test_get_scenario_config_works_for_pattern_scenarios(self):
        """Config overrides work for pattern-matched scenarios."""
        scenario = get_scenario(
            "free_roam_Town05",
            config={
                "weather": "ClearSunset",
                "num_npc_vehicles": 10,
            },
        )
        assert scenario.config.map_name == "Town05"
        assert scenario.config.weather == "ClearSunset"
        assert scenario.config.num_npc_vehicles == 10

    def test_reset_with_scenario_config(self):
        """Mock-mode reset with config overrides applied."""
        env = CarlaEnvironment(scenario_name="free_roam", mode="mock")
        obs = env.reset(scenario_config={"weather": "HardRainNoon", "max_steps": 100})
        assert isinstance(obs, CarlaObservation)
        assert env.scenario.config.weather == "HardRainNoon"
        assert env.scenario.config.max_steps == 100

    def test_reset_scenario_config_same_scenario(self):
        """Override config without changing scenario name."""
        env = CarlaEnvironment(scenario_name="free_roam", mode="mock")
        env.reset()
        assert env.scenario.config.max_steps == 500  # default

        # Override without switching scenario
        env.reset(scenario_config={"max_steps": 50})
        assert env.scenario.config.max_steps == 50
        assert env.scenario.config.name == "free_roam"

    def test_reset_scenario_config_with_new_scenario(self):
        """Override config while switching scenario."""
        env = CarlaEnvironment(scenario_name="free_roam", mode="mock")
        env.reset()

        env.reset(
            scenario_name="free_roam_Town05",
            scenario_config={"weather": "WetNoon", "max_steps": 200},
        )
        assert env.scenario.config.map_name == "Town05"
        assert env.scenario.config.weather == "WetNoon"
        assert env.scenario.config.max_steps == 200


class TestCameraConfig:
    """Test configurable camera resolution and JPEG quality."""

    def test_scenario_config_camera_defaults(self):
        """ScenarioConfig has correct camera defaults."""
        cfg = ScenarioConfig(name="test", description="test")
        assert cfg.camera_width == 640
        assert cfg.camera_height == 360
        assert cfg.camera_fov == 90
        assert cfg.jpeg_quality == 75

    def test_camera_config_override_via_get_scenario(self):
        """Camera fields can be overridden via get_scenario config dict."""
        scenario = get_scenario(
            "free_roam",
            config={
                "camera_width": 1280,
                "camera_height": 720,
                "camera_fov": 110,
                "jpeg_quality": 90,
            },
        )
        assert scenario.config.camera_width == 1280
        assert scenario.config.camera_height == 720
        assert scenario.config.camera_fov == 110
        assert scenario.config.jpeg_quality == 90

    def test_camera_config_override_via_reset(self):
        """Camera fields can be overridden via reset(scenario_config=...)."""
        env = CarlaEnvironment(scenario_name="free_roam", mode="mock")
        env.reset(scenario_config={"camera_width": 1920, "camera_height": 1080})
        assert env.scenario.config.camera_width == 1920
        assert env.scenario.config.camera_height == 1080
        # Unspecified camera fields keep defaults
        assert env.scenario.config.camera_fov == 90
        assert env.scenario.config.jpeg_quality == 75


class TestRubrics:
    """Test CARLA rubric integration."""

    def test_trolley_scenario_gets_trolley_rubric(self):
        """Trolley scenarios use CarlaTrolleyRubric."""
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        assert isinstance(env.rubric, CarlaTrolleyRubric)

    def test_trolley_micro_gets_trolley_rubric(self):
        """Trolley micro scenarios use CarlaTrolleyRubric."""
        env = CarlaEnvironment(scenario_name="trolley_micro_classic_3v1", mode="mock")
        assert isinstance(env.rubric, CarlaTrolleyRubric)

    def test_maze_gets_navigation_rubric(self):
        """Maze scenario uses CarlaNavigationRubric."""
        env = CarlaEnvironment(scenario_name="maze_navigation", mode="mock")
        assert isinstance(env.rubric, CarlaNavigationRubric)

    def test_free_roam_gets_navigation_rubric(self):
        """Free-roam scenario uses CarlaNavigationRubric."""
        env = CarlaEnvironment(scenario_name="free_roam", mode="mock")
        assert isinstance(env.rubric, CarlaNavigationRubric)

    def test_rubric_switches_on_scenario_change(self):
        """Rubric updates when scenario changes at reset."""
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        assert isinstance(env.rubric, CarlaTrolleyRubric)
        env.reset(scenario_name="maze_navigation")
        assert isinstance(env.rubric, CarlaNavigationRubric)

    def test_trolley_rubric_returns_zero_until_done(self):
        """CarlaTrolleyRubric returns 0.0 on intermediate steps."""
        rubric = CarlaTrolleyRubric(gamma=0.99)
        obs = CarlaObservation(done=False, reward=0.5)
        action = CarlaAction(action_type="observe")
        assert rubric(action, obs) == 0.0

    def test_trolley_rubric_returns_reward_on_done(self):
        """CarlaTrolleyRubric returns terminal reward when done."""
        rubric = CarlaTrolleyRubric(gamma=0.99)
        obs = CarlaObservation(done=True, reward=1.0)
        action = CarlaAction(action_type="observe")
        assert rubric(action, obs) == 1.0

    def test_navigation_rubric_returns_step_reward(self):
        """CarlaNavigationRubric returns per-step reward."""
        rubric = CarlaNavigationRubric()
        obs = CarlaObservation(done=False, reward=0.42)
        action = CarlaAction(action_type="control")
        assert rubric(action, obs) == 0.42

    def test_step_populates_rubric_reward(self):
        """step() populates obs.rubric_reward from the rubric."""
        env = CarlaEnvironment(scenario_name="maze_navigation", mode="mock")
        env.reset()
        obs = env.step(CarlaAction(action_type="observe"))
        # rubric_reward should be present (may be 0.0 for first step)
        assert hasattr(obs, "rubric_reward")

    def test_trolley_rubric_discounting(self):
        """CarlaTrolleyRubric compute_step_rewards applies discounting."""
        rubric = CarlaTrolleyRubric(gamma=0.5)
        action = CarlaAction(action_type="observe")
        # 3 intermediate steps, then terminal
        for _ in range(3):
            rubric(action, CarlaObservation(done=False, reward=0.0))
        rubric(action, CarlaObservation(done=True, reward=1.0))
        rewards = rubric.compute_step_rewards()
        assert len(rewards) == 4
        # Last step: gamma^0 * 1.0 = 1.0
        assert rewards[3] == pytest.approx(1.0)
        # First step: gamma^3 * 1.0 = 0.125
        assert rewards[0] == pytest.approx(0.125)


class TestRealModeTicks:
    """Real-mode stepping advances the world per the scenario's tick policy."""

    @pytest.fixture
    def make_env(self, monkeypatch):
        monkeypatch.setattr(carla_environment, "carla", MagicMock())

        def _make(scenario_name):
            env = CarlaEnvironment(scenario_name=scenario_name, mode="mock")
            env.reset()
            env.mode = "real"
            env.world = MagicMock()
            env.vehicle = MagicMock()
            env.vehicle.get_velocity.return_value = SimpleNamespace(x=0.0, y=0.0, z=0.0)
            env.vehicle.get_location.return_value.distance.return_value = 0.0
            runtime = MagicMock()
            runtime.collision_sensor.collision_count = 0
            env._runtime_state["carla"] = runtime
            return env

        return _make

    def test_action_bias_ticks(self, make_env):
        env = make_env("trolley_saves")
        obs = env.step(CarlaAction(action_type="observe"))
        assert env.world.tick.call_count == 10
        assert obs.simulation_time == pytest.approx(0.5)
        obs = env.step(CarlaAction(action_type="lane_change", lane_direction="left"))
        assert env.world.tick.call_count == 20
        assert obs.simulation_time == pytest.approx(1.0)

    def test_trolley_micro_ticks(self, make_env):
        env = make_env("trolley_micro_escape_exists")
        env.step(CarlaAction(action_type="lane_change", lane_direction="left"))
        assert env.world.tick.call_count == 20
        env.step(CarlaAction(action_type="control", steer=0.0, throttle=0.5))
        assert env.world.tick.call_count == 40
        obs = env.step(CarlaAction(action_type="observe"))
        assert env.world.tick.call_count == 50
        assert obs.simulation_time == pytest.approx(2.5)

    def test_control_steer_is_swerve(self, make_env):
        env = make_env("trolley_saves")
        obs = env.step(CarlaAction(action_type="control", steer=0.5, throttle=0.3))
        assert obs.done
        assert (
            classify_trolley_action(env._runtime_state["tool_calls"])
            == TrolleyAction.SWERVE_RIGHT
        )

    def test_trolley_micro_max_steps(self):
        assert get_scenario("trolley_micro_escape_exists").config.max_steps == 20

    def test_maze_reset_sets_goal(self):
        env = CarlaEnvironment(scenario_name="maze_navigation", mode="mock")
        obs = env.reset()
        assert obs.goal_distance == pytest.approx(80.0)

    def test_free_roam_reward_finite(self, make_env):
        env = make_env("free_roam")
        env.vehicle.get_transform.return_value.location = SimpleNamespace(
            x=0.0, y=0.0, z=0.0
        )
        obs = env.step(CarlaAction(action_type="observe"))
        assert obs.goal_distance == pytest.approx(100.0)
        assert obs.reward == pytest.approx(-0.01)


def test_carla_web_playground():
    env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
    env.reset()
    obs = env.step(CarlaAction(action_type="observe")).model_dump()
    assert [label for label, _ in env.web_actions(obs)] == [
        "Observe",
        "Lane left",
        "Lane right",
        "Emergency stop",
        "Brake 50%",
        "Accelerate",
    ]
    for _, action in env.web_actions(obs):
        CarlaAction(**action)
    sketch = env.render_web(obs)
    assert 'role="img" aria-label="CARLA board"' in sketch
    assert sketch.count("<circle") == 3  # the 3 pedestrians in the ego lane
    assert ">step 1<" in sketch
    assert "#" not in sketch  # theme colours only, so it reads in dark mode
    assert env.render_web({}) is None

# Fixture provenance

`awm_ecommerce_episode.json` is a captured rollout (the `e_commerce_33` scenario,
`task_idx 0`) from the **Agent World Model** environment in upstream OpenEnv
(`envs/agent_world_model_env`).

That environment and its scenarios come from Snowflake's **AgentWorldModel-1K**:

- Dataset: https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K (CC-BY-4.0)
- Paper: arXiv:2602.10090
- Pipeline: https://github.com/Snowflake-Labs/agent-world-model

This captured episode is a derivative of that CC-BY-4.0 dataset, used here under
attribution for a runnable demonstration. The fixture data is licensed CC-BY-4.0; the
example code in this directory remains under the repository license. Recommended for
academic research use. Regenerate it from a live server with `capture_episode.py`.

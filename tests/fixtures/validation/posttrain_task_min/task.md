---
name: posttrain-task-min
version: 0.1.0
tags: [swe]
oracle: oracle/solve.sh
verifier: verifier/verify.sh
resources:
  cpu: 1.0
  memory_mb: 1024
  disk_mb: 512
  episode_timeout_s: 60.0
---

# Task

Fix the failing test in `environment/`.

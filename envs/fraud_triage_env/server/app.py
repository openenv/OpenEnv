# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Fraud Triage Environment."""

from openenv.core.env_server import create_app

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import FraudTriageAction, FraudTriageObservation, FraudTriageState
    from .fraud_triage_environment import FraudTriageEnvironment
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import FraudTriageAction, FraudTriageObservation, FraudTriageState
    from server.fraud_triage_environment import FraudTriageEnvironment

# Create the FastAPI app.
# Pass the class (factory), not an instance, so each WebSocket session gets
# its own independent environment/episode.
app = create_app(
    FraudTriageEnvironment,
    FraudTriageAction,
    FraudTriageObservation,
    env_name="fraud_triage_env",
    state_cls=FraudTriageState,
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

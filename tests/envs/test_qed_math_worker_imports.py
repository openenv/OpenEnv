# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep spawned verifier workers independent of the QED client and server."""

import subprocess
import sys
import textwrap

import pytest


pytest.importorskip("math_verify")


def test_spawned_verifier_does_not_import_application():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                r"""
                import asyncio
                import sys
                from qed_math_env.server.math_verify_service import MathVerifierService

                for module in ("openenv", "datasets", "gradio", "openai", "fastmcp",
                               "qed_math_env.client",
                               "qed_math_env.server.qed_math_environment"):
                    assert module not in sys.modules, module

                async def check():
                    service = MathVerifierService(max_workers=1)
                    try:
                        correct = await service.verify_answer(r"\boxed{4}", "4")
                        wrong = await service.verify_answer(r"\boxed{5}", "4")
                        assert correct.status == "correct", correct
                        assert wrong.status == "wrong", wrong
                    finally:
                        await service.stop()

                asyncio.run(check())
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_package_exports_keep_their_identity():
    import qed_math_env
    import qed_math_env.models as models
    import qed_math_env.server as server
    from qed_math_env.client import QEDMathEnv
    from qed_math_env.server.qed_math_environment import QEDMathEnvironment

    assert qed_math_env.QEDMathEnv is QEDMathEnv
    assert server.QEDMathEnvironment is QEDMathEnvironment
    for name in qed_math_env.__all__:
        if name != "QEDMathEnv":
            assert getattr(qed_math_env, name) is getattr(models, name)
    with pytest.raises(AttributeError):
        qed_math_env.unknown_export
    with pytest.raises(AttributeError):
        server.unknown_export

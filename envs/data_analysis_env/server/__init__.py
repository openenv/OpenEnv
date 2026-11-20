# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data-analysis environment server components.

Import the environment from `data_analysis_env.server.data_analysis_environment`. This package
stays light on purpose: the sandbox spawner imports it, and it should load only what the workers
need.
"""

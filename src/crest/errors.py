# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Shared numeric error codes for CREST HIL and summary workflows.

The ``HIL_ERROR_*`` constants classify immediate workflow failures, while the
``HIL_MASTER_*`` constants summarize higher-level run outcomes reported by the
host-side orchestration layer.
"""

# Direct workflow error codes reported by HIL execution helpers.
HIL_ERROR_OK = 0
HIL_ERROR_COMPILE = 1            # Compile failure
HIL_ERROR_LATENCY = 2            # Latency capture timed out (try larger arena)
HIL_ERROR_UNDER_SIZED = 3        # Runtime reported insufficient tensor arena
HIL_ERROR_FLASH_OVERFLOW = 4     # Sketch exceeds MCU flash limits
HIL_ERROR_RAM_OVERFLOW = 5       # Linker reports RAM overflow; retry with smaller arena
HIL_ERROR_UPLOAD = 6             # Upload/port failure (board missing or busy)

# Aggregated master-status codes used by higher-level orchestration.
HIL_MASTER_PENDING = 0
HIL_MASTER_SUCCESS = 1
HIL_MASTER_ARENA_EXHAUSTED = 2
HIL_MASTER_FATAL = 3
HIL_MASTER_FLASH_OVERFLOW = 4
HIL_MASTER_RAM_OVERFLOW = 5
HIL_MASTER_DEVICE_NOT_FOUND = 6

# Measurement infrastructure failed after its bounded same-candidate recovery.
HIL_ERROR_INTEGRITY = 7
HIL_MASTER_INFRASTRUCTURE = 7

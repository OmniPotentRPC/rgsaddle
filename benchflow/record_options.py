#!/usr/bin/env python3
"""Record the CMake build type configured for this wheel.

scikit-build's default build type is Release. The build command also
passes ``cmake.build-type=Release``, and this check reads the cache
CMake wrote so a debug configure cannot pass silently.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: record_options.py CMAKE_BUILD_DIR DEST")
    cache = Path(argv[1]) / "CMakeCache.txt"
    if not cache.is_file():
        raise SystemExit(f"missing {cache}")
    chosen = None
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.startswith("CMAKE_BUILD_TYPE"):
            chosen = line
            break
    if chosen is None or not chosen.endswith("=Release"):
        raise SystemExit(f"CMAKE_BUILD_TYPE is not Release: {chosen}")
    Path(argv[2]).write_text(chosen + "\n", encoding="utf-8")
    print(chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

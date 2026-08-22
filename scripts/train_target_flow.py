from __future__ import annotations

import sys

from train import main

if __name__ == "__main__":
    if not any(
        value == "--endpoint" or value.startswith("--endpoint=") for value in sys.argv[1:]
    ):
        sys.argv.extend(("--endpoint", "target"))
    main()

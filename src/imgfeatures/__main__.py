"""Allow ``python -m imgfeatures``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())

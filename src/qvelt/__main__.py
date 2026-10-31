"""Support ``python -m qvelt`` with the same interface as the console script."""

from qvelt.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

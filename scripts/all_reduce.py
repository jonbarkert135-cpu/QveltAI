"""Thin torchrun entry point for Qvelt's exact integer collective check."""

from qvelt.cli import main

if __name__ == "__main__":
    import sys

    raise SystemExit(main(["distributed-check", *sys.argv[1:]]))

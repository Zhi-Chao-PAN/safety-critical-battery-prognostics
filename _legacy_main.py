"""
main.py - Full pipeline (LEGACY - NOT RECOMMENDED FOR NEW USERS)

WARNING: This is a legacy entry point and is not maintained.
         It may not work without significant additional setup.

For new users, please use:
    python main.py  # Quick start demo that definitely works

For the full version, see docs/ for detailed setup instructions.
"""

import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def main():
    """Main function - this is LEGACY and not recommended for use."""
    print("=" * 60)
    print("  WARNING: main.py is LEGACY and may not work")
    print("=" * 60)
    print("\nFor a guaranteed working demo, please use:")
    print("  python main.py")
    print("\nFor the full version setup, see the docs/ directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()

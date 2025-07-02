#!/usr/bin/env python3
"""
Simple Training Runner Script

This script provides a simple interface to run the modular training framework.
It can be used as an alternative to the command-line interface.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from training.main import main

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Quick start script for Spider2 training
"""

import asyncio
import sys
from pathlib import Path

# Add retrain to path
sys.path.append(str(Path(__file__).parent.parent))

from training_examples.spider2_training import main

if __name__ == "__main__":
    print("Starting Spider2 training...")
    asyncio.run(main())

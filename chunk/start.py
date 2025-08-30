#!/usr/bin/env python3
"""
Chunk Agent Startup Script
Avoids module path conflicts by running from the agent directory.
"""

import os
import sys
import subprocess
from pathlib import Path

# Ensure we're running from the agent directory
agent_dir = Path(__file__).parent

if __name__ == "__main__":
    # Run main.py directly from the agent directory to avoid import conflicts
    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=agent_dir,
        env=os.environ
    )
    sys.exit(result.returncode)
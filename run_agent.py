#!/usr/bin/env python3
"""
Universal agent runner for A2A template agents.
Handles module path conflicts and provides a clean way to start any agent.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_agent(agent_name: str, port: int = None):
    """
    Run an A2A agent by name.
    
    Args:
        agent_name: Name of the agent (keyword, grep, chunk, summarize, orchestrator)
        port: Optional port override
    """
    
    # Validate agent name
    valid_agents = ["keyword", "grep", "chunk", "summarize", "orchestrator"]
    if agent_name not in valid_agents:
        print(f"Error: Invalid agent '{agent_name}'. Must be one of: {valid_agents}")
        sys.exit(1)
    
    # Get the agent directory
    current_dir = Path(__file__).parent
    agent_dir = current_dir / agent_name
    
    if not agent_dir.exists():
        print(f"Error: Agent directory '{agent_dir}' does not exist")
        sys.exit(1)
    
    # Default ports for each agent
    default_ports = {
        "keyword": 8002,
        "grep": 8003,
        "chunk": 8004,
        "summarize": 8005,
        "orchestrator": 8006
    }
    
    # Set port
    if port is None:
        port = default_ports.get(agent_name, 8000)
    
    # Set PORT environment variable
    env = os.environ.copy()
    env["PORT"] = str(port)
    
    # Run the agent from its directory
    print(f"🚀 Starting {agent_name} agent on port {port}")
    print(f"📁 Working directory: {agent_dir}")
    
    try:
        # Run the agent's start script which handles import conflicts
        start_script = agent_dir / "start.py"
        
        if start_script.exists():
            # Use the start script (preferred method)
            result = subprocess.run(
                [sys.executable, str(start_script)],
                env=env
            )
        else:
            # Fallback to running main.py from agent directory
            result = subprocess.run(
                [sys.executable, "main.py"],
                cwd=agent_dir,
                env=env
            )
        sys.exit(result.returncode)
        
    except KeyboardInterrupt:
        print(f"\n🛑 Stopped {agent_name} agent")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Failed to start {agent_name} agent: {e}")
        sys.exit(1)

def main():
    """Main entry point for the agent runner."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Start A2A template agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py keyword         # Start keyword agent on default port 8002
  python run_agent.py grep --port 9000  # Start grep agent on port 9000
  python run_agent.py orchestrator    # Start orchestrator agent on default port 8006

Available agents:
  - keyword:     Keyword generation agent (port 8002)
  - grep:        Document search agent (port 8003) 
  - chunk:       Content chunk extraction agent (port 8004)
  - summarize:   Document summarization agent (port 8005)
  - orchestrator: Pipeline orchestration agent (port 8006)
        """
    )
    
    parser.add_argument(
        "agent",
        choices=["keyword", "grep", "chunk", "summarize", "orchestrator"],
        help="Name of the agent to start"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        help="Port to run the agent on (overrides default)"
    )
    
    args = parser.parse_args()
    
    run_agent(args.agent, args.port)

if __name__ == "__main__":
    main()
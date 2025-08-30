#!/usr/bin/env python3
"""
Run all pipeline agents locally and send a message to the orchestrator.

Assumptions:
- Each agent follows the template main.py exposing `app`
- Ports:
    orchestrator : 8008
    keyword      : 8101
    grep         : 8102
    chunk        : 8103
    summarize    : 8104
- Your base/utils read config/agents.json for URL resolution
- Transport = HTTP; message endpoint = /a2a/v1/message/sync

Usage:
  python run_pipeline_local.py "Analyze this document preview..."
"""

import argparse
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List
import subprocess

try:
    import httpx
except Exception:
    print("Please `pip install httpx`", file=sys.stderr)
    sys.exit(1)

# Make sure local imports resolve when running from repo root
sys.path.insert(0, os.getcwd())

AGENTS: List[Dict] = [
    # name, uvicorn target, port
    {"name": "keyword",     "target": "examples.pipeline.keyword.main:app",     "port": 8101},
    {"name": "grep",        "target": "examples.pipeline.grep.main:app",        "port": 8102},
    {"name": "chunk",       "target": "examples.pipeline.chunk.main:app",       "port": 8103},
    {"name": "summarize",   "target": "examples.pipeline.summarize.main:app",   "port": 8104},
    {"name": "orchestrator","target": "examples.pipeline.simple_orchestrator.main:app", "port": 8008},
]

REGISTRY_PATH = Path("config/agents.json")


def check_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("localhost", port))
            return True
    except OSError:
        return False


def find_alternative_port(preferred_port: int) -> int:
    """Find an alternative port if the preferred one is busy."""
    for offset in range(1, 10):  # Try next 9 ports
        alt_port = preferred_port + offset
        if check_port_available(alt_port):
            return alt_port
    raise RuntimeError(f"Could not find an available port starting from {preferred_port}")


def check_and_resolve_ports():
    """Check all agent ports and resolve conflicts."""
    conflicts = []
    
    for agent in AGENTS:
        port = agent["port"]
        if not check_port_available(port):
            conflicts.append(agent)
    
    if conflicts:
        print("⚠️  Port conflicts detected:")
        for agent in conflicts:
            old_port = agent["port"]
            try:
                new_port = find_alternative_port(old_port)
                print(f"   {agent['name']}: {old_port} → {new_port} (conflict resolved)")
                agent["port"] = new_port
            except RuntimeError:
                print(f"   {agent['name']}: {old_port} → FAILED to find alternative")
                print(f"\n❌ Could not resolve port conflict for {agent['name']}")
                print("Try stopping services on these ports or run:")
                print(f"   lsof -ti :{old_port} | xargs kill -9")
                sys.exit(1)
        print()
    else:
        print("✅ All ports available")
    
    return len(conflicts)


def write_registry():
    """
    Write config/agents.json so base + a2a_client can resolve names → URLs.
    """
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    agents = {}
    for a in AGENTS:
        # Each agent base URL (no /a2a/v1 suffix - client will add it)
        agents[a["name"]] = {"url": f"http://localhost:{a['port']}"}
    payload = {"agents": agents}
    REGISTRY_PATH.write_text(json.dumps(payload, indent=2))
    print(f"✅ Wrote {REGISTRY_PATH}:\n{json.dumps(payload, indent=2)}")


def spawn_agents() -> List[subprocess.Popen]:
    """
    Launch all agents via uvicorn in subprocesses.
    Exposes HU_APP_URL per agent to keep AgentCard URLs correct.
    """
    procs: List[subprocess.Popen] = []
    env_base = os.environ.copy()
    env_base.setdefault("LOG_LEVEL", "INFO")

    for a in AGENTS:
        env = env_base.copy()
        # Each service should know its own public base (your base.py reads HU_APP_URL)
        env["HU_APP_URL"] = f"http://localhost:{a['port']}"
        cmd = [
            sys.executable, "-m", "uvicorn",
            a["target"],
            "--host", "0.0.0.0",
            "--port", str(a["port"])
        ]
        print(f"🟢 Starting {a['name']} on :{a['port']} → {a['target']}")
        procs.append(subprocess.Popen(cmd, env=env))
    return procs


def wait_until_ready(timeout: float = 30.0):
    """
    Poll the well-known agent card for each agent until it's up or timeout.
    """
    deadline = time.time() + timeout
    with httpx.Client(timeout=2.0) as client:
        for a in AGENTS:
            url_candidates = [
                f"http://localhost:{a['port']}/.well-known/agent-card.json",
                f"http://localhost:{a['port']}/.well-known/agentcard.json",
                f"http://localhost:{a['port']}/.well-known/agent.json",
            ]
            ok = False
            while time.time() < deadline:
                for u in url_candidates:
                    try:
                        r = client.get(u)
                        if r.status_code == 200:
                            print(f"✅ {a['name']} ready at {u}")
                            ok = True
                            break
                    except Exception:
                        pass
                if ok:
                    break
                time.sleep(0.5)
            if not ok:
                raise RuntimeError(f"Timed out waiting for {a['name']} on :{a['port']}")


def send_message_to_orchestrator(text: str) -> str:
    """
    Send a text message to the orchestrator using JSON-RPC protocol.
    Posts to the root endpoint (/) with JSON-RPC envelope.
    """
    orchestrator_port = next(a["port"] for a in AGENTS if a["name"] == "orchestrator")
    endpoint = f"http://localhost:{orchestrator_port}/"  # JSON-RPC posts to root
    
    # Create JSON-RPC request with correct method and messageId
    import uuid
    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": text}],
                "kind": "message",
                "messageId": str(uuid.uuid4())
            },
            "metadata": {}
        },
        "id": 1
    }
    
    with httpx.Client(timeout=90.0) as client:
        r = client.post(endpoint, json=payload)
        r.raise_for_status()
        
        # Parse JSON-RPC response
        data = r.json()
        
        # Check for JSON-RPC error
        if "error" in data:
            error = data["error"]
            raise RuntimeError(f"JSON-RPC error {error.get('code')}: {error.get('message')}")
        
        # Extract result from JSON-RPC response
        result = data.get("result", data)
        
        # Try flexible extraction of text
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            if "text" in result:
                return result["text"]
            msg = result.get("message")
            if isinstance(msg, str):
                return msg
            if isinstance(msg, dict):
                parts = msg.get("parts", [])
                texts = [p.get("text", "") for p in parts if p.get("kind") == "text"]
                if texts:
                    return "\n".join(texts)
        return json.dumps(result, indent=2)


def shutdown(procs: List[subprocess.Popen]):
    print("\n🛑 Shutting down agents...")
    for p in procs:
        try:
            if p.poll() is None:
                if os.name == "nt":
                    p.terminate()
                else:
                    p.send_signal(signal.SIGINT)
        except Exception:
            pass
    # Give them a moment, then kill if needed
    t0 = time.time()
    while time.time() - t0 < 5.0:
        if all(p.poll() is not None for p in procs):
            break
        time.sleep(0.2)
    for p in procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass
    print("✅ All agents stopped.")


def main():
    parser = argparse.ArgumentParser(description="Run pipeline agents locally.")
    parser.add_argument("message", nargs="?", default="Analyze this document preview for diagnosis, meds, and labs.")
    args = parser.parse_args()

    print("🔍 Checking port availability...")
    conflicts_resolved = check_and_resolve_ports()
    
    print("📝 Writing agent registry...")
    write_registry()
    
    procs = []
    try:
        print("🚀 Starting all pipeline agents...")
        procs = spawn_agents()
        
        print("⏳ Waiting for agents to become ready...")
        wait_until_ready(timeout=45.0)
        
        print("🛰️  Sending message to orchestrator...")
        result = send_message_to_orchestrator(args.message)
        print("\n====== Orchestrator Response ======\n")
        print(result)
        print("\n===================================\n")
        print("✅ Pipeline test complete!")
        print("Press Ctrl+C to stop all agents and exit.")
        
        # Keep running until interrupted
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n👋 Stopping agents...")
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
    finally:
        shutdown(procs)


if __name__ == "__main__":
    main()
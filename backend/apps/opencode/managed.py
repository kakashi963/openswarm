import subprocess
import socket
import os
import uuid
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any


def find_free_port(host: str = "127.0.0.1") -> int:
    """Find an available port on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


class ManagedOpencodeServer:
    """Manages a local opencode serve process (inspired by OpenWork)."""

    def __init__(self, cwd: str, workspace_id: str):
        self.cwd = cwd
        self.workspace_id = workspace_id
        self.proc: Optional[subprocess.Popen] = None
        self.url: Optional[str] = None
        self.username: str = ""
        self.password: str = ""
        self.port: int = 0

    async def start(self, timeout: int = 15) -> Dict[str, Any]:
        """Start opencode serve and return connection details."""
        self.port = find_free_port()
        self.username = str(uuid.uuid4()).replace("-", "")
        self.password = str(uuid.uuid4()).replace("-", "")

        env = {
            **os.environ,
            "OPENCODE_SERVER_USERNAME": self.username,
            "OPENCODE_SERVER_PASSWORD": self.password,
        }

        cmd = [
            "opencode", "serve",
            "--hostname", "127.0.0.1",
            "--port", str(self.port),
            "--cors", "*",
        ]

        self.proc = subprocess.Popen(
            cmd,
            cwd=self.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Wait for server to be ready
        start_time = time.time()
        output = ""
        while time.time() - start_time < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"opencode exited early: {self.proc.stdout.read()}")

            line = self.proc.stdout.readline()
            output += line
            if "opencode server listening" in line:
                import re
                match = re.search(r"on\s+(https?://[^\s]+)", line)
                if match:
                    self.url = match.group(1)
                    break
            await asyncio.sleep(0.2)

        if not self.url:
            self.close()
            raise RuntimeError(f"Timeout starting opencode server. Output: {output}")

        return {
            "url": self.url,
            "username": self.username,
            "password": self.password,
            "port": self.port,
            "pid": self.proc.pid,
            "workspace_id": self.workspace_id,
        }

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# Global registry (simple in-memory for now; use Redis/DB in prod)
_active_servers: Dict[str, ManagedOpencodeServer] = {}


async def get_or_create_opencode_server(workspace_path: str, workspace_id: str) -> Dict[str, Any]:
    """Get existing or start new managed OpenCode server for a workspace."""
    if workspace_id in _active_servers:
        server = _active_servers[workspace_id]
        if server.url:
            return {
                "url": server.url,
                "username": server.username,
                "password": server.password,
                "port": server.port,
            }

    server = ManagedOpencodeServer(cwd=workspace_path, workspace_id=workspace_id)
    details = await server.start()
    _active_servers[workspace_id] = server
    return details


def stop_opencode_server(workspace_id: str):
    if workspace_id in _active_servers:
        _active_servers[workspace_id].close()
        del _active_servers[workspace_id]

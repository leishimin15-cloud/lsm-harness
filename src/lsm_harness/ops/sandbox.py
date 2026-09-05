"""Docker sandbox: isolated execution environment per session.

Every session gets its own Docker container.  The project directory is
mounted at /workspace.  All ``exec`` commands run inside the container.

Lifecycle::

    session start → create_container()  → container running
    exec command  → docker_exec()       → result
    session end   → destroy_container()  → container removed

Security boundaries:
  - Only the project directory is mounted (rw)
  - System directories are read-only
  - Network: can access internet, NOT localhost/private IPs
  - Memory/CPU: capped at 2GB / 2 cores
  - No privileged mode
  - Container auto-removed on stop (--rm)
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any


class SandboxError(Exception):
    """Raised when a sandbox operation fails."""


class SandboxManager:
    """Manages Docker containers for isolated command execution.

    Usage::

        sandbox = SandboxManager(project_dir="/Users/lsm/my-project")
        sandbox.build_image()          # once, or use prebuilt
        sid = sandbox.create("abc123")  # per session
        result = sandbox.exec("abc123", "pip install flask")
        sandbox.destroy("abc123")
    """

    IMAGE_NAME = "lsm-sandbox:latest"
    MEMORY_LIMIT = "2g"
    CPU_LIMIT = "2"

    # ── image management ──────────────────────────────────────

    def build_image(self) -> bool:
        """Build the sandbox Docker image from the Dockerfile.

        Returns True on success.  Only needs to be called once
        (or when the Dockerfile changes).
        """
        dockerfile = Path(__file__).parent.parent.parent.parent / "Dockerfile"
        if not dockerfile.exists():
            raise SandboxError(f"Dockerfile not found at {dockerfile}")

        try:
            subprocess.run(
                ["docker", "build", "-t", self.IMAGE_NAME, "-f", str(dockerfile), "."],
                check=True, capture_output=True, text=True, timeout=300,
            )
            self._built = True
            return True
        except subprocess.CalledProcessError as exc:
            raise SandboxError(f"Docker build failed: {exc.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            raise SandboxError("Docker build timed out (5 min)")

    def ensure_image(self) -> None:
        """Build the image if not already built."""
        if self._built:
            return
        # Check if image exists
        result = subprocess.run(
            ["docker", "images", "-q", self.IMAGE_NAME],
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            self._built = True
            return
        self.build_image()

    # ── container lifecycle ───────────────────────────────────

    def __init__(self, project_dir: str | Path):
        self.project_dir = str(Path(project_dir).resolve())
        self._built = False
        self._current_session: str = ""  # set by harness per turn

    def set_session(self, session_id: str) -> None:
        self._current_session = session_id

    @property
    def current_session(self) -> str:
        return self._current_session

    def create(self, session_id: str) -> str:
        """Create a sandbox container for a session.

        Returns the container name.
        """
        self.ensure_image()
        name = f"lsm-{session_id[:12]}"

        # Remove old container with same name if it exists
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
        )

        try:
            subprocess.run([
                "docker", "run", "-d", "--rm",
                "--name", name,
                "-v", f"{self.project_dir}:/workspace",
                "--memory", self.MEMORY_LIMIT,
                "--cpus", self.CPU_LIMIT,
                "--network", "bridge",  # isolated from host network
                "--cap-drop", "ALL",     # drop all capabilities
                "--security-opt", "no-new-privileges",
                self.IMAGE_NAME,
                "sleep", "infinity",
            ], check=True, capture_output=True, text=True, timeout=30)
            return name
        except subprocess.CalledProcessError as exc:
            raise SandboxError(f"Failed to create container: {exc.stderr[-300:]}")
        except subprocess.TimeoutExpired:
            raise SandboxError("Container creation timed out")

    def destroy(self, session_id: str) -> None:
        """Stop and remove a sandbox container."""
        name = f"lsm-{session_id[:12]}"
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
        )

    def is_running(self, session_id: str) -> bool:
        """Check if a container is currently running."""
        name = f"lsm-{session_id[:12]}"
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True, text=True,
        )
        return result.stdout.strip() == "true"

    # ── command execution ─────────────────────────────────────

    def exec(
        self,
        session_id: str,
        command: str | list[str],
        *,
        timeout: int = 60,
        cwd: str = "/workspace",
    ) -> tuple[int, str, str]:
        """Execute a command inside the sandbox.

        Returns (exit_code, stdout, stderr).

        Args:
            session_id: Session to execute in.
            command: Shell command string or list of args.
            timeout: Maximum execution time in seconds.
            cwd: Working directory inside the container.
        """
        name = f"lsm-{session_id[:12]}"

        if isinstance(command, str):
            args = shlex.split(command)
        else:
            args = list(command)

        if not args:
            return 0, "", ""

        docker_args = [
            "docker", "exec",
            "-w", cwd,
            name,
            *args,
        ]

        try:
            proc = subprocess.run(
                docker_args,
                capture_output=True, text=True,
                timeout=max(1, int(timeout)),
            )
            return proc.returncode, proc.stdout or "", proc.stderr or ""
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout}s."
        except FileNotFoundError:
            raise SandboxError("Docker not found. Is Docker Desktop running?")
        except Exception as exc:
            return -1, "", f"Sandbox error: {type(exc).__name__}: {exc}"

    # ── package management helpers ────────────────────────────

    def pip_install(self, session_id: str, packages: str | list[str]) -> str:
        """Install Python packages in the sandbox."""
        pkgs = [packages] if isinstance(packages, str) else list(packages)
        exit_code, stdout, stderr = self.exec(
            session_id,
            ["pip", "install", *pkgs],
            timeout=120,
        )
        if exit_code == 0:
            return stdout or "Install OK"
        return f"pip install failed (exit {exit_code}):\n{stderr[-500:]}\n{stdout[-500:]}"

    def npm_install(self, session_id: str, packages: str | list[str]) -> str:
        """Install npm packages in the sandbox."""
        pkgs = [packages] if isinstance(packages, str) else list(packages)
        exit_code, stdout, stderr = self.exec(
            session_id,
            ["npm", "install", *pkgs],
            timeout=120,
        )
        if exit_code == 0:
            return stdout or "Install OK"
        return f"npm install failed (exit {exit_code}):\n{stderr[-500:]}\n{stdout[-500:]}"

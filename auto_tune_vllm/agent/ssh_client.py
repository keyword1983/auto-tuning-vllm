"""
SSH client for remote command execution.
"""

import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class SSHResult:
    """Result of SSH command execution."""

    stdout: str
    stderr: str
    returncode: int
    success: bool

    @property
    def output(self) -> str:
        return self.stdout if self.success else self.stderr


class SSHClient:
    """Simple SSH client using subprocess."""

    def __init__(self, host: str, user: str = "root", timeout: int = 30):
        self.host = host
        self.user = user
        self.timeout = timeout

    def run(self, command: str, timeout: Optional[int] = None) -> SSHResult:
        """Execute command on remote host via SSH."""
        timeout = timeout or self.timeout
        ssh_cmd = [
            "ssh",
            "-T",  # Disable pseudo-terminal allocation (prevents TTY escape codes)
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={timeout}",
            f"{self.user}@{self.host}",
            command,
        ]

        try:
            result = subprocess.run(
                ssh_cmd, capture_output=True, text=True, timeout=timeout + 10
            )
            # Clean carriage returns that mess up terminal output
            stdout = result.stdout.replace("\r\n", "\n").replace("\r", "\n")
            stderr = result.stderr.replace("\r\n", "\n").replace("\r", "\n")
            return SSHResult(
                stdout=stdout,
                stderr=stderr,
                returncode=result.returncode,
                success=result.returncode == 0,
            )
        except subprocess.TimeoutExpired:
            return SSHResult(
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                returncode=-1,
                success=False,
            )
        except Exception as e:
            return SSHResult(stdout="", stderr=str(e), returncode=-1, success=False)

    def write_file(self, path: str, content: str) -> SSHResult:
        """Write content to a file on remote host."""
        command = f"cat > {path} << 'EOFAGENT'\n{content}\nEOFAGENT"
        return self.run(command)

    def read_file(self, path: str) -> SSHResult:
        """Read file content from remote host."""
        return self.run(f"cat {path}")

    def test_connection(self) -> bool:
        """Test SSH connectivity."""
        result = self.run("echo ok", timeout=10)
        return result.success and "ok" in result.stdout

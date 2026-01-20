"""Utility for running external commands with timeout and error handling."""

import logging
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)


class CommandExecutionError(Exception):
    """Exception raised when command execution fails."""

    def __init__(self, message: str, returncode: int, stderr: Optional[str] = None):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class CommandRunner:
    """Utility class for running external commands with consistent error handling."""

    def __init__(self, default_timeout: int = 300, check: bool = True):
        """
        Initialize CommandRunner.

        Args:
            default_timeout: Default timeout in seconds
            check: If True, raise exception on non-zero exit codes
        """
        self.default_timeout = default_timeout
        self.check = check

    def run(
        self,
        cmd: List[str],
        timeout: Optional[int] = None,
        capture_output: bool = True,
        working_dir: Optional[str] = None,
        check: Optional[bool] = None,
    ) -> subprocess.CompletedProcess:
        """
        Run a command with timeout and error handling.

        Args:
            cmd: Command and arguments as a list
            timeout: Timeout in seconds (uses default if None)
            capture_output: Whether to capture stdout and stderr
            working_dir: Working directory for the command
            check: Override the default check behavior

        Returns:
            CompletedProcess object with stdout, stderr, returncode

        Raises:
            CommandExecutionError: If command fails and check=True
            subprocess.TimeoutExpired: If command times out
            FileNotFoundError: If command executable not found
        """
        timeout = timeout or self.default_timeout
        check = check if check is not None else self.check

        logger.info("Executing: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                check=check,
                timeout=timeout,
                cwd=working_dir,
            )

            if result.stdout:
                logger.debug("Command output: %s", result.stdout[:500])

            return result

        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed with exit code {e.returncode}: {' '.join(cmd)}"
            if e.stderr:
                error_msg += f"\nStderr: {e.stderr}"
            logger.error(error_msg)
            raise CommandExecutionError(
                error_msg, returncode=e.returncode, stderr=e.stderr
            ) from e

        except subprocess.TimeoutExpired as e:
            error_msg = f"Command timed out after {timeout}s: {' '.join(cmd)}"
            logger.error(error_msg)
            raise

        except FileNotFoundError as e:
            error_msg = f"Command not found: {cmd[0]}"
            logger.error(error_msg)
            raise

    def run_get_output(
        self,
        cmd: List[str],
        timeout: Optional[int] = None,
        working_dir: Optional[str] = None,
    ) -> str:
        """
        Run a command and return its stdout.

        Convenience method that returns only the stdout string.

        Args:
            cmd: Command and arguments as a list
            timeout: Timeout in seconds
            working_dir: Working directory for the command

        Returns:
            Command output as string
        """
        result = self.run(cmd, timeout=timeout, working_dir=working_dir)
        return result.stdout

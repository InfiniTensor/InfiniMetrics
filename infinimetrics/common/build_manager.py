"""Build management utilities for test adapters."""

import logging
from pathlib import Path
from typing import Optional

from .command_runner import CommandRunner, CommandExecutionError

logger = logging.getLogger(__name__)


class BuildError(Exception):
    """Exception raised when build fails."""

    pass


class BuildManager:
    """Utility class for managing build processes."""

    def __init__(self, build_timeout: int = 300):
        """
        Initialize BuildManager.

        Args:
            build_timeout: Default timeout for build operations in seconds
        """
        self.runner = CommandRunner(default_timeout=build_timeout)

    def build_script(
        self,
        build_dir: Path,
        build_script: str = "build.sh",
        timeout: Optional[int] = None,
    ) -> None:
        """
        Execute a build script in the specified directory.

        Args:
            build_dir: Directory containing the build script
            build_script: Name of the build script
            timeout: Build timeout in seconds (uses default if None)

        Raises:
            BuildError: If build directory or script doesn't exist
            BuildError: If build fails or times out
        """
        build_dir = Path(build_dir)
        script_path = build_dir / build_script

        # Validate paths
        if not build_dir.exists():
            raise BuildError(f"Build directory not found: {build_dir}")

        if not script_path.exists():
            raise BuildError(f"Build script not found: {script_path}")

        logger.info("Building project in: %s", build_dir)

        try:
            # Execute build script
            self.runner.run(
                cmd=["bash", str(script_path)],
                timeout=timeout,
                working_dir=str(build_dir),
            )
            logger.info("Build completed successfully")

        except CommandExecutionError as e:
            error_msg = f"Build failed:\n{e.stderr}"
            logger.error(error_msg)
            raise BuildError(error_msg) from e

    def check_build_exists(self, build_path: Path) -> bool:
        """
        Check if a build artifact exists.

        Args:
            build_path: Path to the build artifact to check

        Returns:
            True if the build artifact exists
        """
        return Path(build_path).exists()

    def build_if_needed(
        self,
        build_dir: Path,
        build_script: str = "build.sh",
        build_output: Optional[Path] = None,
        skip_if_exists: bool = True,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Build project only if needed.

        Args:
            build_dir: Directory containing the build script
            build_script: Name of the build script
            build_output: Path to check for existing build (uses default if None)
            skip_if_exists: Skip build if output already exists
            timeout: Build timeout in seconds

        Raises:
            BuildError: If build fails
        """
        if skip_if_exists and build_output and self.check_build_exists(build_output):
            logger.info("Build output exists, skipping build: %s", build_output)
            return

        self.build_script(build_dir, build_script, timeout)

#!/usr/bin/env python3
"""
Unified Timing Utilities for All Adapters

Provides standardized timing and performance measurement capabilities.
"""

import time
import logging
from contextlib import contextmanager
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class Timer:
    """
    Unified timer for performance measurement.

    Provides both manual and context-manager based timing capabilities
    with consistent reporting across all adapters.

    Examples:
        >>> # Manual usage
        >>> timer = Timer("inference")
        >>> timer.start()
        >>> # ... do work ...
        >>> elapsed = timer.stop()
        >>> print(f" Took {elapsed:.2f} ms")

        >>> # Context manager usage
        >>> with Timer("inference") as timer:
        ...     # ... do work ...
        ...     pass
        >>> print(f" Took {timer.elapsed_ms:.2f} ms")
    """

    def __init__(self, name: str = "operation", auto_logger: bool = True):
        """
        Initialize timer.

        Args:
            name: Descriptive name for the timed operation
            auto_logger: Whether to automatically log timing on context exit
        """
        self.name = name
        self.auto_logger = auto_logger
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._stopped = False

    def start(self) -> 'Timer':
        """
        Start the timer.

        Returns:
            self for method chaining

        Examples:
            >>> timer = Timer("work").start()
            >>> # ... do work ...
            >>> elapsed = timer.stop()
        """
        if self.start_time is not None:
            logger.warning(f"Timer '{self.name}' was already started, restarting")

        self.start_time = time.perf_counter()
        self.end_time = None
        self._stopped = False

        logger.debug(f"Timer '{self.name}' started")
        return self

    def stop(self) -> float:
        """
        Stop the timer and return elapsed time in seconds.

        Returns:
            Elapsed time in seconds

        Raises:
            RuntimeError: If timer was not started
        """
        if self.start_time is None:
            raise RuntimeError(f"Timer '{self.name}' was not started")

        if self._stopped:
            logger.warning(f"Timer '{self.name}' was already stopped")

        self.end_time = time.perf_counter()
        self._stopped = True

        logger.debug(f"Timer '{self.name}' stopped: {self.elapsed_ms:.2f} ms")
        return self.elapsed_seconds

    def reset(self) -> None:
        """Reset the timer to initial state."""
        self.start_time = None
        self.end_time = None
        self._stopped = False

    @property
    def elapsed_seconds(self) -> float:
        """
        Get elapsed time in seconds.

        Returns:
            Elapsed time in seconds (0.0 if not started, current time if not stopped)

        Examples:
            >>> timer = Timer("test").start()
            >>> time.sleep(0.1)
            >>> print(f"{timer.elapsed_seconds:.2f} seconds")
        """
        if self.start_time is None:
            return 0.0

        end = self.end_time if self.end_time else time.perf_counter()
        return end - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """
        Get elapsed time in milliseconds.

        Returns:
            Elapsed time in milliseconds

        Examples:
            >>> with Timer("operation") as timer:
            ...     time.sleep(0.1)
            >>> print(f"{timer.elapsed_ms:.0f} ms")
        """
        return self.elapsed_seconds * 1000.0

    def __enter__(self) -> 'Timer':
        """Context manager entry - start timer."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop timer and log if enabled."""
        if not self._stopped:
            self.stop()

        if self.auto_logger:
            logger.info(f"{self.name} completed in {self.elapsed_ms:.2f} ms")

        return False

    def __repr__(self) -> str:
        """String representation."""
        status = "running" if self.start_time and not self._stopped else "stopped"
        return f"Timer(name='{self.name}', status={status}, elapsed={self.elapsed_ms:.2f}ms)"


@contextmanager
def measure_time(operation_name: str, log_level: str = "info"):
    """
    Context manager to measure and log operation time.

    Convenience function for quick timing measurements without
    creating a Timer object explicitly.

    Args:
        operation_name: Name of the operation being timed
        log_level: Logging level ('debug', 'info', 'warning', 'error')

    Examples:
        >>> with measure_time("batch_inference"):
        ...     # ... do inference work ...
        ...     pass
        # Logs: "batch_inference completed in 123.45 ms"

        >>> # Custom log level
        >>> with measure_time("debug_op", log_level="debug"):
        ...     # ... work ...
        ...     pass
    """
    timer = Timer(operation_name, auto_logger=False)
    timer.start()

    try:
        yield timer
    finally:
        if not timer._stopped:
            timer.stop()

        # Log at specified level
        log_func = getattr(logger, log_level.lower(), logger.info)
        log_func(f"{operation_name} completed in {timer.elapsed_ms:.2f} ms")


class TimedOperation:
    """
    Decorator for timing function execution.

    Examples:
        >>> @TimedOperation("expensive_computation")
        ... def compute():
        ...     # ... expensive work ...
        ...     return result
        >>> # Logs timing information automatically
    """

    def __init__(self, operation_name: Optional[str] = None, log_level: str = "info"):
        """
        Initialize timed operation decorator.

        Args:
            operation_name: Name for the operation (uses function name if None)
            log_level: Logging level for timing output
        """
        self.operation_name = operation_name
        self.log_level = log_level

    def __call__(self, func: Callable) -> Callable:
        """Decorator implementation."""

        def wrapper(*args, **kwargs) -> Any:
            name = self.operation_name or func.__name__
            log_func = getattr(logger, self.log_level.lower(), logger.info)

            with Timer(name, auto_logger=False) as timer:
                result = func(*args, **kwargs)

            log_func(f"{name} completed in {timer.elapsed_ms:.2f} ms")
            return result

        return wrapper


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable form.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45.67s")

    Examples:
        >>> format_duration(3661.5)
        '1h 1m 1.50s'
        >>> format_duration(0.123)
        '123.00ms'
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f}μs"
    elif seconds < 1:
        return f"{seconds * 1_000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


class PerformanceStats:
    """
    Collect and report performance statistics.

    Useful for tracking multiple timing measurements and
    generating summary statistics.

    Examples:
        >>> stats = PerformanceStats("inference")
        >>> for i in range(10):
        ...     with stats.measure():
        ...         # ... do work ...
        ...         pass
        >>> print(stats.summary())
    """

    def __init__(self, name: str):
        """
        Initialize performance stats collector.

        Args:
            name: Name for the stats collection
        """
        self.name = name
        self.timings: list[float] = []

    def measure(self) -> Timer:
        """
        Create a timer that auto-registers its timing.

        Returns:
            Timer that will register itself on exit

        Examples:
            >>> with stats.measure():
            ...     # ... work ...
            ...     pass
        """
        timer = Timer(f"{self.name}_measurement", auto_logger=False)

        class _AutoRegisterTimer(Timer):
            def __exit__(self2, exc_type, exc_val, exc_tb):
                result = super().__exit__(exc_type, exc_val, exc_tb)
                self.timings.append(self2.elapsed_ms)
                return result

        # Copy timer attributes to new instance
        art = _AutoRegisterTimer(timer.name, timer.auto_logger)
        return art

    def add_timing(self, duration_ms: float) -> None:
        """Manually add a timing measurement."""
        self.timings.append(duration_ms)

    @property
    def count(self) -> int:
        """Number of measurements."""
        return len(self.timings)

    @property
    def total_ms(self) -> float:
        """Total time in milliseconds."""
        return sum(self.timings)

    @property
    def avg_ms(self) -> float:
        """Average time in milliseconds."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def min_ms(self) -> float:
        """Minimum time in milliseconds."""
        return min(self.timings) if self.timings else 0.0

    @property
    def max_ms(self) -> float:
        """Maximum time in milliseconds."""
        return max(self.timings) if self.timings else 0.0

    def summary(self) -> str:
        """Generate summary statistics string."""
        if self.count == 0:
            return f"{self.name}: No measurements"

        return (
            f"{self.name}: {self.count} measurements, "
            f"avg={self.avg_ms:.2f}ms, "
            f"min={self.min_ms:.2f}ms, "
            f"max={self.max_ms:.2f}ms, "
            f"total={format_duration(self.total_ms / 1000)}"
        )

    def __repr__(self) -> str:
        return self.summary()

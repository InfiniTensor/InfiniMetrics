from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """
    Defines the standard interface for Adapters.
    Ensures that the Runner does not need code changes when extending to other Backends in the future.
    """

    @abstractmethod
    def process(self, legacy_data: dict) -> dict:
        """
        Process the legacy request and return the legacy response.
        """
        pass

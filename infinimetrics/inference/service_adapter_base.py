#!/usr/bin/env python3
"""
Service Adapter Base Class
Defines interface for service management
Separated from model interaction logic
"""

import abc
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ServiceAdapter(abc.ABC):
    """Base class for service management adapters"""
    
    def __init__(self):
        self.service_started = False
        self.server_port: Optional[int] = None
    
    @abc.abstractmethod
    def launch_service(self, port: int = 8000) -> None:
        """
        Launch inference service
        """
        pass
    
    @abc.abstractmethod
    def stop_service(self) -> None:
        """
        Stop inference service
        """
        pass
    
    @abc.abstractmethod
    def is_service_ready(self, port: int = 8000) -> bool:
        """
        Check whether the service is ready
        """
        pass
    
    @abc.abstractmethod
    def get_service_url(self) -> str:
        """
        Get service URL
        """
        pass


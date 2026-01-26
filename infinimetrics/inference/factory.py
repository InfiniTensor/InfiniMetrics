#!/usr/bin/env python3
"""
Factory helpers for inference framework components
"""


def create_framework_adapter(config):
    """
    Create model adapter for direct inference.
    """
    if config.framework == "infinilm":
        from .adapters.infinilm_adapter import InfiniLMAdapter
        return InfiniLMAdapter(config)
    else:
        from .adapters.vllm_adapter import VLLMAdapter
        return VLLMAdapter(config)


def create_service_manager(config):
    """
    Create service manager for service inference.
    """
    if config.framework == "infinilm":
        from .service_manager import InfiniLMServiceManager
        return InfiniLMServiceManager(config)
    else:
        from .service_manager import VLLMServiceManager
        return VLLMServiceManager(config)


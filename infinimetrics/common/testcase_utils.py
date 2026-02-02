#!/usr/bin/env python3
"""
Testcase Utilities
Common utilities for parsing testcase and generating run_id
Used by both inference and training modules
"""

import re
import random
import string
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def parse_testcase(testcase_str: str) -> tuple[str, str]:
    """
    Parse testcase string to extract framework and mode
    
    Args:
        testcase_str: testcase string, e.g., "infer.InfiniLM.Direct"
    
    Returns:
        tuple: (mode, framework) where mode is "direct" or "service", 
               framework is lowercase framework name
    
    Raises:
        ValueError: if cannot determine framework
    """
    testcase_lower = testcase_str.lower()
    
    # Determine mode
    if "service" in testcase_lower:
        mode = "service"
    elif "direct" in testcase_lower:
        mode = "direct"
    else:
        mode = "direct"  # default mode
    
    # Determine framework (more flexible parsing)
    if "vllm" in testcase_lower:
        framework = "vllm"
    elif "infinilm" in testcase_lower:
        framework = "infinilm"
    elif "megatron" in testcase_lower:
        framework = "megatron"
    elif "infinitrain" in testcase_lower:
        framework = "infinitrain"
    else:
        # Try to extract from dot-separated parts
        parts = testcase_str.split('.')
        if len(parts) < 2:
            raise ValueError(f"Cannot determine framework from testcase: {testcase_str}")
    
        framework_candidate = parts[-2].lower()
        if framework_candidate not in ["infinilm", "vllm", "bmtrain", "megatron", "infinitrain"]:
            raise ValueError(f"Cannot determine framework from testcase: {testcase_str}")
    
        framework = framework_candidate
    
    return mode, framework


def generate_run_id(testcase: str, user_run_id: Optional[str] = None) -> str:
    """
    Generate or enhance run_id
    
    Args:
        testcase: testcase string
        user_run_id: user-provided run_id (optional)
    
    Returns:
        Generated run_id
    """
    if user_run_id:
        return enhance_user_run_id(user_run_id)
    else:
        return generate_auto_run_id(testcase)

def generate_run_id_from_config(config_dict: Dict[str, Any]) -> str:
    """Generate run_id from configuration dictionary"""
    outer_run_id = config_dict.get("run_id")
    outer_testcase = config_dict.get("testcase")
    config_data = config_dict.get("config", {})
    
    # Validate testcase position
    inner_testcase = config_data.get("testcase")
    if outer_testcase:
        if not validate_testcase_format(outer_testcase):
            raise ValueError(f"Invalid testcase format: {outer_testcase}")
    elif inner_testcase:
        raise ValueError(
            "testcase must be at the outer level, not inside 'config'. "
            f"Found: '{inner_testcase}' inside 'config'."
        )
    else:
        raise ValueError("testcase is required at the outer level of the config.")
    
    # Validate run_id position
    inner_run_id = config_data.get("run_id")
    if inner_run_id:
        raise ValueError(
            "run_id must be at the outer level, not inside 'config'. "
            f"Found: '{inner_run_id}' inside 'config'."
        )
    
    # Generate run_id using the existing function
    return generate_run_id(outer_testcase, outer_run_id)

def enhance_user_run_id(user_run_id: str) -> str:
    """Enhance user-provided run_id by adding timestamp and random code"""
    # If already contains timestamp and random code, return directly
    timestamp_pattern = r'\.\d{8}_\d{6}\.[a-z0-9]{8}$'
    if re.search(timestamp_pattern, user_run_id):
        logger.info(f"User run_id already contains timestamp and random code: {user_run_id}")
        return user_run_id

    # Clean user run_id
    cleaned_user_id = user_run_id.strip().strip(".").replace("..", ".")

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 8-character random code
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    # Combine
    enhanced_run_id = f"{cleaned_user_id}.{timestamp}.{random_suffix}"

    logger.info(f"Enhanced user run_id: {user_run_id} -> {enhanced_run_id}")
    return enhanced_run_id


def generate_auto_run_id(testcase: str) -> str:
    """Auto-generate run_id"""
    # Clean testcase
    cleaned_testcase = testcase.strip().strip(".").replace("..", ".")

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 8-character random code
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    # Combine
    run_id = f"{cleaned_testcase}.{timestamp}.{random_suffix}"

    return run_id


def validate_testcase_format(testcase: str) -> bool:
    """Validate testcase format"""
    # Basic validation: should contain at least 2 parts separated by dots
    parts = testcase.split('.')
    if len(parts) < 2:
        return False
    
    # Should contain valid mode
    testcase_lower = testcase.lower()
    if not ("direct" in testcase_lower or "service" in testcase_lower):
        return False
    
    return True


def extract_testcase_components(testcase: str) -> dict:
    """Extract components from testcase string"""
    parts = testcase.split('.')
    
    result = {
        "full": testcase,
        "parts": parts,
        "domain": parts[0] if len(parts) > 0 else "",
        "framework": parts[1] if len(parts) > 1 else "",
        "mode": "service" if "service" in testcase.lower() else "direct"
    }
    
    return result

#!/usr/bin/env python3
import argparse
import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import ConfigManager, create_gpu_monitor

def create_training_runner(framework, config_manager, gpu_monitor):
    """Factory function to create training runner"""
    if framework.lower() == "megatron":
        from frameworks import MegatronRunner
        return MegatronRunner(config_manager, gpu_monitor)
    elif framework.lower() == "infinitrain":
        try:
            from frameworks import InfinitrainRunner
            return InfinitrainRunner(config_manager, gpu_monitor)
        except ImportError:
            print("Warning: Infinitrain runner not found, using Megatron as fallback")
            from frameworks import MegatronRunner
            return MegatronRunner(config_manager, gpu_monitor)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config.json")
    parser.add_argument("--framework", default="megatron", choices=["megatron", "infinitrain"], 
                       help="training framework to use")
    parser.add_argument("--gpu-platform", default="nvidia", choices=["nvidia", "other"],
                       help="GPU platform for monitoring")
    args = parser.parse_args()
    
    # Create configuration manager
    config_manager = ConfigManager(args.config)
    
    # Create GPU monitor
    gpu_monitor = create_gpu_monitor(args.gpu_platform)
    
    # Create training runner
    runner = create_training_runner(args.framework, config_manager, gpu_monitor)
    
    # Execute training
    result_json = runner.run()
    print(f"\nTraining completed. Results saved to: {result_json}")

if __name__ == "__main__":
    main()

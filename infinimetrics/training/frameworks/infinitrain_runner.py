from core.training_runner import TrainingRunner

class InfinitrainRunner(TrainingRunner):
    """Infinitrain training runner implementation (placeholder)"""
    
    def build_training_command(self):
        """Build Infinitrain training command"""
        raise NotImplementedError("Infinitrain runner is not implemented yet")
    
    def parse_training_output(self, line, metrics):
        """Parse Infinitrain training output"""
        raise NotImplementedError("Infinitrain runner is not implemented yet")

import json
import os

class ConfigManager:
    """Configuration manager class, unified handling of training configuration parameters"""
    
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()
        self.extract_parameters()
    
    def load_config(self):
        """Load configuration file"""
        with open(self.config_path, "r") as f:
            cfg = json.load(f)
        
        self.conf = cfg.get("config", {})
        self.raw_config = cfg
    
    def extract_parameters(self):
        """Extract and standardize configuration parameters"""
        # Model configuration
        self.model_name = self.conf.get("model", "gpt").lower()
        
        # Training parameters
        train_args = self.conf.get("train_args", {})
        self.train_args = train_args
        
        # Parallel configuration
        parallel = train_args.get("parallel", {})
        self.dp = int(parallel.get("dp", 1))
        self.tp = int(parallel.get("tp", 1))
        
        pp_raw = parallel.get("pp", 1)
        if isinstance(pp_raw, dict):
            self.pp_size = int(pp_raw.get("size", 1))
            self.pp_type = str(pp_raw.get("type", "default")).lower()
        else:
            self.pp_size = int(pp_raw)
            self.pp_type = "default"
        
        self.sp = int(parallel.get("sp", 0))
        
        # Training hyperparameters
        self.mbs = int(train_args.get("mbs", 1))
        self.gbs = int(train_args.get("gbs", max(1, self.mbs)))
        self.seq_len = int(train_args.get("seq_len", train_args.get("seq", 128)))
        self.lr = train_args.get("lr", 0.00015)
        self.train_iters = int(train_args.get("step", train_args.get("train_iters", train_args.get("train-iters", 10))))
        self.num_layers = int(train_args.get("num_layers", train_args.get("num-layers", 2)))
        self.hidden_size = int(train_args.get("hidden_size", train_args.get("hidden-size", 512)))
        self.num_attention_heads = int(train_args.get("num_attention_heads", train_args.get("num-attention-heads", 8)))
        self.max_position_embeddings = int(train_args.get("max_position_embeddings", train_args.get("max-position-embeddings", self.seq_len)))
        self.vocab_size = int(train_args.get("vocab_size", train_args.get("vocab-size", 128256)))
        
        # Dataset configuration
        self.train_dataset = self.conf.get("train_dataset", None)
        self.validation_dataset = self.conf.get("validation_dataset", None)
        self.test_dataset = self.conf.get("test_dataset", None)
        
        # Runtime configuration
        self.output_dir = self.conf.get("output_dir", "./train")
        self.timeout_ms = train_args.get("timeout_ms", 10000)
        
        # Extract warmup and measured iterations
        self.warmup_iterations = int(train_args.get("warmup_iterations", train_args.get("warmup", 0)))
        self.measured_iterations = int(train_args.get("measured_iterations", self.train_iters))
    
    def get_model_config(self):
        """Get model configuration dictionary"""
        return {
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size
        }
    
    def get_parallel_config(self):
        """Get parallel configuration"""
        return {
            "dp": self.dp,
            "tp": self.tp,
            "pp": {
                "size": self.pp_size,
                "type": self.pp_type
            },
            "sp": self.sp
        }
    
    def get_training_args(self):
        """Get training arguments"""
        return self.train_args

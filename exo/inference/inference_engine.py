import torch  # Assuming PyTorch or similar GPU access library is available
import numpy as np
import os
from exo.helpers import DEBUG  # Make sure to import DEBUG
from typing import Tuple, Optional, List
from abc import ABC, abstractmethod
from .shard import Shard

class InferenceEngine(ABC):
    @abstractmethod
    async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        pass

    @abstractmethod
    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        pass

def get_available_gpus() -> List[int]:
    """Returns a list of available GPU indices."""
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    else:
        print("Warning: No GPUs detected. Running on CPU.")
        return []

def get_inference_engine(inference_engine_name: str, shard_downloader: 'ShardDownloader'):
    if DEBUG >= 2:
        print(f"get_inference_engine called with: {inference_engine_name}")
    
    available_gpus = get_available_gpus()
    if DEBUG >= 1:
        print(f"Detected GPUs: {available_gpus}")

    if inference_engine_name == "mlx":
        from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
        return MLXDynamicShardInferenceEngine(shard_downloader, devices=available_gpus)
    
    elif inference_engine_name == "tinygrad":
        from exo.inference.tinygrad.inference import TinygradDynamicShardInferenceEngine
        import tinygrad.helpers
        tinygrad.helpers.DEBUG.value = int(os.getenv("TINYGRAD_DEBUG", default="0"))
        
        return TinygradDynamicShardInferenceEngine(shard_downloader, devices=available_gpus)
    
    elif inference_engine_name == "dummy":
        from exo.inference.dummy_inference_engine import DummyInferenceEngine
        return DummyInferenceEngine()
    
    raise ValueError(f"Unsupported inference engine: {inference_engine_name}")

from pathlib import Path
import json
import os
from exo.inference.tinygrad.models.llama import Transformer, convert_from_huggingface, fix_bf16
from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from tinygrad.nn.state import load_state_dict
from tinygrad import Tensor, nn, Context
from exo.inference.inference_engine import InferenceEngine
from typing import Optional, Tuple
import numpy as np
from exo.inference.tinygrad.tinygrad_helpers import concat_weights, load
from exo.download.shard_download import ShardDownloader
from concurrent.futures import ThreadPoolExecutor
import asyncio

Tensor.no_grad = True
# default settings
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.85))
TOP_K = 25
TOP_P = 0.9
ALPHA_F = 0.1
ALPHA_P = 0.0
MODEL_PARAMS = {
    "8B": {"args": {"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 14336}, "files": 1},
    "70B": {"args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 28672}, "files": 8}
}

def build_transformer(model_path: Path, shard: Shard, model_size="8B", devices=None):
    # build model
    linear = nn.Linear
    with Context(THREEFRY=0):
        model = Transformer(**MODEL_PARAMS[model_size]["args"], linear=linear, max_context=8192, jit=True, shard=shard)

    # load weights and distribute across devices if multiple are available
    if model_path.is_dir():
        if (model_path / "model.safetensors.index.json").exists():
            weights = load(str(model_path / "model.safetensors.index.json"), shard)
        elif (model_path / "model.safetensors").exists():
            weights = load(str(model_path / "model.safetensors"), shard)
        else:
            weights = concat_weights(
                [load(str(model_path / f"consolidated.{i:02d}.pth"), shard) for i in range(MODEL_PARAMS[model_size]["files"])],
                devices[0] if isinstance(devices, list) and devices else None
            )
    else:
        weights = load(str(model_path), shard)

    weights = convert_from_huggingface(weights, model, MODEL_PARAMS[model_size]["args"]["n_heads"], MODEL_PARAMS[model_size]["args"]["n_kv_heads"])
    weights = fix_bf16(weights)

    for i, device in enumerate(devices or [None]):
        with Context(device=device):
            load_state_dict(model, weights, strict=False, consume=False)  # consume=True if needed

    return model

class TinygradDynamicShardInferenceEngine(InferenceEngine):
    def __init__(self, shard_downloader: ShardDownloader, devices=None):
        self.shard = None
        self.shard_downloader = shard_downloader
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.devices = devices or [0]  # Default to GPU 0 if no devices are provided

    async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        await self.ensure_shard(shard)
        start_pos = json.loads(inference_state or "{}").get("start_pos", 0)
        n_captured_toks = json.loads(inference_state or "{}").get("n_captured_toks", 0)

        toks = await asyncio.get_event_loop().run_in_executor(self.executor, self.tokenizer.encode, prompt)
        h_results = []

        for device in self.devices:
            with Context(device=device):
                h = await asyncio.get_event_loop().run_in_executor(self.executor, lambda: self.model(Tensor([toks]), start_pos, TEMPERATURE).realize())
                h_results.append(h)

        h = self.aggregate_results(h_results)

        if h.shape == (1,):
            start_pos += len(toks) + 1
            n_captured_toks = 0
            return np.array([[h.item()]]), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), h.item() == self.tokenizer.eos_token_id
        else:
            n_captured_toks = len(toks)
            return h.numpy(), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), False

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
        await self.ensure_shard(shard)
        start_pos = json.loads(inference_state or "{}").get("start_pos", 0)
        n_captured_toks = json.loads(inference_state or "{}").get("n_captured_toks", 0)

        h_results = []
        for device in self.devices:
            with Context(device=device):
                h = await asyncio.get_event_loop().run_in_executor(self.executor, lambda: self.model(Tensor(input_data), start_pos, TEMPERATURE).realize())
                h_results.append(h)

        h = self.aggregate_results(h_results)

        if h.shape == (1,):
            start_pos += n_captured_toks + 1
            n_captured_toks = 0
            return np.array([[h.item()]]), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), h.item() == self.tokenizer.eos_token_id
        else:
            return h.numpy(), json.dumps({"start_pos": start_pos, "n_captured_toks": n_captured_toks}), False

    async def ensure_shard(self, shard: Shard):
        if self.shard == shard:
            return

        model_path = await self.shard_downloader.ensure_shard(shard)

        if self.shard != shard:
            self.model = await asyncio.get_event_loop().run_in_executor(
                self.executor, build_transformer, model_path, shard, "8B" if "8b" in shard.model_id.lower() else "70B", self.devices
            )

            tokenizer_path = str((model_path if model_path.is_dir() else model_path.parent))
            self.tokenizer = await resolve_tokenizer(tokenizer_path)
            self.shard = shard

    def aggregate_results(self, results):
        # This function can be adjusted based on how aggregation should be handled. Currently, it returns the first result.
        return results[0]

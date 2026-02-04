"""LoRA adapter provider for MLX backend.

Downloads LoRA checkpoints via Moondream API and converts them to MLX tensors.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, Any

import requests

import mlx.core as mx

from md3.lora import TextLoRA
from md3.config import TextConfig

logger = logging.getLogger(__name__)


class AdapterLoadError(Exception):
    pass


class MoondreamAdapterProvider:
    def __init__(
        self,
        text_config: TextConfig,
        *,
        api_key: str,
        api_base_url: str = "https://api.moondream.ai",
        max_lora_rank: int = 16,
        cache_size: int = 32,
        dtype: Optional[mx.Dtype] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._text_config = text_config
        self._api_key = api_key
        self._api_base_url = api_base_url.rstrip("/")
        self._max_lora_rank = max_lora_rank
        self._cache_size = cache_size
        # Default to bf16 to match the base model precision.
        self._dtype = dtype or mx.bfloat16
        self._cache: OrderedDict[str, TextLoRA] = OrderedDict()
        self._cache_dir = cache_dir or self._default_cache_dir()

    @staticmethod
    def _default_cache_dir() -> Path:
        env_dir = os.environ.get("MOONDREAM_STATION_MODELS_DIR")
        if env_dir:
            return Path(env_dir).expanduser()

        config_path = Path.home() / ".moondream-station" / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)
                models_dir = data.get("models_dir")
                if models_dir:
                    return Path(models_dir).expanduser()
            except Exception:
                pass
        return Path.home() / ".moondream-station" / "models"

    def get(self, adapter: str, *, dtype: Optional[mx.Dtype] = None) -> TextLoRA:
        if adapter in self._cache:
            self._cache.move_to_end(adapter)
            return self._cache[adapter]

        if "@" not in adapter:
            raise ValueError(
                f"Invalid adapter ID format: '{adapter}'. Expected format: 'finetune_id@step'"
            )
        finetune_id, step = adapter.split("@", 1)
        if not finetune_id or not step:
            raise ValueError(
                f"Invalid adapter ID format: '{adapter}'. Expected format: 'finetune_id@step'"
            )

        lora = self._load_adapter(finetune_id, step, dtype=dtype or self._dtype)
        self._cache[adapter] = lora
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return lora

    def _load_adapter(self, finetune_id: str, step: str, dtype: Optional[mx.Dtype]) -> TextLoRA:
        ckpt_path = self._find_cached_checkpoint(finetune_id, step)
        if ckpt_path is None:
            presigned_url = self._fetch_presigned_url(finetune_id, step)
            ckpt_path = self._download_checkpoint(finetune_id, step, presigned_url)
        checkpoint = self._load_checkpoint(ckpt_path)

        if not isinstance(checkpoint, dict):
            raise AdapterLoadError("Invalid checkpoint format: expected dict")

        state_dict = checkpoint.get("lora_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise AdapterLoadError("Invalid checkpoint format: missing lora_state_dict")

        lora = TextLoRA.from_state_dict(
            state_dict,
            text_config=self._text_config,
            max_rank=self._max_lora_rank,
            dtype=dtype or self._dtype,
            adapter_id=f"{finetune_id}@{step}",
        )
        return lora

    def _fetch_presigned_url(self, finetune_id: str, step: str) -> str:
        url = (
            f"{self._api_base_url}"
            f"/v1/tuning/finetunes/{finetune_id}/checkpoints/{step}/download"
        )
        try:
            response = requests.get(url, headers={"X-Moondream-Auth": self._api_key}, timeout=60)
            if response.status_code == 404:
                raise AdapterLoadError(
                    f"Adapter not found: finetune_id={finetune_id}, step={step}"
                )
            if response.status_code == 410:
                raise AdapterLoadError(
                    f"Adapter checkpoint is no longer available: finetune_id={finetune_id}, step={step}"
                )
            response.raise_for_status()
            data = response.json()
            presigned_url = data.get("url")
            if not presigned_url:
                raise AdapterLoadError("API response missing 'url' field")
            return presigned_url
        except requests.HTTPError as e:
            raise AdapterLoadError(f"Failed to fetch presigned URL: {e}") from e
        except requests.RequestException as e:
            raise AdapterLoadError(f"Network error fetching presigned URL: {e}") from e

    def _download_checkpoint(self, finetune_id: str, step: str, presigned_url: str) -> Path:
        adapter_dir = self._cache_dir / "loras" / "mlx" / finetune_id / step
        adapter_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = adapter_dir / "adapter.pt"

        if ckpt_path.exists() and ckpt_path.stat().st_size > 0:
            return ckpt_path

        try:
            response = requests.get(presigned_url, timeout=60)
            response.raise_for_status()
            ckpt_path.write_bytes(response.content)
            return ckpt_path
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                raise AdapterLoadError("Presigned URL expired or invalid") from e
            raise AdapterLoadError(f"Failed to download checkpoint: {e}") from e
        except requests.RequestException as e:
            raise AdapterLoadError(f"Network error downloading checkpoint: {e}") from e

    def _find_cached_checkpoint(self, finetune_id: str, step: str) -> Optional[Path]:
        adapter_dir = self._cache_dir / "loras" / "mlx" / finetune_id / step
        for name in ("adapter.pt", "adapter.safetensors", "adapter.npz"):
            path = adapter_dir / name
            if path.exists() and path.stat().st_size > 0:
                return path
        return None

    def _load_checkpoint(self, path: Path) -> Dict[str, Any]:
        if path.suffix in {".safetensors", ".npz"}:
            data = mx.load(str(path))
            return dict(data)

        try:
            import torch
        except Exception as e:
            raise AdapterLoadError(
                "torch is required to load LoRA checkpoints from Moondream API. "
                "Install torch or provide a pre-converted safetensors LoRA file."
            ) from e

        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # Older torch versions do not support weights_only
            return torch.load(path, map_location="cpu")

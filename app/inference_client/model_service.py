import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self, model_name: str, revision: str):
        self.model_name = model_name
        self.revision = revision
        self.device = self._get_best_device()
        logger.info(f"Initializing model on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=None, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=None, trust_remote_code=True
        )
        self.model.to(self.device)
        logger.info(f"Model commit hash: {self.model.config._commit_hash}")

    @staticmethod
    def _get_best_device() -> str:
        """Determine the best available device: CUDA, then MPS, then CPU."""
        if torch.cuda.is_available():
            return "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def caption(
        self,
        image: Image.Image,
        length: str,
        stream: bool = False,
        settings: dict = {},
        variant: str = None,
    ) -> dict:
        settings["variant"] = variant
        return self.model.caption(
            image, length=length, stream=stream, settings=settings
        )

    def query(
        self,
        image: Image.Image,
        question: str,
        stream: bool = False,
        settings: dict = {},
        variant: str = None,
        reasoning: bool = False,
    ) -> dict:
        settings["variant"] = variant
        return self.model.query(
            image, question, stream=stream, settings=settings, reasoning=reasoning
        )

    def detect(
        self,
        image: Image.Image,
        obj: str,
        settings: dict = {},
        variant=None,
    ) -> dict:
        settings["variant"] = variant
        return self.model.detect(image, obj, settings)

    def point(
        self, image: Image.Image, obj: str, settings: dict = {}, variant=None
    ) -> dict:
        settings["variant"] = variant
        return self.model.point(image, obj, settings)

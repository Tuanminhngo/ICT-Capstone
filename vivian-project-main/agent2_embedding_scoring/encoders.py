from __future__ import annotations



import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import Settings

LOGGER = logging.getLogger(__name__)


#  Base interface 

class BaseEncoder(ABC):
    """Abstract encoder: maps a video clip path and a text string to embeddings."""

    model_name: str = "base"

    @abstractmethod
    def encode_video(self, clip_path: Path) -> Optional[np.ndarray]:
        """Return a 1-D unit-normalised embedding for the given clip, or None on failure."""

    @abstractmethod
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """Return a 1-D unit-normalised embedding for the given text string, or None on failure."""

    @staticmethod
    def l2_normalise(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return vector
        return vector / norm


#  CLIP encoder 

class CLIPEncoder(BaseEncoder):
    

    model_name = "clip"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._preprocess = None
        self._device = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            import clip as openai_clip  

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            LOGGER.info("Loading CLIP model '%s' on %s …", self.settings.clip_model_variant, self._device)
            self._model, self._preprocess = openai_clip.load(
                self.settings.clip_model_variant, device=self._device
            )
            self._model.eval()
            LOGGER.info("CLIP model loaded.")
        except ImportError as exc:
            raise ImportError(
                "The 'clip' package is required for CLIPEncoder.  "
                "Install it with:  pip install git+https://github.com/openai/CLIP.git"
            ) from exc

    def _sample_frames(self, clip_path: Path) -> Optional[List[np.ndarray]]:
        """Return a list of RGB uint8 arrays sampled from the clip."""
        try:
            import cv2  # opencv-python
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for frame sampling.  Install with:  pip install opencv-python"
            ) from exc

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            LOGGER.error("Cannot open clip: %s", clip_path)
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            LOGGER.warning("Clip reports 0 frames: %s", clip_path)
            cap.release()
            return None

        n = min(self.settings.frames_per_clip, total_frames)
        # Evenly-spaced indices within [0, total_frames - 1]
        indices = [int(round(i * (total_frames - 1) / max(n - 1, 1))) for i in range(n)]
        # Deduplicate while preserving order 
        seen = set()
        indices = [i for i in indices if not (i in seen or seen.add(i))]

        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        if not frames:
            LOGGER.warning("No frames could be read from clip: %s", clip_path)
            return None

        return frames

    def encode_video(self, clip_path: Path) -> Optional[np.ndarray]:
        import torch
        from PIL import Image

        frames = self._sample_frames(clip_path)
        if frames is None:
            return None

        processed: List[torch.Tensor] = [
            self._preprocess(Image.fromarray(f)) for f in frames
        ]

        # Encode in mini-batches to avoid OOM on large frame counts
        batch_size = self.settings.frame_batch_size
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(processed), batch_size):
            batch = processed[start : start + batch_size]
            batch_tensor = torch.stack(batch).to(self._device)
            with torch.no_grad():
                features = self._model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())

        stacked = np.concatenate(all_embeddings, axis=0)  
        mean_embedding = stacked.mean(axis=0)
        return self.l2_normalise(mean_embedding)

    def encode_text(self, text: str) -> Optional[np.ndarray]:
        import torch
        import clip as openai_clip

        try:
            tokens = openai_clip.tokenize([text], truncate=True).to(self._device)
            with torch.no_grad():
                features = self._model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            return self.l2_normalise(features.cpu().numpy()[0])
        except Exception as exc: 
            LOGGER.error("CLIP text encoding failed for text='%.60s': %s", text, exc)
            return None


#  ImageBind encoder 

class ImageBindEncoder(BaseEncoder):
    

    model_name = "imagebind"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._device = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from imagebind import data as ib_data  
            from imagebind.models import imagebind_model
            from imagebind.models.imagebind_model import ModalityType  

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            if self._device == "cpu":
                LOGGER.warning(
                    "ImageBind is running on CPU.  Inference will be very slow (~30–120 s per clip).  "
                    "This encoder is intended only for the small model-comparison sample."
                )
            LOGGER.info("Loading ImageBind model on %s …", self._device)
            self._model = imagebind_model.imagebind_huge(pretrained=True)
            self._model.eval()
            self._model.to(self._device)
            LOGGER.info("ImageBind model loaded.")
        except ImportError as exc:
            raise ImportError(
                "The 'imagebind' package is required for ImageBindEncoder.  "
                "Install it with:  pip install imagebind  "
                "(or from source: pip install git+https://github.com/facebookresearch/ImageBind.git)"
            ) from exc

    def encode_video(self, clip_path: Path) -> Optional[np.ndarray]:
        try:
            import torch
            from imagebind import data as ib_data
            from imagebind.models.imagebind_model import ModalityType

            inputs = {
                ModalityType.VISION: ib_data.load_and_transform_video_data(
                    [str(clip_path)], self._device
                )
            }
            with torch.no_grad():
                embeddings = self._model(inputs)
            raw = embeddings[ModalityType.VISION].cpu().numpy()[0]
            return self.l2_normalise(raw)
        except Exception as exc:  
            LOGGER.error("ImageBind video encoding failed for '%s': %s", clip_path, exc)
            return None

    def encode_text(self, text: str) -> Optional[np.ndarray]:
        try:
            import torch
            from imagebind import data as ib_data
            from imagebind.models.imagebind_model import ModalityType

            inputs = {
                ModalityType.TEXT: ib_data.load_and_transform_text([text], self._device)
            }
            with torch.no_grad():
                embeddings = self._model(inputs)
            raw = embeddings[ModalityType.TEXT].cpu().numpy()[0]
            return self.l2_normalise(raw)
        except Exception as exc:  
            LOGGER.error("ImageBind text encoding failed for text='%.60s': %s", text, exc)
            return None


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_encoder(model_name: str, settings: Settings) -> BaseEncoder:
    
    name = model_name.strip().lower()
    if name == "clip":
        return CLIPEncoder(settings)
    if name == "imagebind":
        return ImageBindEncoder(settings)
    raise ValueError(
        f"Unknown encoder '{model_name}'.  Supported values: 'clip', 'imagebind'."
    )

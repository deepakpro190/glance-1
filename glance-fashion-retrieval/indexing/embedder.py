import torch
import numpy as np

from transformers import AutoModel, AutoProcessor


class SigLIPEmbedder:

    def __init__(
        self,
        model_name="google/siglip-base-patch16-224",
        device=None
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"Using device : {self.device}")

        print("Loading SigLIP...")

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(model_name)

        self.model.to(self.device)

        self.model.eval()

        print("SigLIP Loaded Successfully!")

    def embed_batch(self, images):

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.get_image_features(**inputs)

            # transformers may return either a tensor or an object with pooler_output
            if hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs

            embeddings = embeddings / embeddings.norm(
                dim=-1,
                keepdim=True
            )

        return embeddings.cpu().numpy().astype(np.float32)
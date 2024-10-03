import os
import torch
import numpy as np
import pickle
from PIL import Image
from einops import rearrange
from diffusers.models.vae import DiagonalGaussianDistribution
from typing import List, Dict, Tuple, Union, Any
from pathlib import Path

# --- Checkpoint Downloading (One-time operation) ---

def download_checkpoint(checkpoint_url: str, save_dir: str) -> str:
    """Downloads and saves a checkpoint file."""
    import requests
    from tqdm import tqdm

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = checkpoint_url.split("/")[-1]
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        print(f"Checkpoint already exists at {save_path}")
        return save_path

    response = requests.get(checkpoint_url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("Error, something went wrong")
        raise RuntimeError("Download failed")

    return save_path


# --- Core Glaze Algorithm ---

class Glaze:
    def __init__(self, params: Dict[str, Any], device: str, target_params: Dict[str, Any], project_root_path: str, jpg: bool):
        self.params = params
        self.device = device
        self.jpg = jpg
        self.half = device == "cuda"
        self.target_params = target_params
        self.project_root_path = project_root_path
        self.preview_mask_tensor = None  # For preview mode

        # Load the necessary models (glaze.p, glaze-qc.p, and preview mask)
        self.load_models()


    def load_models(self) -> None:

        # Download checkpoints if they don't exist
        if not os.path.exists(os.path.join(self.project_root_path, 'glaze.p')):
            download_checkpoint("http://mirror.cs.uchicago.edu/fawkes/files/glaze/glaze.p", self.project_root_path)

        if not os.path.exists(os.path.join(self.project_root_path, 'glaze-qc.p')):
            download_checkpoint("http://mirror.cs.uchicago.edu/fawkes/files/glaze/glaze-qc.p", self.project_root_path)

        if not os.path.exists(os.path.join(self.project_root_path, 'preview_mask.p')):
            download_checkpoint("http://mirror.cs.uchicago.edu/fawkes/files/glaze/preview_mask.p", self.project_root_path)



        self.model = torch.load(os.path.join(self.project_root_path, 'glaze.p'), map_location=torch.device('cpu')).to(self.device).to(torch.float32)
        self.model_qc = torch.load(os.path.join(self.project_root_path, 'glaze-qc.p'), map_location=torch.device('cpu')).to(self.device).to(torch.float32)
        if self.half:
            self.model = self.model.half()
            self.model_qc = self.model_qc.half()

        if self.params["opt_setting"] == "0": # Preview mode
            preview_mask = pickle.load(open(os.path.join(self.project_root_path, 'preview_mask.p'), 'rb'))
            self.preview_mask_tensor = torch.tensor(preview_mask, dtype=torch.float32).to(self.device)
            if self.half:
                self.preview_mask_tensor = self.preview_mask_tensor.half()




    def model_encode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        h = self.model(input_tensor)
        moments = self.model_qc(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.mean


    def img2tensor(self, img: Image.Image) -> torch.Tensor:
        img = np.array(img).astype(np.float32)
        img = (img / 127.5 - 1).astype(np.float32)
        img = rearrange(img, 'h w c -> c h w')
        img = torch.tensor(img).unsqueeze(0).to(self.device)
        if self.half:
            img = img.half()
        return img


    def tensor2img(self, tensor: torch.Tensor) -> Image.Image:
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        tensor = torch.clamp((tensor.detach() + 1) / 2, 0, 1)
        img = 255 * rearrange(tensor[0], 'c h w -> h w c').cpu().numpy()
        return Image.fromarray(img.astype(np.uint8))

    def segment_image(self, img: Image.Image) -> Tuple[List[Image.Image], int, int]:
        """Segments image into 512x512 squares."""
        img_array = np.array(img).astype(np.float32)
        og_width, og_height = img.size
        short_height = og_height <= og_width

        segments = []
        last_index = 0
        square_size = og_height if short_height else og_width

        if short_height:
            for cur_idx in range(0, og_width, square_size):
                if cur_idx + square_size < og_width:
                    cropped_square = img_array[0:square_size, cur_idx:cur_idx + square_size, :]
                else:
                    cropped_square = img_array[0:square_size, -square_size:, :]
                    last_index = square_size - (og_width - cur_idx)
                segments.append(Image.fromarray(cropped_square.astype(np.uint8)).resize((512, 512)))

        else:  # og_height > og_width
            for cur_idx in range(0, og_height, square_size):
                if cur_idx + square_size < og_height:
                    cropped_square = img_array[cur_idx : cur_idx + square_size, 0:square_size, :]
                else:
                    cropped_square = img_array[-square_size:, 0:square_size, :]
                    last_index = square_size - (og_height - cur_idx)
                segments.append(Image.fromarray(cropped_square.astype(np.uint8)).resize((512, 512)))

        return segments, last_index, square_size


    def compute_batch(self, source_batch: torch.Tensor, target_batch: torch.Tensor, max_change: float, tot_steps: int) -> torch.Tensor:
        """Computes the adversarial perturbation for a batch of segments."""

        if self.preview_mask_tensor is not None:  # Preview mode
            cloaked_batch = source_batch + self.preview_mask_tensor
            return torch.clamp(cloaked_batch, -1, 1)


        X_batch = source_batch.clone().detach().to(self.device)
        with torch.no_grad():
            target_emb = self.model_encode(target_batch).detach()

        step_size = max_change * 0.5 if tot_steps > 10 else max_change * 0.75
        modifiers = torch.zeros_like(X_batch)

        for i in range(tot_steps):
            modifiers.requires_grad_(True)
            X_adv = torch.clamp(modifiers + X_batch, -1, 1)

            loss_normal = (self.model_encode(X_adv) - target_emb).norm()
            tot_loss = loss_normal
            grad = torch.autograd.grad(tot_loss, modifiers)[0].detach()

            modifiers = modifiers.detach()
            final_cur_update = grad.sign() * step_size * (1 - 0.01 * (i / tot_steps * 100))
            modifiers = torch.clamp(modifiers - final_cur_update, -max_change, max_change)


        best_adv_tensors = torch.clamp(modifiers + X_batch, -1, 1)
        return best_adv_tensors

    def cloak_image(self, image: Image.Image) -> Union[Image.Image, None]:
        """Applies the cloaking perturbation to the entire image."""

        segments, last_index, square_size = self.segment_image(image)
        source_segments = torch.cat([self.img2tensor(seg) for seg in segments])

        # Create dummy target segments (replace with your actual target generation if needed)
        target_segments = source_segments.clone()  # Or generate your target segments

        cloaked_segments = self.compute_batch(
            source_segments, target_segments, self.params['max_change'], self.params['tot_steps']
        )

        cloaked_image_array = np.array(image).astype(np.float32)
        og_height, og_width, _ = cloaked_image_array.shape
        short_height = og_height <= og_width


        for idx, cloaked_tensor in enumerate(cloaked_segments):
            cloaked_segment = self.tensor2img(cloaked_tensor)
            cloaked_segment = np.array(cloaked_segment.resize((square_size, square_size))).astype(np.float32)


            if short_height:
                start_x = idx * square_size
                end_x = min((idx + 1) * square_size, og_width)  # Handle edge case
                cloaked_image_array[0:square_size, start_x:end_x, :] = cloaked_segment[:, :end_x - start_x, :]  # Adjust slicing
            else:
                start_y = idx * square_size
                end_y = min((idx+1) * square_size, og_height)
                cloaked_image_array[start_y:end_y, 0:square_size, :] = cloaked_segment[:end_y - start_y, :, :]


        return Image.fromarray(cloaked_image_array.astype(np.uint8))
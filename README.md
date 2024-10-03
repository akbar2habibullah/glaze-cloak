# Glaze Image Cloaking

This module provides the `Glaze` class for applying the Glaze image cloaking algorithm. This code is ported from original implementation of Glaze in this [repository](https://github.com/EspacioLatente/Glaze)

## Installation

```bash
pip install glaze-cloak
```

## Usage

```
from glaze-cloak import Glaze
from PIL import Image

# Example parameters:
params = {
    "max_change": 0.05,
    "n_runs": 1,
    "tot_steps": 25,
    "setting": 50,
    "opt_setting": "1",
}

target_params = {  # Placeholder, adapt as needed
    "style": "your_target_style",
    "strength": 0.5,
    "seed": 42,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
project_root_path = ".glaze"  # Ensure this directory exists
jpg = False

# Initialize Glaze
glaze = Glaze(params, device, target_params, project_root_path, jpg)

# Load and cloak an image
image = Image.open("your_image.png")
cloaked_image = glaze.cloak_image(image)

if cloaked_image:
    cloaked_image.save("cloaked_image.png")
    print("Image cloaked successfully!")
```


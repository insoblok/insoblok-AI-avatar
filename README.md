# insoblok-AI-avatar

## AI Avatar Generator

This repository contains a Python-based tool that transforms regular photos into stylized AI avatars using Stable Diffusion XL. The tool can create avatars in various artistic styles based on your input image and text prompts.

## Features

- Transform any photo into a stylized avatar
- Support for both local images and image URLs
- Customizable prompts to control avatar style
- Side-by-side comparison of original and generated images
- Support for GPU acceleration (CUDA)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/insoblok-AI-avatar.git
   cd insoblok-AI-avatar
   ```

2. Install the required dependencies:
   ```
   pip install torch diffusers pillow numpy matplotlib requests ipython
   ```

## Usage

### Running in Jupyter Notebook

Open the `avatar-generator.ipynb` file in Jupyter Notebook or JupyterLab and run the cells.

### Using the Avatar Generator

The main function to generate avatars is:

```python
create_avatar_from_image(
    image_path="path/to/image.jpg",  # Local path or URL
    prompt="your style description",  # E.g., "anime", "cartoon", "oil painting"
    output_path="output.png",        # Optional
    show_comparison=True             # Optional
)
```

### Example

```python
avatar = create_avatar_from_image(
    image_path="input_image.jpg",
    prompt="cartoon", 
    output_path="my_new_avatar.png"
)
```

## How It Works

1. **Image Loading**: The tool loads your input image (local file or URL)
2. **Preprocessing**: The image is resized while maintaining aspect ratio
3. **Avatar Generation**: The Stable Diffusion XL model transforms your image based on your text prompt
4. **Visualization**: A side-by-side comparison shows the original and generated images
5. **Saving**: You can save the resulting avatar to a file

## System Requirements

- Python 3.7+
- PyTorch
- For GPU acceleration: NVIDIA GPU with CUDA support (recommended for faster generation)

## Advanced Usage

You can fine-tune the avatar generation by modifying these parameters in the `generate_avatar` function:

- `strength`: Controls how much the model modifies the input image (0.0-1.0)
- `guidance_scale`: Controls how closely the model follows your prompt

## License

MIT License

Copyright (c) 2025 Insoblok

## Acknowledgments

This project uses the Stable Diffusion XL model from Stability AI.
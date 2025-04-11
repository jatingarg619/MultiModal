# SmolVLM2 Demo

This project demonstrates how to use the SmolVLM2-2.2B-Instruct model for image and video analysis. SmolVLM2 is a lightweight multimodal model that can process images, videos, and text inputs to generate text outputs.

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 5.2GB+ GPU RAM for video inference

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

The demo script (`demo.py`) provides examples of how to use SmolVLM2 for both image and video analysis.

### Running the Demo

```bash
python demo.py
```

By default, the script will:
1. Load the SmolVLM2 model
2. Process a sample image of a bee and generate a description
3. (Optional) Process a video file (needs to be uncommented and configured in the code)

### Customizing the Demo

To process your own images or videos:

1. For images: Modify the `image_url` variable in `main()` to point to your image
2. For videos: Uncomment the video processing section and set `video_path` to your video file

## Model Details

- Model: SmolVLM2-2.2B-Instruct
- Size: 2.2B parameters
- Memory Requirements: ~5.2GB GPU RAM
- Capabilities:
  - Image analysis and description
  - Video understanding
  - Visual question answering
  - Multi-image comparison # MultiModal

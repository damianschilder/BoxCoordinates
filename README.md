# SAM Lid Segmentation & Coordinate Extractor

## Overview

This project uses Meta AI's Segment Anything Model (SAM) to automatically identify and segment multiple "lids" or "trays" (up to 6, based on current configuration) within a single image. After identifying these lids, it displays them with an index, saves the annotated image, and prints the top-left (xmin, ymin) and bottom-right (xmax, ymax) bounding box coordinates for each identified lid to the console.

This script is designed for analyzing a single image at a time and is particularly useful for quickly extracting object boundaries and locations without manual annotation or model training for this specific image.

## Prerequisites

* **Python:** Python 3.8 - 3.11 recommended (script was developed with Python, ensure compatibility if using much newer versions like 3.12+ which might require specific library versions).
* **pip:** Python package installer (usually comes with Python).
* **Git:** For cloning the repository (if applicable) and installing SAM.
* **(Optional) NVIDIA GPU:** For significantly faster processing.
    * NVIDIA drivers installed.
    * CUDA toolkit compatible with your PyTorch version (PyTorch often bundles necessary CUDA runtime libraries).

## Setup Instructions

1.  **Clone the Repository (if you're sharing it on GitHub):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
    If you are just setting this up from a script file, you can skip this step and just ensure you have the script.

2.  **Create a Virtual Environment (Recommended):**
    This helps manage dependencies and avoid conflicts.
    ```bash
    python -m venv sam_env
    # On Windows
    sam_env\Scripts\activate
    # On macOS/Linux
    source sam_env/bin/activate
    ```

3.  **Install Dependencies:**
    The required Python libraries are listed in `requirements.txt`. Create a file named `requirements.txt` in your project's root directory with the following content:

    ```txt
    torch
    torchvision
    torchaudio
    opencv-python
    matplotlib
    numpy
    Pillow
    # For SAM, install directly from GitHub:
    # segment-anything  <- This line is a comment, install SAM as per command below
    ```

    Then, install the dependencies:
    ```bash
    pip install -r requirements.txt
    pip install git+[https://github.com/facebookresearch/segment-anything.git](https://github.com/facebookresearch/segment-anything.git)
    ```
    **Note:** Ensure your PyTorch installation includes CUDA support if you intend to use a GPU. If you encounter issues or the script defaults to CPU, visit the [PyTorch Get Started page](https://pytorch.org/get-started/locally/) to get the correct `pip install` command for your specific OS and CUDA version. For example:
    ```bash
    # Example for PyTorch with CUDA 11.8 (check official site for latest)
    # pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```

4.  **Download SAM Model Checkpoint:**
    * This script is configured to use the ViT-Base model (`vit_b`). Download the checkpoint file:
        * **`sam_vit_b_01ec64.pth`**: [Direct Link (from SAM GitHub)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
    * Place the downloaded `.pth` file (e.g., `sam_vit_b_01ec64.pth`) in the **same directory as your Python script**, or update the `SAM_CHECKPOINT_PATH` variable within the script to its correct location.
    * Other model checkpoints (e.g., `vit_l`, `vit_h`) can be found on the [SAM GitHub repository](https://github.com/facebookresearch/segment-anything#model-checkpoints). If you use a different model, update `MODEL_TYPE` and `SAM_CHECKPOINT_PATH` in the script.

5.  **Prepare Input Image:**
    * The script expects an input image located at `Original/output_image.jpg` relative to where the script is run.
    * Create a folder named `Original` in your project directory.
    * Place your target image (e.g., `output_image.jpg`) inside this `Original` folder.
    * You can change the `IMAGE_PATH` variable in the script if your image is located elsewhere or named differently.

## Running the Script

Once the setup is complete, you can run the script from your terminal (ensure your virtual environment is activated):

```bash
python your_script_name.py
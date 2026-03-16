# Coordinates Extractor using SAM

A desktop application that leverages Meta's Segment Anything Model (SAM) to automatically identify objects in an image and extract their bounding box coordinates.

![screenshot2](https://github.com/user-attachments/assets/c622615e-e02e-4037-a9d8-242b6272b181)

## About The Project

This tool was developed for a researcher at Utrecht University to automate the manual annotation of fungi container lids in images. This process is a necessary step for research involving the scanning of containers at time intervals to measure fungi growth via grey value analysis within specific coordinates.

The application provides a graphical user interface (GUI) for the Segment Anything Model, allowing users to load an image, configure filtering parameters, and extract machine-readable coordinates for detected objects. It isolates specific objects (referred to as "lids") based on size and shape.

The application is packaged as a standalone desktop executable, making it accessible to users without Python or command-line experience.

## Key Features

* **SAM Integration:** Implements the `vit_b` variant of Meta's Segment Anything Model for object segmentation.
* **Graphical Interface:** GUI developed with PySide6 for Windows environments.
* **Automatic Model Download:** Prompts automatic downloading of the required SAM model file (`.pth`) if not present locally.
* **Configurable Parameters:** Adjust filters for object size (min/max area) and aspect ratio to refine detection.
* **Visual Feedback:** Displays the analyzed image with segmentation masks and numbered bounding boxes.
* **Data Export:** Generates coordinates in a machine-readable format (`xmin,ymin,xmax,ymax`) alongside a detailed processing log.
* **Clipboard Integration:** Copy extracted coordinates directly to the clipboard.

## 🚀 Download

Download the latest Windows executable from the **[Releases page](https://github.com/damianschilder/BoxCoordinates/releases/latest)**.

The application is a single `.exe` file that requires no installation.

## How to Use

1. Download `CoordinatesExtractor.exe` from the [Releases page](https://github.com/damianschilder/BoxCoordinates/releases/latest).
2. Run the executable. On the first run, allow the application to download the required SAM model file (~350MB).
3. Configure the **Tuning Parameters**:
   * **Num Lids Target:** The expected number of objects.
   * **Min/Max Area Fraction:** Filters objects relative to the total image size.
   * **Min Aspect Ratio:** Filters objects based on shape geometry (1.0 represents a perfect square).
4. Click **"Select Image & Process"** and load an image file.
5. The annotated image will render on the left panel upon completion.
6. Extracted coordinates will populate in the right panels. Use the "Copy" button to export the coordinates.
7. To re-process the image with different parameters, adjust the values and click **"Retry with Current Params"**.

## Tips for Better Results

Parameter configuration directly impacts detection accuracy:

* **Num Lids Target:** Ensure this matches the exact number of expected objects.
* **SAM Pts/Side:** Controls the analysis density of the model. Increasing this value to `50` or `90` improves identification accuracy but increases processing time.

## Troubleshooting

* **Model Download Fails:** Run `CoordinatesExtractor.exe` as an administrator. Elevated permissions may be required to write the downloaded model file to the local directory.

## 🔧 Building from Source

**Prerequisites:**

* Python 3.9+
* Git

**Setup:**

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/damianschilder/BoxCoordinates.git](https://github.com/damianschilder/BoxCoordinates.git)
   cd BoxCoordinates

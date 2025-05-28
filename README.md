# Audit System with YOLOv8 Segmentation

This project uses [Ultralytics YOLOv8](https://docs.ultralytics.com/) to train, validate, and perform segmentation predictions on images.

## Requirements

- Python 3.8 or higher
- [pip](https://pip.pypa.io/en/stable/installation/)
- NVIDIA GPU (optional, but recommended for training)

## Installation

1. **Clone the repository** (if applicable):

   ```sh
   git clone <REPOSITORY_URL>
   cd <PROJECT_FOLDER>
   ```

2. **Create a virtual environment (optional, but recommended):**

   ```sh
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**

   ```sh
   pip install -r requirementst.txt
   ```

   If using a GPU, simply uncomment the `torch` line in `requirements.txt` to install PyTorch with CUDA support, or follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for more options.

## Data Structure

- `train_data/`: should contain the `data.yaml` file and training images organized according to the YOLO standard.
- `test_data/`: place test/prediction images here.

## Pre-trained Weights

- The project already includes `yolov8n-seg.pt` and `yolo11n.pt` files for use as a base for training or inference.
- After training, the best model will be saved at `runs/segment/train/weights/best.pt`.

## How to Use

Edit the `main.py` file to uncomment the desired function:

```python
# audit_model.train()      # To train the model
# audit_model.val()        # To validate the model
# audit_model.predict()    # To run predictions on test images
```

Run the script:

```sh
python main.py
```

### Training

Uncomment `audit_model.train()` to start training with the data in `train_data/`.

### Validation

Uncomment `audit_model.val()` to validate the model saved at `runs/segment/train/weights/best.pt`.

### Prediction

Uncomment `audit_model.predict()` to run predictions on images in `test_data/`. Results will be saved and displayed in the terminal.

## Notes

- Make sure the `data.yaml` file in `train_data/` is correctly configured according to the YOLO standard.
- Prediction results are saved in the `runs/segment/predict/` folder by default.

## References

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [Example data.yaml configuration](https://docs.ultralytics.com/datasets/segmentation/)
# Speedy Sign Sensing

Traffic sign detection using YOLOv8, OpenCV, and a trained model covering 28 classes. Includes a live webcam demo, image/video inference, and a Tkinter GUI app.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the project directory:

```sh
uv sync --locked
uv run python tsd.py
```

uv manages Python 3.13 and the project dependencies. The trained model is included at `models/best.pt`. The first setup requires an internet connection to download dependencies.

## Webcam demo

```sh
uv run python tsd.py
```

Allow camera access when prompted. Press `q` to exit. To use another camera:

```sh
uv run python tsd.py --camera 1
```

The webcam demo processes frames in grayscale and displays detections with confidence scores of at least 0.4.

## Images and videos

Pass your own image or video file:

```sh
uv run python inference.py --source path/to/image.jpg
uv run python inference.py --source path/to/video.mp4
uv run python inference.py --source path/to/image.jpg --show
```

Predictions are saved under `runs/detect/predict*`. A display window is opened only with `--show`, so image inference can also run without a desktop display.

Both demos use the CPU by default. Use `--device mps` if using an Apple Silicon Mac, or `--device 0` with a CUDA-enabled PyTorch installation on an NVIDIA GPU. Webcam speed depends on the hardware.

## Optional desktop interface

```sh
uv run python "tsd+gui.py"
```

This interface uses the default camera and CPU. It requires a desktop display and Python with Tk support. Check Tk with `uv run python -m tkinter`. If it's unavailable, use a Python 3.13 installation with Tk support or the OpenCV demo instead. Close the window to release the camera.

## Training

The training dataset is not included. Supply a YOLO-format dataset YAML with paths to your images and labels:

```sh
uv run python train.py --data path/to/data.yaml --device 0 --epochs 500
```

Training defaults to CPU and 500 epochs. The pretrained `yolov8n.pt` backbone downloads automatically if it is not already present. New checkpoints and training results are written under `runs/detect/train*`. The bundled demo model is kept separately.

## Project files

- `tsd.py`: OpenCV webcam demo.
- `inference.py`: image and video predictions.
- `tsd+gui.py`: Tkinter webcam interface.
- `train.py`: training and validation.
- `gui.py` and `calculator.py`: earlier interface experiments.

## Troubleshooting

- **Camera unavailable:** check system camera permissions, close other apps using the camera, or try `--camera 1`.
- **Missing model:** restore `models/best.pt` from the repository.
- **No desktop display:** run `inference.py` without `--show`.
- **Unavailable accelerator:** leave out `--device` to use the CPU.

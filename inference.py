import argparse
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect traffic signs in an image or video.')
    parser.add_argument('--source', type=Path, required=True, help='Image or video file')
    parser.add_argument('--device', default='cpu', help='cpu, mps, or a CUDA device such as 0')
    parser.add_argument('--show', action='store_true', help='Display predictions in a window')
    args = parser.parse_args()

    if not args.source.is_file():
        parser.error(f'Source file not found: {args.source}')

    root = Path(__file__).resolve().parent
    weights = root / 'models' / 'best.pt'
    if not weights.is_file():
        parser.error(f'Model weights not found: {weights}')

    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.predict(
        source=str(args.source), save=True, imgsz=1024,
        device=args.device, show=args.show,
        project=str(root / 'runs' / 'detect'), name='predict',
    )

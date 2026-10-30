import argparse
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the traffic sign detector.')
    parser.add_argument('--data', type=Path, required=True, help='YOLO dataset YAML file')
    parser.add_argument('--device', default='cpu', help='cpu, mps, or a CUDA device such as 0')
    parser.add_argument('--epochs', type=int, default=500)
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f'Dataset YAML not found: {args.data}')
    if args.epochs < 1:
        parser.error('--epochs must be positive')

    from ultralytics import YOLO

    root = Path(__file__).resolve().parent
    model = YOLO(str(root / 'yolov8n.pt'))
    results = model.train(
        data=str(args.data.resolve()), epochs=args.epochs, device=args.device,
        imgsz=1024, patience=0, project=str(root / 'runs' / 'detect'), name='train',
    )
    results = model.val()

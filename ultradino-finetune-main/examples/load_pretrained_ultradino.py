#!/usr/bin/env python3


from pathlib import Path


from ultradino_finetune.models import load_pretrained_ultradino


WEIGHTS_BASE_PATH = Path('/data/proto/CommonCode/UltraDINO/weights')
DEFAULT_MODEL_TYPE = 'vits16'
DEFAULT_WEIGHTS_PATH = WEIGHTS_BASE_PATH / 'vits16_dinus_13m/20241205-132822-468746/pretrain/model_final.rank_0.pth'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-type',
        metavar='TYPE',
        default=DEFAULT_MODEL_TYPE,
        choices=['vits16', 'vitb16'],
        help='Options: "%(choices)s", default=%(default)s',
    )
    parser.add_argument(
        '--weights-path',
        metavar='PATH',
        default=DEFAULT_WEIGHTS_PATH,
        type=Path,
        help='Path to the weights (default="%(default)s")',
    )
    args = parser.parse_args()

    model = load_pretrained_ultradino(args.model_type, args.weights_path)


if __name__ == '__main__':
    main()

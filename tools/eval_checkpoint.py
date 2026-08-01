"""Evaluate an explicit Jittor checkpoint with the configured val dataset."""
import argparse

import jittor as jt

from jdet.config import init_cfg
from jdet.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    jt.flags.use_cuda = 1
    init_cfg(args.config)
    runner = Runner()
    runner.load(args.checkpoint, model_only=True)
    runner.val()


if __name__ == '__main__':
    main()

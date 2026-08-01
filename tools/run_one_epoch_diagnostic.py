"""Resume a checkpoint and stop at an explicit epoch for diagnostics."""
import argparse

import jittor as jt

from jdet.config import init_cfg
from jdet.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--stop-epoch', required=True, type=int)
    args = parser.parse_args()
    jt.flags.use_cuda = 1
    init_cfg(args.config)
    runner = Runner()
    # Runner.load restores max_epoch from the source checkpoint.  Override it
    # after resume so this diagnostic cannot spill into another epoch.
    runner.max_epoch = args.stop_epoch
    runner.total_iter = runner.max_epoch * len(runner.train_dataset)
    runner.run()


if __name__ == '__main__':
    main()

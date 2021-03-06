'''
DNNAnnotator: CLI interface
'''

# built-in
import pdb
import os
import argparse

# external
import dsargparse

# customs
from .. import engine
from .. import data
from . import train, predict
from ..utils import dump
from ..utils import load


def main(prog='python3 -m segmentator.runs'):
    parser = dsargparse.ArgumentParser(main=main, prog=prog)
    subparsers = parser.add_subparsers(help='command')
    subparsers.add_parser(train.train, add_arguments_auto=True)
    subparsers.add_parser(predict.predict, add_arguments_auto=True)
    #subparsers.add_parser(evaluate.evaluate)
    # subparsers.add_parser(data.generate_tfrecords, add_arguments_auto=True)
    return parser.parse_and_run()


if __name__ == '__main__':
    main()

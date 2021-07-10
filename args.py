"""Contains settings that vary per run.

All global, immutable settings should be in settings.py.
"""

import argparse


def parse_args():
    """Defines command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type=str)
    parser.add_argument("--collect_data", required=True, type=int)
    parser.add_argument("--start_seed", required=True, type=int)
    parser.add_argument("--num_seeds", required=True, type=int)
    args = parser.parse_args()
    print_config(args)
    return args


def print_config(args):
    """Print all info for this experiment.
    """
    end = args.start_seed+args.num_seeds
    print(f"Seeds: {list(range(args.start_seed, end))}")
    print(f"Env: {args.env}")
    print()

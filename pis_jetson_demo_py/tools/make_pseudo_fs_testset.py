# Convert CSV files of video outputs into FS dataset JSON format
# Usage: python convert_fs_dataset.py <path_to_csv_file> <path_to_output_json_file>

import csv
import glob
import json
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "stats_dir", type=str, help="Directory path to read stats csv files"
    )

    parser.add_argument(
        "output_dir", type=str, help="Directory path to save output JSON file"
    )

    return parser.parse_args()


def main(args: Namespace) -> int:
    print(os.path.join(args.stats_dir, "**", "*.csv"))
    all_csv_files = glob.glob(
        os.path.join(args.stats_dir, "**", "*.csv"), recursive=True
    )

    for csv_filename in all_csv_files:
        csv_relpath = os.path.relpath(csv_filename, args.stats_dir)
        with open(csv_filename, "r", encoding="utf-8") as cf:
            reader = iter(csv.reader(cf))
            headers = next(reader)
            for frame_idx, *row in reader:
                frame_idx = int(frame_idx)
                print(row)
        break
    return 0


if __name__ == "__main__":
    args: Namespace = parse_args()
    sys.exit(main(args))

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from typing import List, Dict, Any
import csv
import sys


def parse_args() -> Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("csv1", type=str, help="CSV file 1")
    parser.add_argument("csv2", type=str, help="CSV file 2")

    return parser.parse_args()


def argsort(seq) -> List[int]:
    return sorted(range(len(seq)), key=seq.__getitem__)


def rename_header(header: str) -> str:
    if "distracted" in header:
        return header.replace("distracted", "distraction")
    if "no_wheel_grab" in header:
        return header.replace("no_wheel_grab", "negative_wheelgrab")
    return header


def read_sort_csv(csv_filepath: str) -> Dict[str, Any]:
    with open(csv_filepath, "r", encoding="utf-8") as f:
        reader = iter(csv.reader(f))

        # Remove first column (frame number)
        _, *header = next(reader)
        items = [row for (_, *row) in reader]

        # Rename header if applicable
        header = [rename_header(item) for item in header]

        # Filter unusable header
        if "negative_wheelgrab_passenger" in header:
            removal_idx = header.index("negative_wheelgrab_passenger")
            header = [item for (idx, item) in enumerate(header) if idx != removal_idx]
            items = [
                [value for (col_idx, value) in enumerate(row) if col_idx != removal_idx]
                for row in items
            ]

        # Reorder items by column name
        order = argsort(header)

        header = [header[i] for i in order]
        items = [[row[i] for i in order] for row in items]
        return header, items


def main(args: Namespace) -> int:
    csv1_header, csv1_items = read_sort_csv(args.csv1)
    csv2_header, csv2_items = read_sort_csv(args.csv2)

    assert len(csv1_items) == len(csv2_items), "Frame counts are not equal"
    assert csv1_header == csv2_header, "Headers are not equal"
    assert csv1_items == csv2_items, "Items are not equal"

    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))

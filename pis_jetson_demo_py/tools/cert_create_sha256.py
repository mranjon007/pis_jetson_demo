import glob
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from hashlib import sha256
from tqdm.auto import tqdm
import csv


def parse_args() -> Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "datadir",
        type=str,
        help="Test data directory to generate sha256sum",
    )
    return parser.parse_args()


def sha256sum(filename: str) -> str:
    sha256sum = sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256sum.update(byte_block)
    return sha256sum.hexdigest()


def main(args: Namespace) -> int:
    datadir = args.datadir
    for test_case in ["TC_A", "TC_B", "TC_C"]:
        test_case_dir = os.path.join(datadir, test_case)
        file_basenames = [
            os.path.splitext(os.path.basename(item_name))[0]
            for item_name in sorted(glob.glob(os.path.join(test_case_dir, "*.mp4")))
        ]

        # Sanity check
        for base_name in file_basenames:
            # base_name: fs_test_case_a_001

            mp4_filename = f"{base_name}.mp4"
            gt_json_filename = f"{base_name}_gt.json"
            pred_json_filename = f"{base_name}_pred.json"

            assert os.path.exists(
                os.path.join(test_case_dir, mp4_filename)
            ), f"File {mp4_filename} does not exist."
            assert os.path.exists(
                os.path.join(test_case_dir, gt_json_filename)
            ), f"File {gt_json_filename} does not exist."
            assert os.path.exists(
                os.path.join(test_case_dir, pred_json_filename)
            ), f"File {pred_json_filename} does not exist."

        # Calculate sha256sum
        sha256sums = {}
        for base_name in tqdm(file_basenames, desc="Generating sha256sums..."):
            # base_name: fs_test_case_a_001

            mp4_filename = f"{base_name}.mp4"
            gt_json_filename = f"{base_name}_gt.json"
            pred_json_filename = f"{base_name}_pred.json"

            sha256sums[base_name] = {
                "mp4": sha256sum(os.path.join(test_case_dir, mp4_filename)),
                "gt_json": sha256sum(os.path.join(test_case_dir, gt_json_filename)),
                "pred_json": sha256sum(os.path.join(test_case_dir, pred_json_filename)),
            }

        # Write into single csv file
        with open(f"{test_case.lower()}_sha256sums.csv", "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["test_case", "mp4", "gt_json", "pred_json"])
            for base_name in file_basenames:
                writer.writerow(
                    [
                        base_name,
                        sha256sums[base_name]["mp4"],
                        sha256sums[base_name]["gt_json"],
                        sha256sums[base_name]["pred_json"],
                    ]
                )

    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))

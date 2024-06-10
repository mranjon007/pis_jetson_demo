import json
import math
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from csv import writer
from datetime import datetime
from glob import glob
from typing import Dict, List

from loguru import logger


def parse_args() -> Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--tvpaths",
        nargs="+",
        default=["./TestData/TC_A", "./TestData/TC_B", "./TestData/TC_C"],
        help="List of location of directory including Task condition's video/label files",
    )

    return parser.parse_args()


def load_pred_results(test_video_dir: str) -> Dict[str, Dict]:
    pred_items = {}
    for pred_json in sorted(glob(os.path.join(test_video_dir, "*_pred.json"))):
        with open(pred_json, "r", encoding="utf-8") as f:
            pred_item = json.load(f)
        video_filename = os.path.basename(pred_item["metadata"]["filename"])
        video_filename = os.path.splitext(pred_item["metadata"]["filename"])[0]
        pred_items[video_filename] = (
            1000 / pred_item["performance_metrics"]["fps"]
        )  # Processing speed in ms
    return pred_items


def get_latency(pred_items) -> List[float]:
    return [pred_items[filename] for filename in pred_items.keys()]


def main(args: Namespace) -> int:
    test_video_dir_list = args.tvpaths

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("===================================")
    logger.info("      Beginning Test Case D        ")
    logger.info("- Date: {} ".format(current_date))
    logger.info("===================================")

    all_latencies = []
    all_pred_items = {}
    per_case_latencies = []
    for test_video_dir in test_video_dir_list:
        assert os.path.exists(test_video_dir), "Test video path does not exist"
        assert os.path.isdir(test_video_dir), "Test video path is not a directory"

        # Calculate latency of every Test Case
        pred_items = load_pred_results(test_video_dir)
        all_pred_items.update(pred_items)
        latencies = get_latency(pred_items)
        if len(pred_items) == 0:
            raise RuntimeError(f"Pred items in {test_video_dir} is empty")

        all_latencies.extend(latencies)
        per_case_latencies.append(latencies)

        per_task_avg_latency = sum(latencies) / len(latencies)
        logger.info(
            f"Folder {test_video_dir} Avg. latency: {per_task_avg_latency:.02f} ms"
        )

    avg_fps = 1000 / (sum(all_latencies) / len(all_latencies))
    logger.info(f"===== Prediction of {len(all_latencies):03d} items =====")
    logger.info(f"- Test iteration: 1")
    logger.info(f"- Avg. FPS: {avg_fps:.02f} ({1000/avg_fps:.02f} ms)")
    logger.info(f"===================================")

    with open("fs_test_case_d_pred.csv", "w", encoding="utf-8") as f:
        csv_writer = writer(f)
        csv_writer.writerow(["VideoName", "AvgLatency"])
        for video_name in sorted(all_pred_items.keys()):
            csv_writer.writerow(
                [video_name, math.floor(all_pred_items[video_name] * 100) / 100]
            )

    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))

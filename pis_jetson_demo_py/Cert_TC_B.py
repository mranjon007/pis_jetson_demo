import json
import os
import sys
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from datetime import datetime
from glob import glob
from typing import Dict, List, Tuple

from hydra import compose, initialize
from loguru import logger

G_TEST_ITERATION = 1


def parse_args() -> Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--tvpath",
        type=str,
        help="Location of directory including Task condition's video/label files",
        default="./TestData/TC_B",
    )

    parser.add_argument(
        "--platform",
        type=str,
        choices=["xavier", "pc"],
        default="xavier",
        help="Platform type to run certification",
    )

    parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="Skip inference stage and forward to accuracy check stage",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Headless mode (do not show output window)",
    )

    return parser.parse_args()


def load_gt_labels(test_video_dir: str) -> Dict[str, Dict]:
    gt_items = {}
    for gt_json in sorted(glob(os.path.join(test_video_dir, "*_gt.json"))):
        with open(gt_json, "r", encoding="utf-8") as f:
            gt_item = json.load(f)
        video_filename = os.path.basename(gt_item["metadata"]["filename"])
        video_filename = os.path.splitext(gt_item["metadata"]["filename"])[0]
        gt_items[video_filename] = gt_item["labels"]
    return gt_items


def load_pred_results(test_video_dir: str) -> Dict[str, Dict]:
    pred_items = {}
    for pred_json in sorted(glob(os.path.join(test_video_dir, "*_pred.json"))):
        with open(pred_json, "r", encoding="utf-8") as f:
            pred_item = json.load(f)
        video_filename = os.path.basename(pred_item["metadata"]["filename"])
        video_filename = os.path.splitext(pred_item["metadata"]["filename"])[0]
        pred_items[video_filename] = {
            "predictions": pred_item["predictions"],
            "performance_metrics": pred_item["performance_metrics"],
        }
    return pred_items


def calculate_accuracy(test_video_dir, gt_items) -> Tuple[List[bool], List[float]]:
    pred_items = load_pred_results(test_video_dir)

    # Calculate accuracy
    if len(gt_items) != len(pred_items):
        logger.warning(
            f"GT({len(gt_items)}) and Prediction({len(pred_items)}) result items differs!"
        )

    fps_metrics = []
    acc_metrics = []

    for key in sorted(gt_items.keys()):
        # key -> fs_test_case_*_{001...300}
        if key not in pred_items:
            logger.warning(f"Prediction result not found for {key}")
            continue

        predictions = pred_items[key]["predictions"]
        fps = pred_items[key]["performance_metrics"]["fps"]
        labels = gt_items[key]

        criteria_list = []
        # 각 label에 존재하는 task에 대해서만 정답을 계산
        for task_name in gt_items[key].keys():
            pred = predictions[task_name]
            label = labels[task_name]

            criteria_list.append(pred == label)

        # 모든 조건이 맞았을 경우에만 정답 (AND operation)
        final_criteria = sum(criteria_list) == len(criteria_list)
        acc_metrics.append(final_criteria)
        fps_metrics.append(fps)

    return acc_metrics, fps_metrics


def main(args: Namespace) -> int:
    global G_TEST_ITERATION

    test_video_dir = args.tvpath
    headless = args.headless
    platform = args.platform
    skip_infer = args.skip_infer
    assert os.path.exists(test_video_dir), "Test video path does not exist"
    assert os.path.isdir(test_video_dir), "Test video path is not a directory"

    # Find and load GTs for each events
    gt_items = load_gt_labels(test_video_dir)

    # Load base configuration for event split
    initialize(config_path="./recipes", job_name="certification", version_base=None)

    if platform == "xavier":
        config_name = "certificate_TC_B.xavier.yaml"
    else:
        config_name = "certificate_TC_B.yaml"

    config = compose(config_name=config_name)

    # Define event split callback
    G_TEST_ITERATION = 1  # will be incremented inside event callback

    def _event_callback(
        video_input_filepath: str,
        output_basedir: str,
        video_save_filename: str,
        temporary_video_path: str,
        cumulative_events: Dict[str, Dict[str, bool]],
        all_events: List[Dict[int, Dict[str, bool]]],
        events_per_passengers: Dict[str, List[str]],
        all_latency: int,
        save_stats: bool = True,
    ):
        global G_TEST_ITERATION
        video_filename = os.path.basename(video_input_filepath)

        # We only need cumulative_events for each video's result
        p_events = cumulative_events["passenger"]
        d_events = cumulative_events["driver"]

        video_basename = os.path.splitext(video_filename)[0]
        # if video_basename not in gt_items:
        #     logger.warning(f"GT not found for {video_filename}")

        fs_tc_events = {
            "metadata": {
                "filename": video_filename,  # fs_test_case_a_{001...300}.mp4
                "id": G_TEST_ITERATION,
                "type": "status",  # onboard, status, action
            },
            "predictions": {
                "phone_answer": d_events["phone_answer"] or p_events["phone_answer"],
                "smoke": d_events["smoke"] or p_events["smoke"],
                "beltoff": d_events["beltoff"] or p_events["beltoff"],
                "drink": d_events["drink"] or p_events["drink"],
            },
            "performance_metrics": {
                "total_frames": len(all_events),
                "latency_ms": all_latency,
                "fps": 1000 / (all_latency / len(all_events)),
            },
        }

        # Save JSON based on video filename
        # Must match fs_test_case_?_{001...300}_pred.json
        pred_json_name = os.path.splitext(video_filename)[0] + "_pred.json"
        pred_json_path = os.path.join(test_video_dir, pred_json_name)
        with open(pred_json_path, "w", encoding="utf-8") as f:
            json.dump(fs_tc_events, f, indent=2)

        G_TEST_ITERATION += 1
        # End of event split callback

    # Run event split with predefined configuration and callback
    config.recipe.pipeline.source.url = test_video_dir
    if headless:
        output_type = "disable"
    else:
        output_type = "x11"
        if "DISPLAY" in os.environ:
            config.recipe.pipeline.sink.url = os.environ["DISPLAY"]

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("===================================")
    logger.info("      Beginning Test Case B        ")
    logger.info("- Date: {} ".format(current_date))
    logger.info("===================================")
    begin_time = time.time()

    if not skip_infer:
        # Lazy load (due to TensorRT engine initialization)
        from utils.demo.base import run_event_split

        if (
            run_event_split(
                config=config,
                save_video_fn=_event_callback,
                output_type=output_type,
            )
            != 0
        ):
            logger.error("Pipeline failed.")
            return 1

    end_time = time.time()
    time_spent_secs = end_time - begin_time

    # Calculate accuracy of this Test Case
    acc_metrics, fps_metrics = calculate_accuracy(test_video_dir, gt_items)

    accuracy = sum(acc_metrics) / len(acc_metrics)
    avg_fps = sum(fps_metrics) / len(fps_metrics)

    logger.info(f"===== Prediction of {len(acc_metrics):03d} items =====")
    logger.info(f"- Time spent: {time_spent_secs:.02f} seconds")
    logger.info(f"- Test iteration: 1")
    logger.info(f"- Accuracy: {accuracy * 100:.02f}%")
    logger.info(f"- Avg. FPS: {avg_fps:.02f} ({1000/avg_fps:.02f} ms)")
    logger.info(f"===================================")

    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))

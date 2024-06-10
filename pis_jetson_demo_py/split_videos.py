"""Split videos by event types

Choose videos when matching following criteria:
1. Event A occurs
2. Event B occurs
...

Criteria can be manually implemented within this code.
"""
import csv
import os
import shutil
import sys
from typing import Dict, List

import hydra
from loguru import logger
from omegaconf import DictConfig

from base.types import EVENT_TYPES
from utils.demo.base import run_event_split


def save_video_cb(
    video_input_filepath: str,
    output_basedir: str,
    video_save_filename: str,
    temporary_video_path: str,
    cumulative_events: Dict[str, Dict[str, bool]],
    all_events: List[Dict[int, Dict[str, bool]]],
    events_per_passengers: Dict[str, List[str]],
    all_latency: int,
    save_stats: bool = True,
) -> None:
    """Callback function to save (original/result) video to arbitary folder
    based on event criteria (e.g. outputs/phone_on/{VIDEO_FILENAME}.mp4)

    See:
        utils/demo/video_split_pipeline.py:L67 (run_video_split_pipeline())

    Args:
        video_input_filepath (str): Input video full filepath(In a/b/c/d/e.mp4 format).
        output_basedir (str): Output video should be stored inside this folder.
        video_save_filename (str): Rename output video with given filename (and path included).
        temporary_video_path (str): Temporary video path which is a pipeline's drawn output.
                                    (Note that this file and directory will be cleared once application exits.)
        cumulative_events (Dict[str, Dict[str, bool]]): Event information generated with single video pipeline.
    """
    passenger_events = cumulative_events["passenger"]
    driver_events = cumulative_events["driver"]

    passenger_event_types = events_per_passengers["passenger"]
    driver_event_types = events_per_passengers["driver"]

    # Aggregate all event stats of single video file and save into single CSV file (stats)
    if save_stats:
        save_basedir = os.path.join(
            output_basedir,
            "stats",
            os.path.dirname(
                video_save_filename
            ),  # Also add path component of save filename
        )
        os.makedirs(save_basedir, exist_ok=True)
        save_path = os.path.join(
            save_basedir,
            os.path.basename(
                video_save_filename
            ),  # Strip path component of save filename
        )

        csv_save_path = os.path.splitext(save_path)[0] + ".csv"
        all_criteria_names = sum(
            [
                [
                    f"{criteria_name}_passenger"
                    for criteria_name in passenger_event_types
                ],
                [f"{criteria_name}_driver" for criteria_name in driver_event_types],
            ],
            [],
        )  # P, P, P, ..., D, D, D, ...

        with open(csv_save_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(
                [
                    "frame_idx",
                    *all_criteria_names,
                ]
            )

            for frame_idx, event in enumerate(all_events):
                single_frame_event_stats = []

                for criteria_name in passenger_event_types:
                    single_frame_event_stats.append(
                        "1" if event["passenger"][criteria_name] else ""
                    )

                for criteria_name in driver_event_types:
                    single_frame_event_stats.append(
                        "1" if event["driver"][criteria_name] else ""
                    )
                # P, P, P, ..., D, D, D, ...
                writer.writerow([frame_idx, *single_frame_event_stats])

    # None of events are occured
    if True not in sum(
        [list(passenger_events.values()), list(driver_events.values())], []
    ):
        save_basedir = os.path.join(
            output_basedir,
            "no_event",
            os.path.dirname(
                video_save_filename
            ),  # Also add path component of save filename
        )
        os.makedirs(save_basedir, exist_ok=True)
        save_path = os.path.join(
            save_basedir,
            os.path.basename(
                video_save_filename
            ),  # Strip path component of save filename
        )

        shutil.copy(temporary_video_path, save_path)

        logger.info(
            f"[SplitVideos] No event with video {os.path.basename(video_save_filename)}, copying to {save_path}"
        )
        return

    # Save every class occurences into separate folders
    all_event_types = list(set(passenger_event_types + driver_event_types))
    all_event_types.sort()

    for event_name in all_event_types:
        # passenger_criteria, driver_criteria:
        #     only True if <criteria_name> event occurs.
        #     and False if <criteria_name> event didn't occured.
        passenger_criteria = None
        driver_criteria = None

        if event_name in passenger_event_types:
            passenger_criteria = passenger_events[event_name]
        if event_name in driver_event_types:
            driver_criteria = driver_events[event_name]

        all_criteria = {}
        if passenger_criteria is not None:
            all_criteria["passenger"] = passenger_criteria
        if driver_criteria is not None:
            all_criteria["driver"] = driver_criteria

        for passenger_type, criteria in all_criteria.items():
            if criteria:
                save_basedir = os.path.join(
                    output_basedir,
                    f"{event_name}_{passenger_type}",
                    os.path.dirname(
                        video_save_filename
                    ),  # Also add path component of save filename
                )

                os.makedirs(save_basedir, exist_ok=True)
                save_path = os.path.join(
                    save_basedir,
                    os.path.basename(
                        video_save_filename
                    ),  # Strip path component of save filename
                )

                shutil.copy(temporary_video_path, save_path)

                # Write detailed CSV about per-frame event information
                csv_save_path = os.path.splitext(save_path)[0] + ".csv"
                with open(csv_save_path, "w", newline="", encoding="utf-8") as cf:
                    writer = csv.writer(cf)
                    writer.writerow(
                        [
                            "frame_idx",
                            f"{event_name}_{passenger_type}",
                        ]
                    )
                    for frame_idx, event in enumerate(all_events):
                        writer.writerow(
                            [
                                frame_idx,
                                "1" if event[passenger_type][event_name] else "",
                            ]
                        )

                logger.info(
                    f'[SplitVideos] {passenger_type} event "{event_name}" of {os.path.basename(video_save_filename)} saved'
                )

        if not sum(all_criteria.values()):  # only 0 if everything is False
            save_basedir = os.path.join(
                output_basedir,
                f"{event_name}_negative",
                os.path.dirname(
                    video_save_filename
                ),  # Also add path component of save filename
            )

            os.makedirs(save_basedir, exist_ok=True)
            save_path = os.path.join(
                save_basedir,
                os.path.basename(
                    video_save_filename
                ),  # Strip path component of save filename
            )

            shutil.copy(temporary_video_path, save_path)
            logger.info(
                f'[SplitVideos] Negative (None of) event "{event_name}" of {os.path.basename(video_save_filename)} saved'
            )


@hydra.main(version_base=None, config_path="recipes")
def main(config: DictConfig) -> int:
    return run_event_split(config, save_video_cb, output_type="default")


if __name__ == "__main__":
    script_name = sys.argv[0]

    if "--config-name" not in sys.argv and "--config-path" not in sys.argv:
        logger.error(
            "Recipe (Hydra config file) mustbe specified with --config-name <recipe_name>"
        )
        logger.error(
            f"Usage: python3 {script_name} [--config-name <recipe_name>] [--config-path <recipe_yaml_path>] [KEY=VALUE, [KEY=VALUE ...]]"
        )
        logger.error(
            f"(Example: python3 {script_name} --config-path recipes/pms2.testvideo.yaml recipe.general.verbose=True)"
        )
        sys.exit(1)
    sys.exit(main())  # pylint: disable=E1120

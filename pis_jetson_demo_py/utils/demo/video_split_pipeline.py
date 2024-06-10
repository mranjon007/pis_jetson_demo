import glob
import os
import tempfile
from typing import Any, Callable, Dict, List

from loguru import logger
from tqdm.auto import tqdm

from utils.urlchecker import URLType

from .video_single_pipeline import run_single_video_pipeline

G_DECODER_FPS_LIMIT: int = 90
G_VIDEO_FILE_EXT: set = set([".mp4", ".mkv", ".avi", ".mov"])


def filter_video_file(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in G_VIDEO_FILE_EXT


def run_video_split_pipeline(
    pipeline_config: Dict[str, Any],
    save_video_fn: Callable[
        [
            str,
            str,
            str,
            str,
            Dict[str, Dict[str, bool]],
            List[Dict[str, Dict[str, bool]]],
        ],
        None,
    ],
) -> int:
    input_url: str = pipeline_config["input_url"]
    input_url_type: URLType = pipeline_config["input_url_type"]
    output_basedir: str = pipeline_config["output_url"]
    output_url_type: URLType = pipeline_config["output_url_type"]
    output_type: str = pipeline_config["output_type"]

    video_filelist = list(
        sorted(
            filter(
                filter_video_file,
                glob.glob(os.path.join(input_url, "**", "*.*"), recursive=True),
            )
        )
    )

    with tempfile.TemporaryDirectory() as tempdir:
        pbar = tqdm(video_filelist, desc="Running video pipeline ...")
        for video_idx, single_video_url in enumerate(pbar):
            video_filename = os.path.basename(single_video_url)
            pbar.set_description(f"Running video pipeline ({video_filename}) ...")

            input_video_reldir = os.path.relpath(
                os.path.dirname(single_video_url), input_url
            )
            basename_with_ext = os.path.basename(single_video_url)

            new_output_url = os.path.join(input_video_reldir, basename_with_ext)
            temporary_video_path = os.path.join(tempdir, basename_with_ext)

            prev_pipeline_url = pipeline_config["output_url"]

            pipeline_config.update(
                {
                    "input_url": single_video_url,
                    "input_url_type": URLType.FILE,
                    "output_url": temporary_video_path,
                    "output_url_type": URLType.FILE,
                    "multi_video_file_pipeline": True,  # Exception handling + Logging
                }
            )

            output_type = output_type.lower()
            if output_type == "disable":
                pipeline_config.update(
                    {
                        "output_url": None,
                        "output_url_type": URLType.NONE,
                    }
                )
            elif output_type == "x11":
                pipeline_config.update(
                    {
                        "output_url": prev_pipeline_url,
                        "output_url_type": URLType.X11,
                    }
                )
            elif output_type == "default":
                pass  # Default overriden configuration

            input_video_relpath = os.path.join(input_video_reldir, basename_with_ext)
            try:
                (
                    cumulative_events,
                    all_events,
                    events_per_passengers,
                    lat_all_without_vis,
                ) = run_single_video_pipeline(pipeline_config)

                # Save video based on event information
                save_video_fn(
                    video_input_filepath=single_video_url,
                    output_basedir=output_basedir,
                    video_save_filename=new_output_url,
                    temporary_video_path=temporary_video_path,
                    cumulative_events=cumulative_events,
                    all_events=all_events,
                    events_per_passengers=events_per_passengers,
                    all_latency=lat_all_without_vis,
                )
            except (KeyboardInterrupt, SystemExit):
                logger.warning(
                    f"Interrupted while processing file (video file metadata is safe): {input_video_relpath}"
                )
                return 1

            logger.debug(
                f"[{video_idx+1:04d}/{len(video_filelist):04d}] Done processing file: {input_video_relpath}"
            )

    return 0

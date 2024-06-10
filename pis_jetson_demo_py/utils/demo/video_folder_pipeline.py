import glob
import os
from typing import Any, Dict

from loguru import logger

from utils.urlchecker import URLType

from .video_single_pipeline import run_single_video_pipeline

G_DECODER_FPS_LIMIT: int = 90
G_VIDEO_FILE_EXT: set = set([".mp4", ".mkv", ".avi", ".mov"])


def filter_video_file(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in G_VIDEO_FILE_EXT


def run_video_folder_pipeline(
    pipeline_config: Dict[str, Any],
) -> int:
    input_url: str = pipeline_config["input_url"]
    input_url_type: URLType = pipeline_config["input_url_type"]
    output_basedir: str = pipeline_config["output_url"]
    output_url_type: URLType = pipeline_config["output_url_type"]

    if input_url_type is URLType.VIDEO_FOLDER:
        if output_url_type not in [URLType.VIDEO_FOLDER, URLType.X11]:
            logger.error("Input URL and output URL must be both folder.")
            return 1

    video_filelist = list(
        filter(
            filter_video_file,
            glob.glob(os.path.join(input_url, "**", "*.*"), recursive=True),
        )
    )

    for video_idx, single_video_url in enumerate(sorted(video_filelist)):
        input_video_reldir = os.path.relpath(
            os.path.dirname(single_video_url), os.path.dirname(input_url)
        )
        basename_with_ext = os.path.basename(single_video_url)
        basename_without_ext = os.path.splitext(basename_with_ext)[0]

        output_dir = os.path.join(output_basedir, input_video_reldir)
        if output_url_type != URLType.X11:
            os.makedirs(output_dir, exist_ok=True)

        new_output_url = os.path.join(output_dir, basename_without_ext + ".mp4")
        pipeline_config.update(
            {
                "input_url": single_video_url,
                "input_url_type": URLType.FILE,
                "output_url": new_output_url,
                "output_url_type": URLType.FILE,
                "multi_video_file_pipeline": True,  # Exception handling + Logging
            }
        )

        if output_url_type == URLType.X11:
            x11_display_id = output_basedir
            pipeline_config.update(
                {
                    "output_url": x11_display_id,
                    "output_url_type": URLType.X11,
                    "multi_video_file_pipeline": True,  # Exception handling + Logging
                }
            )

        input_video_relpath = os.path.join(input_video_reldir, basename_with_ext)
        logger.debug(
            f"[{video_idx+1:04d}/{len(video_filelist):04d}] Video pipeline begin: {input_video_relpath}"
        )
        try:
            run_single_video_pipeline(pipeline_config)
        except (KeyboardInterrupt, SystemExit):
            logger.warning(
                f"Interrupted while processing file (video file metadata is safe): {input_video_relpath}"
            )
            break

        logger.debug(
            f"[{video_idx+1:04d}/{len(video_filelist):04d}] Done processing file: {input_video_relpath}"
        )

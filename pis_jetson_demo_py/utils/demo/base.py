# This file is not a standalone.
# You should provide configuration via hydra to run this demo.
#
# @author An Jung-In <jian@fssolution.co.kr>


import sys
from typing import Callable, Dict

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue

from engine.core.detections.classes import ClassNamesManager
from engine.processor import SBPProcessor, SixDRepNetProcessor, YoloNASProcessor
from engine.trt import TRTInfer
from engine.vcap import VideoSource
from utils.draw import DrawOptions
from utils.urlchecker import URLType, get_url_type

from .video_folder_pipeline import run_video_folder_pipeline
from .video_single_pipeline import run_single_video_pipeline
from .video_split_pipeline import run_video_split_pipeline


def run_demo(config: DictConfig) -> int:
    # Parse configuration recipes
    try:
        assert config.metaVersion >= 2
        recipe_engine: DictConfig = config.recipe.engine
        recipe_pipeline: DictConfig = config.recipe.pipeline
        recipe_general: DictConfig = config.recipe.general

        verbose: bool = recipe_general.verbose
        if not verbose:
            logger.remove()
            logger.add(sys.stdout, level="INFO")

        embedded_parser: bool = recipe_engine.detector.postprocess.use_embedded_parser
        model_type = "with_parser" if embedded_parser else "without_parser"
        od_engine_path: str = recipe_engine.detector.models[model_type]

        tracker_engine_type: str = recipe_engine.tracker.type
        tracker_thresh: float = recipe_engine.tracker.options.threshold
        tracker_ttm: int = recipe_engine.tracker.options.time_to_match
        tracker_ttl: int = recipe_engine.tracker.options.time_to_live

        embedded_parser: bool = recipe_engine.headpose.postprocess.use_embedded_parser
        model_type = "with_parser" if embedded_parser else "without_parser"
        head_pose_engine_path: str = recipe_engine.headpose.models[model_type]

        human_pose_engine_path: str = recipe_engine.humanpose.models.humanpose

        od_padded_resize: bool = recipe_engine.detector.preprocess.padded_resize
        od_input_size = (
            recipe_engine.detector.preprocess.input_height,
            recipe_engine.detector.preprocess.input_width,
        )
        od_conf_thresh: float = recipe_engine.detector.postprocess.conf_thresh
        od_nms_thresh: float = recipe_engine.detector.postprocess.nms_iou_thresh
        od_class_separated_nms: bool = (
            recipe_engine.detector.postprocess.class_separated_nms
        )

        od_all_classes = [
            item.lower() for item in recipe_engine.detector.postprocess.classes
        ]
        human_pose_padded_resize: bool = (
            recipe_engine.humanpose.preprocess.padded_resize
        )
        human_pose_conf_thresh: float = recipe_engine.humanpose.postprocess.conf_thresh
        human_pose_input_size = (
            recipe_engine.humanpose.preprocess.input_height,
            recipe_engine.humanpose.preprocess.input_width,
        )

        assert recipe_pipeline.source.type in [
            "camera",
            "camera_ov2311",
            "live",
            "file",
            "video_folder",
        ]
        input_url_type: URLType = get_url_type(recipe_pipeline.source.type)
        input_url: str = recipe_pipeline.source.url

        assert recipe_pipeline.sink.type in ["x11", "rtsp", "file", "video_folder"]
        output_url_type: URLType = get_url_type(recipe_pipeline.sink.type)
        output_url: str = recipe_pipeline.sink.url

        assert recipe_pipeline.sink.options.engine in ["default", "gstreamer"]
        output_gst: bool = recipe_pipeline.sink.options.engine == "gstreamer"

        sync: bool = recipe_pipeline.source.options.sync
        loop_source: bool = recipe_pipeline.source.options.loop
        force_resize: bool = recipe_pipeline.source.options.force_resize

        sequential: bool = recipe_pipeline.source.options.sequential
        draw_option_overrides: Dict[str, bool] = dict(recipe_pipeline.draw_options)
    except MissingMandatoryValue as error:
        logger.error("Recipe is missing required values.")
        logger.error(error.msg)
        return 1

    # Initialize variables
    draw_options = DrawOptions(
        # Default draw options
        **DrawOptions.DEFAULT_OPTIONS_SET
    ).strict_update(
        {"untracked_items": tracker_engine_type == "none", **draw_option_overrides}
    )
    logger.debug(f"DrawOptions: {draw_options}")

    # Initialize event params
    event_params = OmegaConf.to_container(config.event_params)

    # Create engine (will not create tracker)
    od_engine = TRTInfer(engine_path=od_engine_path)
    head_pose_engine = TRTInfer(engine_path=head_pose_engine_path)
    human_pose_engine = TRTInfer(engine_path=human_pose_engine_path)

    # Configure global pre/postprocessors
    YoloNASProcessor.get_instance().target_size = od_input_size
    YoloNASProcessor.get_instance().conf_thresh = od_conf_thresh
    YoloNASProcessor.get_instance().nms_thresh = od_nms_thresh
    YoloNASProcessor.get_instance().padded_resize = od_padded_resize
    YoloNASProcessor.get_instance().class_separated_nms = od_class_separated_nms

    SBPProcessor.get_instance().target_size = human_pose_input_size
    SBPProcessor.get_instance().padded_resize = human_pose_padded_resize
    SBPProcessor.get_instance().conf_thresh = human_pose_conf_thresh

    ClassNamesManager.create_instance(class_names=od_all_classes)

    logger.info("Warming up ...")
    YoloNASProcessor.get_instance().warmup_engine(od_engine)
    SixDRepNetProcessor.get_instance().warmup_engine(head_pose_engine)
    SBPProcessor.get_instance().warmup_engine(human_pose_engine)
    VideoSource.set_is_ov2311(recipe_pipeline.source.type == "camera_ov2311")

    pipeline_config = {
        "od_engine": od_engine,
        "head_pose_engine": head_pose_engine,
        "human_pose_engine": human_pose_engine,
        "input_url": input_url,
        "input_url_type": input_url_type,
        "output_url": output_url,
        "output_gst": output_gst,
        "force_resize": force_resize,
        "output_url_type": output_url_type,
        "tracker_engine_type": tracker_engine_type,
        "tracker_thresh": tracker_thresh,
        "tracker_ttm": tracker_ttm,
        "tracker_ttl": tracker_ttl,
        "sequential": sequential,
        "loop_source": loop_source,
        "sync": sync,
        "draw_options": draw_options,
        "event_params": event_params,
        "multi_video_file_pipeline": False,
        "profile": False,
    }

    if input_url_type == URLType.VIDEO_FOLDER:
        run_video_folder_pipeline(pipeline_config=pipeline_config)
    else:
        run_single_video_pipeline(pipeline_config=pipeline_config)

    logger.info("Teardown ...")
    # on_exit_cleanup() will be automatically run

    return 0


def run_event_split(
    config: DictConfig,
    save_video_fn: Callable,
    output_type: str,
    profile: bool = False,
) -> int:
    # Parse configuration recipes
    try:
        assert config.metaVersion >= 2
        recipe_engine: DictConfig = config.recipe.engine
        recipe_pipeline: DictConfig = config.recipe.pipeline
        recipe_general: DictConfig = config.recipe.general
        event_params: dict = OmegaConf.to_container(config.event_params)

        verbose: bool = recipe_general.verbose
        if not verbose:
            logger.remove()
            logger.add(sys.stdout, level="INFO")

        embedded_parser: bool = recipe_engine.detector.postprocess.use_embedded_parser
        model_type = "with_parser" if embedded_parser else "without_parser"
        od_engine_path: str = recipe_engine.detector.models[model_type]

        tracker_engine_type: str = recipe_engine.tracker.type
        tracker_thresh: float = recipe_engine.tracker.options.threshold
        tracker_ttm: int = recipe_engine.tracker.options.time_to_match
        tracker_ttl: int = recipe_engine.tracker.options.time_to_live

        embedded_parser: bool = recipe_engine.headpose.postprocess.use_embedded_parser
        model_type = "with_parser" if embedded_parser else "without_parser"
        head_pose_engine_path: str = recipe_engine.headpose.models[model_type]

        human_pose_engine_path: str = recipe_engine.humanpose.models.humanpose

        od_padded_resize: bool = recipe_engine.detector.preprocess.padded_resize
        od_input_size = (
            recipe_engine.detector.preprocess.input_height,
            recipe_engine.detector.preprocess.input_width,
        )
        od_conf_thresh: float = recipe_engine.detector.postprocess.conf_thresh
        od_nms_thresh: float = recipe_engine.detector.postprocess.nms_iou_thresh
        od_class_separated_nms: bool = (
            recipe_engine.detector.postprocess.class_separated_nms
        )

        od_all_classes = [
            item.lower() for item in recipe_engine.detector.postprocess.classes
        ]
        human_pose_padded_resize: bool = (
            recipe_engine.humanpose.preprocess.padded_resize
        )
        human_pose_conf_thresh: float = recipe_engine.humanpose.postprocess.conf_thresh
        human_pose_input_size = (
            recipe_engine.humanpose.preprocess.input_height,
            recipe_engine.humanpose.preprocess.input_width,
        )

        assert recipe_pipeline.source.type in [
            "camera",
            "camera_ov2311",
            "live",
            "file",
            "video_folder",
        ]
        input_url_type: URLType = get_url_type(recipe_pipeline.source.type)
        input_url: str = recipe_pipeline.source.url

        assert recipe_pipeline.sink.type in ["x11", "rtsp", "file", "video_folder"]
        output_url_type: URLType = get_url_type(recipe_pipeline.sink.type)
        output_url: str = recipe_pipeline.sink.url

        assert recipe_pipeline.sink.options.engine in ["default", "gstreamer"]
        output_gst: bool = recipe_pipeline.sink.options.engine == "gstreamer"

        sync: bool = recipe_pipeline.source.options.sync
        loop_source: bool = recipe_pipeline.source.options.loop
        force_resize: bool = recipe_pipeline.source.options.force_resize

        sequential: bool = recipe_pipeline.source.options.sequential
        draw_option_overrides: Dict[str, bool] = dict(recipe_pipeline.draw_options)
    except MissingMandatoryValue as error:
        logger.error("Recipe is missing required values.")
        logger.error(error.msg)
        return 1

    if (
        input_url_type == URLType.VIDEO_FOLDER
        and output_url_type == URLType.VIDEO_FOLDER
    ):
        # Normal case
        pass
    elif input_url_type == URLType.VIDEO_FOLDER and output_url_type == URLType.X11:
        # Visualization for events
        pass
    else:
        logger.error(
            "Given input and output must be either VIDEO_FOLDER or VIDEO_FOLDER+X11."
        )
        return 1

    # Initialize variables
    draw_options = DrawOptions(
        # Default draw options
        **DrawOptions.DEFAULT_OPTIONS_SET
    ).strict_update(
        {
            "untracked_items": tracker_engine_type == "none",
            **draw_option_overrides,
        }
    )
    logger.debug(f"DrawOptions: {draw_options}")

    # Create engine (will not create tracker)
    od_engine = TRTInfer(engine_path=od_engine_path)
    head_pose_engine = TRTInfer(engine_path=head_pose_engine_path)
    human_pose_engine = TRTInfer(engine_path=human_pose_engine_path)

    # Configure global pre/postprocessors
    YoloNASProcessor.get_instance().target_size = od_input_size
    YoloNASProcessor.get_instance().conf_thresh = od_conf_thresh
    YoloNASProcessor.get_instance().nms_thresh = od_nms_thresh
    YoloNASProcessor.get_instance().padded_resize = od_padded_resize
    YoloNASProcessor.get_instance().class_separated_nms = od_class_separated_nms

    SBPProcessor.get_instance().target_size = human_pose_input_size
    SBPProcessor.get_instance().padded_resize = human_pose_padded_resize
    SBPProcessor.get_instance().conf_thresh = human_pose_conf_thresh

    ClassNamesManager.create_instance(class_names=od_all_classes)

    logger.info("Warming up ...")
    YoloNASProcessor.get_instance().warmup_engine(od_engine)
    SixDRepNetProcessor.get_instance().warmup_engine(head_pose_engine)
    SBPProcessor.get_instance().warmup_engine(human_pose_engine)

    pipeline_config = {
        "od_engine": od_engine,
        "head_pose_engine": head_pose_engine,
        "human_pose_engine": human_pose_engine,
        "input_url": input_url,
        "input_url_type": input_url_type,
        "output_url": output_url,
        "output_gst": output_gst,
        "force_resize": force_resize,
        "output_url_type": output_url_type,
        "tracker_engine_type": tracker_engine_type,
        "tracker_thresh": tracker_thresh,
        "tracker_ttm": tracker_ttm,
        "tracker_ttl": tracker_ttl,
        "sequential": sequential,
        "loop_source": loop_source,
        "sync": sync,
        "draw_options": draw_options,
        "event_params": event_params,
        "multi_video_file_pipeline": False,
        "output_type": output_type,
        "profile": profile,
    }

    return run_video_split_pipeline(
        pipeline_config=pipeline_config,
        save_video_fn=save_video_fn,
    )

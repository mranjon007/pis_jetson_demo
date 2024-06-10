import os
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from loguru import logger

from engine.core.blob_cropper import BlobCropExpandType, BlobCropper
from engine.core.detections import (
    DetectionItem,
    HeadPoseDetectionItem,
    HumanPoseDetectionItem,
    parse_od_predictions,
)
from engine.core.detections.classes import ClassNamesManager
from engine.core.detections.humanpose import PassengerType
from engine.core.groupping import divide_passenger_single_
from engine.processor import (
    HeadPoseCalibrationProcessor,
    SBPProcessor,
    SixDRepNetProcessor,
    YoloNASProcessor,
)
from engine.trt import TRTInfer
from engine.vcap import SequentialVideoCapture, ThreadedVideoCapture, VideoSource
from event.base import BaseEvent
from event.distraction_event import get_distraction_event
from event.util import EventSerializer, create_event_processors
from tracker import BaseTracker, get_tracker_engine
from utils.draw import DrawOptions, draw_boxes_and_tracks, draw_info_text
from utils.latency import MeasureLatency
from utils.urlchecker import URLType
from utils.writer import DummyWriter

G_DECODER_FPS_LIMIT: int = 90
G_CV2_WINDOW_ALREADY_RESIZED: bool = False

EventStatus = Dict[str, Dict[str, bool]]


def run_single_video_pipeline(
    pipeline_config: Dict[str, Any],
) -> Tuple[EventStatus, List[EventStatus], Dict[str, List[str]], float]:
    global G_CV2_WINDOW_ALREADY_RESIZED

    od_engine = pipeline_config["od_engine"]
    head_pose_engine = pipeline_config["head_pose_engine"]
    human_pose_engine = pipeline_config["human_pose_engine"]
    input_url: str = pipeline_config["input_url"]
    input_url_type: URLType = pipeline_config["input_url_type"]
    output_url: str = pipeline_config["output_url"]
    output_gst: bool = pipeline_config["output_gst"]
    force_resize: bool = pipeline_config["force_resize"]
    output_url_type: URLType = pipeline_config["output_url_type"]
    tracker_engine_type: str = pipeline_config["tracker_engine_type"]
    tracker_thresh: float = pipeline_config["tracker_thresh"]
    tracker_ttm: int = pipeline_config["tracker_ttm"]
    tracker_ttl: int = pipeline_config["tracker_ttl"]
    sequential: bool = pipeline_config["sequential"]
    loop_source: bool = pipeline_config["loop_source"]
    sync: bool = pipeline_config["sync"]
    draw_options: DrawOptions = pipeline_config["draw_options"]
    event_params: dict = pipeline_config["event_params"]
    multi_video_file_pipeline: bool = pipeline_config["multi_video_file_pipeline"]
    profile: bool = pipeline_config["profile"]

    if not (
        input_url_type is not URLType.VIDEO_FOLDER
        and output_url_type is not URLType.VIDEO_FOLDER
    ):
        logger.error("One of Input URL and output URL must not be the folder.")
        return None, []

    cap = cv2.VideoCapture(input_url)
    if not cap.isOpened():
        logger.error(f'Cannot open path "{input_url}"')
        return None, []

    # Doesn't matter which VideoCapture we are trying to use
    if VideoSource.get_is_ov2311():
        fourcc = cv2.VideoWriter_fourcc(*"BG10")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, cv2.COLOR_BAYER_RG2BGR)

    ret, frame = cap.read()
    if not ret:
        logger.error(f'Cannot read from path "{input_url}"')
        return None, []

    is_source_live = cap.get(cv2.CAP_PROP_FRAME_COUNT) == -1
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if is_source_live and sequential:
        logger.error("Refusing to create sequential pipeline on live video")
        sys.exit(1)

    if sequential:
        logger.debug("Using sequential (single thread) pipeline")
        video_source = SequentialVideoCapture(
            cap=cap,
            repeat_input=loop_source,
            force_resize_shape=(1600, 1300) if force_resize else None,
        )
    else:
        logger.debug("Using parallel (separate thread for VideoCapture) pipeline")
        video_source = ThreadedVideoCapture(
            cap=cap,
            repeat_input=loop_source,
            fps_limit=source_fps if sync else G_DECODER_FPS_LIMIT,
            force_resize_shape=(1600, 1300) if force_resize else None,
        )

    fps = video_source.fps
    logger.debug(f"[cv2] fps: {fps}")
    src_height, src_width = video_source.height, video_source.width

    tracker_kwargs = {
        "threshold": tracker_thresh,
        "ttm": tracker_ttm,
        "ttl": tracker_ttl,
        "image_size": (src_width, src_height),
    }

    tracker_engine = get_tracker_engine(
        engine_type=tracker_engine_type, **tracker_kwargs
    )

    if input_url_type == URLType.LIVE:
        imshow_title = "Camera/RTSP Live"
    elif multi_video_file_pipeline:
        imshow_title = "Event Preview"
    elif input_url_type == URLType.FILE:
        imshow_title = f"File: {os.path.basename(input_url)}"

    if output_url_type == URLType.X11:
        os.environ["DISPLAY"] = output_url
        cv2.namedWindow(
            imshow_title,
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED,
        )

        if not (multi_video_file_pipeline and G_CV2_WINDOW_ALREADY_RESIZED):
            # Do not resize on every video (while on multi video pipeline)
            # cv2.resizeWindow(imshow_title, (1280, 1020))
            cv2.resizeWindow(imshow_title, (800, 650))
            G_CV2_WINDOW_ALREADY_RESIZED = True
        writer = None
    else:
        if output_url_type == URLType.NONE:
            writer = DummyWriter()
        elif output_gst:
            logger.debug(f"[cv2] GStreamer pipeline: {output_url}")
            writer = cv2.VideoWriter(
                output_url,
                cv2.CAP_GSTREAMER,
                0,
                fps,
                (src_width, src_height),
                True,
            )
        else:
            writer = cv2.VideoWriter(
                output_url,
                cv2.VideoWriter_fourcc(*"avc1"),
                fps,
                (src_width, src_height),
            )

        if not writer.isOpened():
            logger.error("VideoWriter cannot be opened!")
            sys.exit(1)

    face_classes = ClassNamesManager.get_instance().get_face_class_ids()
    person_classes = ClassNamesManager.get_instance().get_person_class_ids()

    face_cropper = BlobCropper(
        class_filter=face_classes,
        crop_expand_type=BlobCropExpandType.PERCENTAGE,
        crop_expand_amount=0.2,
        face_head_expansion=False,
    )

    human_cropper = BlobCropper(
        class_filter=person_classes,
        crop_expand_type=BlobCropExpandType.NONE,
        crop_expand_amount=0.2,  # Not used
    )

    event_processors, events_per_passengers = create_event_processors(
        event_params,
        src_dims=(src_height, src_width),
    )

    driver_event_types = events_per_passengers["driver"]
    passenger_event_types = events_per_passengers["passenger"]

    driver_event = EventSerializer(event_types=driver_event_types)
    passenger_event = EventSerializer(event_types=passenger_event_types)

    # Key: frame number, Value: event information per events or NoneType
    all_events = []

    def on_event(event: BaseEvent):
        event_message = f"{event.ptype.name} {type(event).__name__} occured"
        event_message_extra = []
        if hasattr(event, "det_head"):
            event_message_extra.append(f"Head Track {event.det_head.tracker_id}")
        if hasattr(event, "det_person"):
            event_message_extra.append(f"Human Track {event.det_person.tracker_id}")
        if event_message_extra:
            event_message += f" ({', '.join(event_message_extra)})"
        logger.debug(event_message)

        if event.ptype == PassengerType.DRIVER:
            driver_event.update_current(event)
        elif event.ptype == PassengerType.PASSENGER:
            passenger_event.update_current(event)

        return True

    # Register same event callback for event processor
    # (Passenger/Driver)
    for processor in event_processors.values():
        processor.register(on_event)

    if not multi_video_file_pipeline:
        logger.info("Beginning pipeline ...")
    video_source.start()

    liveview_pause_state = False
    liveview_toggle_next_frame = False
    lat_single_image = 0  # will be measured at the end
    lat_draw_vis = 0
    lat_all_without_vis = 0
    det_items = 0

    try:
        while video_source.running:
            with MeasureLatency(
                name="single_image", strategy="pytime", profile=profile
            ) as m_single_image:
                frame = video_source.get_frame()
                thread_draw_fps, thread_last_ms = video_source.get_statistics()
                logger.debug(
                    f"Thread statistics: {thread_draw_fps:.02f} FPS (Latest image took {thread_last_ms} ms)"
                )

                if frame is None:
                    _ = m_single_image.measure()  # avoid warnings
                    continue

                (
                    detection_items,
                    latency_items,
                ) = infer_single_frame(
                    frame=frame,
                    human_cropper=human_cropper,
                    face_cropper=face_cropper,
                    od_engine=od_engine,
                    tracker_engine=tracker_engine,
                    human_pose_engine=human_pose_engine,
                    head_pose_engine=head_pose_engine,
                )

                current_event_info = {
                    "passenger": passenger_event.serialize_current(),
                    "driver": driver_event.serialize_current(),
                }

                # Prevent memory fillup
                if input_url_type != URLType.LIVE:
                    all_events.append(current_event_info)

                logger.debug(
                    f"Total single image latency (end-to-end): {lat_single_image:.01f} ms"
                )

                # Also draw measured latency values
                lat_det = latency_items["det"]
                lat_all_headpose = latency_items["all_headpose"]
                lat_all_humanpose = latency_items["all_humanpose"]
                lat_tracker = latency_items["tracker"]
                lat_extra_process = latency_items["extra_process"]
                lat_event = latency_items["event"]

                with MeasureLatency(name="draw", strategy="pytime") as m_draw:
                    draw_boxes_and_tracks(
                        frame=frame,
                        face_classes=face_classes,
                        detection_items=detection_items,
                        tracker_engine=tracker_engine,
                        draw_options=draw_options,
                        event_info=current_event_info,
                        per_passenger_event_types=events_per_passengers,
                    )

                    if draw_options.latency_info:
                        if draw_options.event_stats:
                            latency_info_pad = 400
                        else:
                            latency_info_pad = 0

                        draw_info_text(
                            frame,
                            f"Detector {lat_det:.01f} ms\n"
                            + f"Head Pose: {lat_all_headpose:.01f} ms\n"
                            + f"Human Pose: {lat_all_humanpose:.01f} ms\n"
                            + f"Tracker: {lat_tracker:.01f} ms\n"
                            + f"Pre/postprocess: {lat_extra_process:.01f} ms\n"
                            + f"Event: {lat_event:.01f} ms\n",
                            left=4 + latency_info_pad,
                        )

                        draw_info_text(
                            frame,
                            f"(t-1) Draw + Visualize: {lat_draw_vis:.01f} ms\n"
                            f"(t-1) End2End (throughput): {lat_single_image:.01f} ms\n",
                            left=400 + latency_info_pad,
                        )

                    if output_url_type == URLType.X11:
                        if draw_options.legends:
                            draw_info_text(
                                frame,
                                "Quit: ESC\n"
                                + "Pause/Resume: SPC\n"
                                + "Next frame: n\n"
                                + "Toggle headpose: 1\n"
                                + "Toggle humanpose: 2\n"
                                + "Toggle bboxes: 3\n"
                                + "Toggle attribs: 4\n"
                                + "Toggle notrack/backseat: 5\n",
                                left=1100,
                            )

                        def toggle_option(draw_options: DrawOptions, item: str):
                            prev_value = draw_options[item]
                            next_state = "off" if prev_value else "on"
                            logger.info(f"Turning {item} {next_state}")

                            draw_options[item] = not prev_value

                        cv2.imshow(imshow_title, frame)
                        while True:
                            ret = cv2.waitKey(1)
                            if ret != -1:
                                logger.debug(f"waitKey() ret={ret}")
                            if ret == 49:  # "1"
                                # Toggle headpose
                                toggle_option(draw_options, "headpose_item_values")
                                toggle_option(draw_options, "headpose")
                            if ret == 50:  # "2"
                                # Toggle humanpose
                                toggle_option(draw_options, "humanpose")
                            if ret == 51:  # "3"
                                # Toggle bboxes
                                toggle_option(draw_options, "bboxes")
                            if ret == 52:  # "4"
                                # Toggle attributes
                                toggle_option(draw_options, "box_track_info")
                            if ret == 53:  # "5"
                                # Toggle notrack/backseat
                                draw_options["untracked_items"] = draw_options[
                                    "backseat"
                                ]
                                toggle_option(draw_options, "untracked_items")
                                toggle_option(draw_options, "backseat")
                            if ret == 32:  # "SPC"
                                # Pause/Resume pipeline
                                liveview_pause_state = not liveview_pause_state
                            if ret == 110:  # "n"
                                liveview_toggle_next_frame = True
                                if not sequential:
                                    logger.warning(
                                        "Parallel decoder in progress in background"
                                    )
                                    logger.warning(
                                        "After seek, you will have latest frame instead of right next frame."
                                    )
                            if ret == 27:  # "ESC"
                                # Exit pipeline
                                logger.info("Interrupted pipeline")
                                raise KeyboardInterrupt()
                            if liveview_pause_state and not liveview_toggle_next_frame:
                                time.sleep(0.1)
                                continue
                            liveview_toggle_next_frame = False
                            break
                    else:
                        writer.write(frame)
                    lat_draw_vis = m_draw.measure()

                with MeasureLatency(name="event", strategy="pytime") as m_event:
                    # Reset last event stats
                    passenger_event.clear_current()
                    driver_event.clear_current()

                    # Process events
                    for event_processor in event_processors.values():
                        event_processor.update(detection_items)

                    lat_event = m_event.measure()

                time_to_wait = max(0, 1 / fps - lat_single_image / 1000)
                if sync and time_to_wait > 0:
                    logger.debug(
                        f"Waiting for next frame: {int(time_to_wait * 1000)}ms"
                    )
                    time.sleep(time_to_wait)

                lat_single_image = m_single_image.measure()
                lat_all_without_vis += lat_single_image - lat_draw_vis
            det_items += 1

    except (KeyboardInterrupt, SystemExit) as error:
        logger.info(f"Pipeline gracefully stopped (Total {det_items} frames detected)")
        if multi_video_file_pipeline:
            raise error

    if not multi_video_file_pipeline:
        logger.info(f"End of pipeline (Total {det_items} frames detected)")

    if output_url_type == URLType.X11 and not multi_video_file_pipeline:
        cv2.destroyAllWindows()
    if writer and writer.isOpened():
        writer.release()
    if video_source and video_source.cap.isOpened():
        video_source.stop()
        video_source.destroy()

    cumulative_events = {
        "passenger": passenger_event.serialize_cumulates(),
        "driver": driver_event.serialize_cumulates(),
    }

    return cumulative_events, all_events, events_per_passengers, lat_all_without_vis


def infer_single_frame(
    frame: np.ndarray,
    human_cropper: BlobCropper,
    face_cropper: BlobCropper,
    od_engine: TRTInfer,
    tracker_engine: BaseTracker,
    human_pose_engine: TRTInfer,
    head_pose_engine: TRTInfer,
) -> Tuple[List[DetectionItem], Dict[str, float]]:
    src_height, src_width, _ = frame.shape

    lat_core_items = []
    lat_all_events = 0
    with MeasureLatency(name="engine", strategy="pytime") as m_engine:
        od_batch = YoloNASProcessor.get_instance().preprocess(frame)
        with MeasureLatency(name="detector", strategy="cuda") as m_det:
            od_outputs = od_engine.infer(od_batch)
            lat_det = m_det.measure()
            lat_core_items.append(lat_det)

        (
            pred_boxes,
            pred_scores,
            pred_class_id,
        ) = YoloNASProcessor.get_instance().postprocess(od_outputs)

        detection_items: List[DetectionItem] = parse_od_predictions(
            pred_boxes, pred_scores, pred_class_id
        )

        with MeasureLatency(name="tracker", strategy="pytime") as m_tracker:
            tracker_engine.track(detection_items)
            lat_tracker = m_tracker.measure()
            lat_core_items.append(lat_tracker)

        human_dets, human_blobs = human_cropper.filter_crop_blob(frame, detection_items)

        lat_all_humanpose = 0
        for (det_idx, human_det), (human_blob, cropped_amount) in zip(
            human_dets, human_blobs
        ):
            human_batch = SBPProcessor.get_instance().preprocess(human_blob)
            with MeasureLatency(name="humanpose", strategy="cuda") as m_humanpose:
                heatmaps = human_pose_engine.infer(human_batch)
                lat_humanpose = m_humanpose.measure()
                lat_core_items.append(lat_humanpose)

            human_pose = SBPProcessor.get_instance().postprocess(heatmaps, human_det)
            human_pose = human_cropper.shrink_pose(human_pose, cropped_amount)

            human_pose_item: HumanPoseDetectionItem = (
                HumanPoseDetectionItem.derive_from(
                    detection_items[det_idx], human_pose=human_pose
                )
            )
            detection_items[det_idx] = human_pose_item

            with MeasureLatency(name="event", strategy="pytime") as m_event:
                divide_passenger_single_(
                    human_pose_item,
                    (src_height, src_width),
                    accept_threshold=0.2,
                    backseat_face_threshold=[160, 160],
                    is_reversed=False,
                )

                if human_pose_item.passenger_type == PassengerType.PASSENGER:
                    pass
                elif human_pose_item.passenger_type == PassengerType.DRIVER:
                    pass

                lat_event = m_event.measure()
                lat_all_events += lat_event

        if not human_dets:
            logger.debug("No human pose")

        face_dets, face_blobs = face_cropper.filter_crop_blob(frame, detection_items)

        lat_all_headpose = 0

        # Baseline videos for those yaw/pitch mean values
        # - basedir: \\172.30.1.5\videoDrive\_VideoData2023\PIS\raw_datasets
        #
        # 1. data_230905\PIS_sort\group_08\31_02_2\20230825_010337_Video_USB.avi
        # 2. data_230905\PIS_sort\group_08\31_02_2\20230825_011544_Video_USB.avi
        # 3. data_230905\PIS_sort\group_08\29_02_2\20230814_025816_Video_USB.avi
        # 4. data_230728\PIS_sort\group_03\3_03_1\20230621_110536_Video_USB.avi
        driver_yaw_mean = -40.0
        driver_pitch_mean = 5.0
        shotgun_yaw_mean = 28.0
        shotgun_pitch_mean = 0.0

        yaw_warn = 35.0
        pitch_warn = 10.0

        for (det_idx, face_det), (face_blob, cropped_amount) in zip(
            face_dets, face_blobs
        ):
            face_batch = SixDRepNetProcessor.get_instance().preprocess(face_blob)

            with MeasureLatency(name="headpose", strategy="cuda") as m_headpose:
                face_output_batch = head_pose_engine.infer(face_batch)
                lat_headpose = m_headpose.measure()
                lat_core_items.append(lat_headpose)
                lat_all_headpose += lat_headpose

            face_output = SixDRepNetProcessor.get_instance().postprocess(
                face_output_batch
            )
            head_pose_item: HeadPoseDetectionItem = HeadPoseDetectionItem.derive_from(
                detection_items[det_idx], head_angles=face_output
            )
            detection_items[det_idx] = head_pose_item

            with MeasureLatency(name="event", strategy="pytime") as m_event:
                divide_passenger_single_(
                    head_pose_item,
                    (src_height, src_width),
                    accept_threshold=0.2,
                    backseat_face_threshold=[170, 170],
                    is_reversed=False,
                )

                if head_pose_item.passenger_type == PassengerType.DRIVER:
                    (
                        driver_yaw_calib,
                        driver_pitch_calib,
                    ) = HeadPoseCalibrationProcessor.get_instance().calib(
                        face_output, driver_yaw_mean, driver_pitch_mean
                    )

                    head_pose_item.events = get_distraction_event(
                        driver_yaw_calib, driver_pitch_calib, yaw_warn, pitch_warn
                    )

                elif head_pose_item.passenger_type == PassengerType.PASSENGER:
                    (
                        shotgun_yaw_calib,
                        shotgun_pitch_calib,
                    ) = HeadPoseCalibrationProcessor.get_instance().calib(
                        face_output, shotgun_yaw_mean, shotgun_pitch_mean
                    )

                    head_pose_item.events = get_distraction_event(
                        shotgun_yaw_calib, shotgun_pitch_calib, yaw_warn, pitch_warn
                    )

                # print(head_pose_item.events)
                lat_event = m_event.measure()
                lat_all_events += lat_event

        if not face_dets:
            logger.debug("No face pose")

        lat_all_engine = m_engine.measure()

    lat_core_processes = sum(lat_core_items)
    lat_extra_process = lat_all_engine - lat_core_processes

    latency_items = {
        "det": lat_det,
        "all_headpose": lat_all_headpose,
        "all_humanpose": lat_all_humanpose,
        "event": lat_all_events,
        "tracker": lat_tracker,
        "extra_process": lat_extra_process,
    }

    return (detection_items, latency_items)

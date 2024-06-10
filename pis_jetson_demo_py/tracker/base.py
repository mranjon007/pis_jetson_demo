from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from base.types import TrackInfo
from engine.core.detections import DetectionItem


def argmax(items: List[Any]) -> int:
    return sorted(enumerate(items), key=lambda item: item[1], reverse=True)[0][0]


def argmin(items: List[Any]) -> int:
    return sorted(enumerate(items), key=lambda item: item[1], reverse=False)[0][0]


class TrackItem:
    @staticmethod
    def create(
        track_id: int,
        dets: List[DetectionItem],
        ttl: int,
        cumulative_dets: int,
        metrics: Any,
    ) -> TrackItem:
        for det in dets:
            det.tracker_id = track_id
            det.tracker_metrics = metrics

        return TrackItem(
            track_id=track_id,
            dets=dets,
            ttl=ttl,
            cumulative_dets=cumulative_dets,
        )

    @staticmethod
    def destroy(item: TrackItem) -> None:
        for det in item.dets:
            det.tracker_id = None
            det.tracker_metrics = None

    def __init__(
        self,
        track_id: int,
        dets: List[DetectionItem],
        ttl: int,
        cumulative_dets: int,
        track_max_dets: int = 30,
    ):
        self.track_id = track_id
        self.dets = dets
        self.ttl = ttl
        self.cumulative_dets = cumulative_dets
        self.track_max_dets = track_max_dets

    def add_detection(self, det: DetectionItem, metrics: Any):
        if len(self.dets) >= self.track_max_dets:
            self.dets.pop(0)

        det.tracker_id = self.track_id
        det.tracker_metrics = metrics
        self.dets.append(det)
        self.cumulative_dets += 1


class Tracker(ABC):
    """Tracker base class"""

    @abstractmethod
    def track(
        self, detections: List[DetectionItem]
    ) -> Tuple[List[Tuple[int, DetectionItem, TrackInfo]], List[DetectionItem]]:
        """Track prediction results

        Args:
            preds (List[DetectionItem]): Predicted bounding boxes [[xmin, ymin, xmax, ymax, conf, class_id], ...]

        Returns:
            Tuple[List[Tuple[int, DetectionItem, TrackInfo]], List[DetectionItem]]: Tracked results (matched_items, unmatched_items)
            - matched_items: [[track_id, DetectionItem, (cumulative_dets, last_track_metrics)], ...]
            - unmatched_items: [[DetectionItem], ...]
        """


class BaseTracker(Tracker):
    """Tracker with basic features

    You must inherit this class and implement custom metrics in `calculate_metric`.
    for instance, implement IoU in `calculate_metric` for IoU Tracker.

    - TTM(Time To Measure): object appearance counts until making new track
    - TTL(Time To Live): disappearance counts until track to be removed
    - Track: Set of object of tracker which is considered as independent one.

    """

    def __init__(
        self,
        threshold: float,
        ttm: int,
        ttl: int,
        image_size: Tuple[int, int],
        best_metric_type: str = "max",
    ):
        self.threshold = threshold
        self.ttm = ttm
        self.ttl = ttl
        self.image_size = image_size

        assert best_metric_type in ["min", "max"], "Type must be either 'min' or 'max'."
        self.best_metric_type = best_metric_type

        self.tracks: Dict[int, TrackItem] = {}
        self.candidates: List[Tuple[int, List[DetectionItem]]] = []

    def track(self, detections: List[DetectionItem]) -> None:
        # Shallow copy from detections list
        all_detections = detections.copy()

        # Make every track ttl increased by one
        for track in self.tracks.values():
            track.ttl += 1

        for candidate_track in self.candidates:
            candidate_track[0] += 1

        # Match box with existing track's representational bbox
        for track_item in self.tracks.values():
            # last track det -> track representational bbox
            metrics = [
                self.calculate_metric(track_item.dets[-1].bbox(), det.bbox())
                for det in all_detections
            ]

            if not metrics:
                continue

            if self.best_metric_type == "min":
                best_metric_occurence = min(metrics)
                bbox_idx = argmin(metrics)
                over_threshold = best_metric_occurence <= self.threshold
            elif self.best_metric_type == "max":
                best_metric_occurence = max(metrics)
                bbox_idx = argmax(metrics)
                over_threshold = best_metric_occurence >= self.threshold

            if over_threshold:
                det = all_detections.pop(bbox_idx)
                track_item.ttl = 0
                track_item.add_detection(
                    det, best_metric_occurence
                )  # Will assign track id

        # Boxes which are not belong the track are considered as 'candidate'
        for det in all_detections:
            candidate_metrics = [
                self.calculate_metric(det.bbox(), candidate_rep_bbox.bbox())
                for ttl, (candidate_rep_bbox, *_) in self.candidates
            ]

            # Add to unmatched items (whether it matches candidates or not)
            # because the box matches with existing track is already removed above
            # unmatched_items.append(det)
            det.tracker_id = None
            det.tracker_metrics = None

            # No candidate tracks
            if candidate_metrics:
                if self.best_metric_type == "min":
                    best_metric_occurence = min(candidate_metrics)
                    candidate_idx = argmin(candidate_metrics)
                    over_threshold = best_metric_occurence <= self.threshold
                elif self.best_metric_type == "max":
                    best_metric_occurence = max(candidate_metrics)
                    candidate_idx = argmax(candidate_metrics)
                    over_threshold = best_metric_occurence >= self.threshold

                if over_threshold:
                    # Match existing candidate tracks with bbox
                    det.tracker_metrics = best_metric_occurence
                    self.candidates[candidate_idx][1].append(det)
                    continue  # do not create new candidates boxes

            # Non-matching bboxes are considered as new candidates
            self.candidates.append([0, [det]])

        # Turn all candidate tracks into tracks which matches period
        while True:
            for idx, (ttl, candidate_dets) in enumerate(self.candidates):
                if len(candidate_dets) >= self.ttm:
                    if not self.tracks:
                        new_track_id = 0
                    else:
                        new_track_id = int(
                            max([det.track_id for det in self.tracks.values()]) + 1
                        )

                    _, candidate_dets = self.candidates.pop(idx)
                    # track_id, track_dets, track_ttl, cumulative_dets
                    self.tracks[
                        new_track_id
                    ] = TrackItem.create(  # also tells dets for its track id
                        track_id=new_track_id,
                        dets=candidate_dets,
                        ttl=0,
                        cumulative_dets=len(candidate_dets),
                        metrics=0,
                    )

                    continue
            break

        # Remove tracks which are over TTL
        for track_item in self.tracks.values():
            if track_item.ttl >= self.ttl:
                TrackItem.destroy(track_item)

        self.tracks = {
            track_id: track_item
            for track_id, track_item in self.tracks.items()
            if track_item.ttl < self.ttl
        }

        self.candidates = [
            [ttl, candidate_dets]
            for ttl, candidate_dets in self.candidates
            if ttl < self.ttl
        ]

    @abstractmethod
    def calculate_metric(self, bbox1: np.ndarray, bbox2: np.ndarray) -> Any:
        pass

    @abstractmethod
    def get_tracker_type(self) -> str:
        pass

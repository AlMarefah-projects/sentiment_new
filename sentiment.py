"""
Customer Satisfaction System
TransformsAI v3.0

Detects and tracks peak person counts across multiple camera streams,
then periodically sends the peak frame and metadata to a backend API.
"""

import cv2
import json
import logging
import os
import sys
import time
import threading
from queue import Queue

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes

from libraries.datasend import DataUploader
from libraries.utils import time_to_string, mat_to_response
from libraries.stream_publisher import StreamPublisher
from libraries.async_capture import VideoCaptureAsync


__version__ = "3.0"
__author__ = "TransformsAI"

logger = logging.getLogger(__name__)


def draw_detection_boxes(frame: np.ndarray, results) -> np.ndarray:
    """
    Draw bounding boxes and confidence scores on a frame.

    Args:
        frame:   The original BGR frame from the camera.
        results: The YOLO prediction result for this frame.

    Returns:
        A new frame with boxes and labels rendered on it.
    """
    annotated = frame.copy()
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = f"{results[0].names[class_id]} {confidence:.2f}"

        # Draw rectangle and label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(
            annotated, label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return annotated


def run_detection(model: YOLO, config: dict):
    """
    Main detection loop.

    For each configured stream this function:
      1. Opens an async video capture.
      2. Optionally starts an MQTT livestream publisher.
      3. Periodically runs YOLO inference and tracks the peak person count.
      4. At every datasend_interval, uploads the peak frame + metadata.
      5. At every heartbeat_interval, sends a heartbeat to the backend.

    Args:
        model: A loaded YOLO model used for person detection.
        config:       The parsed JSON configuration dictionary.
    """
    # ------------------------------------------------------------------ #
    # Read tunable intervals and feature flags from config                 #
    # ------------------------------------------------------------------ #
    inference_interval  = config["inference_interval"]      # seconds between inference calls
    heartbeat_interval  = config["heartbeat_interval"]      # seconds between heartbeat pings
    datasend_interval   = config["datasend_interval"]       # seconds between data uploads
    frame_send_width    = config["frame_send_width"]        # resize width before upload
    frame_send_quality  = config["frame_send_jpeg_quality"] # JPEG quality for upload
    enable_livestream   = config["livestream"]               # MQTT livestream toggle
    enable_draw         = config["draw"]                     # draw bounding boxes toggle

    secret_header = {"X-Secret-Key": config["X-Secret-Key"]}

    # Shared uploader used by all streams
    uploader = DataUploader(
        config["data_send_url"],
        config["heartbeat_url"],
        secret_header,
        project_version=__version__,
    )

    # ------------------------------------------------------------------ #
    # Initialise every stream defined in config["streams"]                 #
    # ------------------------------------------------------------------ #
    streams = []

    for stream_cfg in config.get("streams", []):
        serial_number = stream_cfg.get("sn")
        if not serial_number:
            logger.warning(
                f"Stream configuration missing 'sn'. Skipping this stream: {stream_cfg}"
            )
            continue

        # Prefer a local file source when local_video is true
        if stream_cfg["local_video"]:
            video_source = stream_cfg["local_video_source"]
        else:
            video_source = stream_cfg["video_source"]

        if not video_source:
            logger.warning(f"Stream {serial_number}: Video source not specified. Skipping.")
            continue

        logger.info(f"Initializing stream: {serial_number} from source: {video_source}")

        # Heartbeat config forwarded to the async capture wrapper
        heartbeat_config = {
            "enabled": True,
            "sn": serial_number,
            "uploader_config": {
                "api_url": None,
                "heartbeat_url": config["heartbeat_url"],
                "headers": {"X-Secret-Key": config["X-Secret-Key"]},
                "debug": True,
                "max_workers": 2,
                "source": "Video Capture",
                "project_version": __version__,
            },
        }

        capture = VideoCaptureAsync(
            src=video_source,
            heartbeat_config=heartbeat_config,
            auto_restart_on_fail=True,
        )
        capture.start()

        # Optional MQTT livestream publisher
        livestreamer = None
        
        streams.append({
            "sn":                   serial_number,
            "config":               stream_cfg,
            "cap":                  capture,
            "streamer":             livestreamer,
            # detection state
            "max_sentiment_count":     0,
            "max_person_frame":     None,   # raw (un-annotated) peak frame
            # timestamps
            "last_inference_time":  time.time(),
            "last_datasend_time":   time.time(),
            "last_heartbeat_time":  time.time(),
            "frame_count":          0,
        })

    if not streams:
        logger.error("No streams were successfully initialized. Exiting live detection.")
        return

    # ------------------------------------------------------------------ #
    # Main processing loop                                                 #
    # ------------------------------------------------------------------ #
    try:
        while True:
            for stream in streams:
                capture = stream["cap"]

                # Skip streams that haven't started yet
                if not capture or not capture.started:
                    continue

                success, frame = capture.read()
                if not success:
                    continue

                now = time.time()
                stream["frame_count"] += 1

                # ---- Inference ---------------------------------------- #
                if now - stream["last_inference_time"] >= inference_interval:
                    logger.debug(
                        f"Stream {stream['sn']}: Triggering inference. "
                        f"Frame: {stream['frame_count']}"
                    )

                    results = model.predict(frame, verbose=False)
                    person_count = len(results[0].boxes)

                    # Update peak count and save the raw frame for upload
                    if person_count > stream["max_sentiment_count"]:
                        stream["max_sentiment_count"] = person_count
                        stream["max_sentiment_frame"] = frame.copy()  # always store raw frame
                        logger.debug(
                            f"Stream {stream['sn']}: New max sentiment count: "
                            f"{person_count} at frame {stream['frame_count']}"
                        )

                    # Optionally annotate the live frame for the livestream
                    if enable_draw:
                        frame = draw_detection_boxes(frame, results)

                    stream["last_inference_time"] = now

                # ---- Data upload -------------------------------------- #
                if now - stream["last_datasend_time"] >= datasend_interval:
                    if stream["max_sentiment_count"] == 0:
                        stream["last_datasend_time"] = now
                        logger.debug(
                            f"Stream {stream['sn']}: No persons detected. Skipping data send."
                        )
                        continue

                    payload = {
                        "sn":               stream["sn"],
                        "is_ai_annotated":  False,
                        "start_time":       time_to_string(stream["last_datasend_time"]),
                        "end_time":         time_to_string(now),
                    }

                    # Always upload the clean (non-annotated) peak frame
                    upload_frame   = stream["max_person_frame"]
                    encoded_image  = mat_to_response(
                        upload_frame, frame_send_width, frame_send_quality, timestamp=now
                    )

                    try:
                        uploader.send_data(payload, files={"image": encoded_image})
                    except Exception as exc:
                        logger.error(
                            f"Stream {stream['sn']}: Error sending data: {exc}",
                            exc_info=True,
                        )

                    # Reset peak state for the next interval
                    stream["max_sentiment_count"] = 0
                    stream["max_person_frame"] = None
                    stream["last_datasend_time"] = now

                # ---- Livestream frame push ---------------------------- #
                if stream["streamer"]:
                    stream["streamer"].updateFrame(frame)

                # ---- Heartbeat --------------------------------------- #
                if now - stream["last_heartbeat_time"] >= heartbeat_interval:
                    try:
                        uploader.send_heartbeat(stream["sn"], timestamp=time_to_string(now))
                        logger.info(f"Stream {stream['sn']}: Heartbeat sent.")
                        stream["last_heartbeat_time"] = now
                    except Exception as exc:
                        logger.error(
                            f"Stream {stream['sn']}: Error sending heartbeat: {exc}",
                            exc_info=True,
                        )

    except Exception as exc:
        logger.error(f"Fatal error in detection loop: {exc}", exc_info=True)


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(logging.DEBUG)
    logger.info("Starting Customer Satisfaction System...")

    if len(sys.argv) < 2:
        print("Usage: python sentiment.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    print(f"Received config path: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as exc:
        logger.error(f"Failed to load configuration from {config_path}: {exc}")
        sys.exit(1)

    run_detection(YOLO(config["model"]), config)
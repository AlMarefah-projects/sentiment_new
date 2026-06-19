import cv2, torch, numpy as np, json, time, os, threading
from queue import Queue
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import logging, sys
from libraries.datasend import DataUploader
from libraries.utils import time_to_string, mat_to_response
from libraries.stream_publisher import StreamPublisher
from libraries.async_capture import VideoCaptureAsync

logger = logging.getLogger(__name__)
__version__ = '3.0'
__author__ = 'TransformsAI'


def run_live_detection(person_model, config):
    inference_interval      = config.get('inference_interval', 1)
    heartbeat_interval      = config.get('heartbeat_interval', 30)
    datasend_interval       = config.get('datasend_interval', 120)
    frame_send_width        = config.get('frame_send_width', 1920)
    frame_send_jpeg_quality = config.get('frame_send_jpeg_quality', 65)
    use_livestream          = config.get('livestream', False)
    iou_threshold           = config.get('iou_threshold', 0.45)
    target_device           = config.get('target', 'cpu')
    draw_boxes              = config.get('draw', False)
    show_preview            = config.get('show', False)
    send_data_enabled       = config.get('send_data', True)
    demo_mode               = config.get('demo', False)
    mac_address             = config.get('mac_address', '')

    uploader = DataUploader(
        config['data_send_url'],
        config['heartbeat_url'],
        {'X-Secret-Key': config['X-Secret-Key']},
        project_version=__version__,
    )

    streams = []
    for stream_cfg in config.get('streams', []):
        sn = stream_cfg.get('sn')
        if not sn:
            logger.warning(f"Stream configuration missing 'sn'. Skipping this stream: {stream_cfg}")
            continue

        video_source = stream_cfg.get('local_video_source') if stream_cfg.get('local_video', True) else stream_cfg.get('video_source')
        if not video_source:
            logger.warning(f"Stream {sn}: Video source not specified. Skipping.")
            continue

        logger.info(f"Initializing stream: {sn} from source: {video_source}")

        heartbeat_config = {
            'enabled': True,
            'sn': sn,
            'uploader_config': {
                'api_url': None,
                'heartbeat_url': config.get('heartbeat_url'),
                'headers': {'X-Secret-Key': config.get('X-Secret-Key', '')},
                'debug': True,
                'max_workers': 2,
                'source': 'Video Capture',
                'project_version': __version__,
            },
        }

        cap = VideoCaptureAsync(src=video_source, heartbeat_config=heartbeat_config, auto_restart_on_fail=True)
        cap.start()

        streamer = None
        if use_livestream:
            try:
                streamer = StreamPublisher(
                    f"live_{sn}",
                    start_stream=False,
                    host=config.get('local_ip', '127.0.0.1'),
                    port=config.get('mqtt_port', 1883),
                )
                streamer.start_streaming()
                logger.info(f"Livestream publisher started for stream {sn} on topic live_{sn}")
            except Exception as exc:
                logger.error(f"Failed to start livestreamer for stream {sn}: {exc}", exc_info=True)

        streams.append({
            'sn': sn,
            'config': stream_cfg,
            'cap': cap,
            'streamer': streamer,
            'max_person_count': 0,
            'max_person_frame': None,
            'last_inference_time': time.time(),
            'last_datasend_time': time.time(),
            'last_heartbeat_time': time.time(),
            'frame_count': 0,
        })

    if not streams:
        logger.error('No streams were successfully initialized. Exiting live detection.')
        return

    try:
        while True:
            for stream in streams:
                if not stream['cap'] or not stream['cap'].started:
                    continue

                success, frame = stream['cap'].read()
                if not success:
                    continue

                now = time.time()
                stream['frame_count'] += 1

                if now - stream['last_inference_time'] >= inference_interval:
                    logger.debug(f"Stream {stream['sn']}: Triggering inference. Frame: {stream['frame_count']}")

                    results = person_model.predict(frame, verbose=False, iou=iou_threshold, device=target_device)
                    boxes = results[0].boxes
                    person_count = len(boxes)

                    if draw_boxes:
                        annotated_frame = frame.copy()
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = results[0].names[class_id]
                            label = f"{class_name} " #{confidence:.2f}"
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + text_size[0] + 10, y1), (0, 255, 0), -1)
                            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        frame = annotated_frame

                    if person_count > stream['max_person_count']:
                        stream['max_person_count'] = person_count
                        stream['max_person_frame'] = frame.copy()
                        logger.debug(f"Stream {stream['sn']}: New max person count: {person_count} at frame {stream['frame_count']}")

                    stream['last_inference_time'] = now

                if now - stream['last_datasend_time'] >= datasend_interval:
                    if stream['max_person_count'] == 0:
                        stream['last_datasend_time'] = now
                        logger.debug(f"Stream {stream['sn']}: No persons detected. Skipping data send.")
                        continue

                    if send_data_enabled:
                        payload = {
                            'sn': stream['sn'],
                            'is_ai_annotated': False,
                            'start_time': time_to_string(stream['last_datasend_time']),
                            'end_time': time_to_string(now),
                            'mac_address': mac_address,
                        }
                        image_file = mat_to_response(
                            stream['max_person_frame'],
                            frame_send_width,
                            frame_send_jpeg_quality,
                            timestamp=now,
                        )
                        try:
                            uploader.send_data(payload, files={'image': image_file})
                        except Exception as exc:
                            logger.error(f"Stream {stream['sn']}: Error sending data: {exc}", exc_info=True)

                    stream['max_person_count'] = 0
                    stream['max_person_frame'] = None
                    stream['last_datasend_time'] = now

                if stream['streamer']:
                    stream['streamer'].updateFrame(frame)

                if show_preview:
                    cv2.imshow(f"Stream {stream['sn']}", frame)
                    cv2.waitKey(1)

                if now - stream['last_heartbeat_time'] >= heartbeat_interval:
                    try:
                        uploader.send_heartbeat(stream['sn'], timestamp=time_to_string(now))
                        logger.info(f"Stream {stream['sn']}: Heartbeat sent.")
                        stream['last_heartbeat_time'] = now
                    except Exception as exc:
                        logger.error(f"Stream {stream['sn']}: Error sending heartbeat: {exc}", exc_info=True)

    except Exception as exc:
        logger.error(f"Error in processing stream {stream['sn']}: {exc}", exc_info=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python sentiment.py <config_path>')
        sys.exit(1)

    config_path = sys.argv[1]
    print(f"Received config path: {config_path}")

    try:
        with open(config_path, 'r') as fh:
            config = json.load(fh)
    except Exception as exc:
        print(f"Failed to load configuration from {config_path}: {exc}")
        sys.exit(1)

    log_level = getattr(logging, config.get('logging_level', 'info').upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(log_level)
    logger.info('Starting Customer Satisfaction System...')

    run_live_detection(YOLO(config['person_model']), config)
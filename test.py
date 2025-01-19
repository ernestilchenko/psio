import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

TARGET_CLASSES = {
    2: 'car',
    0: 'person',
    1: 'bicycle'
}


def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    if len(detections) > 0:
        mask = np.array([class_id in TARGET_CLASSES.keys() for class_id in detections.class_id])
        if np.any(mask):
            detections = detections[mask]
            detections = tracker.update_with_detections(detections)

            labels = [
                f"#{tracker_id} {TARGET_CLASSES[class_id]}"
                for class_id, tracker_id
                in zip(detections.class_id, detections.tracker_id)
            ]

            annotated_frame = box_annotator.annotate(
                frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(
                annotated_frame, detections=detections, labels=labels)
            return trace_annotator.annotate(
                annotated_frame, detections=detections)

    return frame


sv.process_video(
    source_path="/Users/ernestilchenko/Downloads/parking/IMG_6300.MOV",
    target_path="result_3.mp4",
    callback=callback
)

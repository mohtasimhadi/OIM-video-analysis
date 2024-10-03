import cv2
import yaml
import os
import numpy as np
from ultralytics.utils.plotting import Annotator

def initialize_video_capture(video_path, result_path):
    cap = cv2.VideoCapture(video_path)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, out, total_frames

def write_annotated_video(annotator, out):
    if annotator:
        out.write(annotator.result())

def save_tracked_objects(track_history, tracked_objects_path):
    tracked_data = []
    for track_id, data in track_history.items():
        mask_img_path = os.path.join(tracked_objects_path, f"track_{track_id}.png")

        data_frame_contiguous = np.ascontiguousarray(data['frame'])
        annotator = Annotator(data_frame_contiguous, line_width=2)
        annotator.seg_bbox(mask=data['mask'], mask_color=(0, 255, 0))  # Use a fixed color for now
        cv2.imwrite(mask_img_path, annotator.result())

        tracked_data.append({
            "track_id": track_id,
            "confidence": data['confidence'],
            "image_path": f"track_{track_id}.png",
            "mask": data['mask'].tolist()
        })

    yaml_path = os.path.join(tracked_objects_path, "annotations.yaml")
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(tracked_data, yaml_file)

    print(f"Total unique objects tracked: {len(track_history.items())}")
    print(f"Tracked objects metadata saved to: {yaml_path}")

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator
from engines.quality_checker import quality_assessment

def initialize_video_capture(video_path, result_path):
    print('..................................................')
    cap = cv2.VideoCapture(video_path)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*"h264"), fps, (w, h))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, out, total_frames

def write_annotated_video(annotator, out):
    if annotator:
        out.write(annotator.result())

def get_tracked_objects(track_history):
    tracked_data = []
    for track_id, data in track_history.items():

        data_frame_contiguous = np.ascontiguousarray(data['frame'])
        annotator = Annotator(data_frame_contiguous, line_width=2)
        annotator.seg_bbox(mask=data['mask'], mask_color=(0, 255, 0))

        quality_data = quality_assessment(data['mask'].tolist())

        tracked_data.append({
            "track_id": track_id,
            "confidence": round(data['confidence'], 2),
            "image": annotator.result(),
            'area': round(quality_data['area'], 2),
            'perimeter': round(quality_data['perimeter'], 2),
            'circularity': round(quality_data['circularity'], 2),
            'eccentricity': round(quality_data['eccentricity'], 2),
            'class_name': data['class_name']
        })

    return tracked_data

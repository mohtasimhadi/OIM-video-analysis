import os
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from utils.video_utils import initialize_video_capture, write_annotated_video, get_tracked_objects
import random

class YOLOTracker:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path, verbose=False)
        self.video_path = video_path
        self.result_path = 'temp_result.mp4'
        self.track_history = {}
        self.color_map = {}
        self.class_names = self.model.names


    def process_frame(self, im0):
        results = self.model.track(im0, persist=True, verbose=False)

        if results[0].boxes.id is not None and results[0].boxes.conf is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cuda().tolist()
            confidences = results[0].boxes.conf.cuda().tolist()
            class_ids = results[0].boxes.cls.int().cuda().tolist()

            self.update_track_history(im0, masks, track_ids, confidences, class_ids)
            return self.annotate_frame(im0, masks, track_ids)
        return None

    def update_track_history(self, im0, masks, track_ids, confidences, class_ids):
        for mask, track_id, confidence, class_id in zip(masks, track_ids, confidences, class_ids):
            if track_id not in self.track_history or confidence > self.track_history[track_id]['confidence']:
                x_min, y_min = mask.min(axis=0)
                x_max, y_max = mask.max(axis=0)
                x_min, y_min = int(x_min), int(y_min)
                x_max, y_max = int(x_max), int(y_max)

                cropped_frame = im0[y_min:y_max, x_min:x_max]
                cropped_frame = np.ascontiguousarray(cropped_frame)
                self.track_history[track_id] = {
                    "confidence": confidence,
                    "frame": cropped_frame,
                    "mask": mask - [x_min, y_min],
                    "class_name": self.class_names[class_id]
                }

    def annotate_frame(self, im0, masks, track_ids):
        annotator = Annotator(im0, line_width=2)
        for mask, track_id in zip(masks, track_ids):
            annotator.seg_bbox(mask=mask, mask_color=self.get_color(track_id))
        return annotator

    def get_color(self, track_id):
        if track_id not in self.color_map:
            self.color_map[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return self.color_map[track_id]

    def run(self):
        cap, out, total_frames = initialize_video_capture(self.video_path, self.result_path)
        frame_counter = 0

        while True:
            ret, im0 = cap.read()
            if not ret:
                break

            frame_counter += 1
            print(f"Tracking frame: {frame_counter}/{total_frames}")

            annotator = self.process_frame(im0)
            if annotator:
                write_annotated_video(annotator, out)

        out.release()
        cap.release()
        cv2.destroyAllWindows()
        track_data = get_tracked_objects(self.track_history)
        return {
            "video_path": self.result_path,
            "track_data": track_data
        }
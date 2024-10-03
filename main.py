import os
from engines.tracker import YOLOTracker
from utils.request_handler import get_video, put_video, put_image
import cv2

def process_video(video_id):
    response = {}
    video_path = get_video(video_id)
    tracker = YOLOTracker('multiplant.pt', video_path)
    result = tracker.run()

    status = put_video(result['video_path'])
    if status.status_code == 200:
        print(status.text)
        response['video_id'] = status.json()['unique_id']
        os.remove(video_path)
        os.remove('temp_result.mp4')

    for data in result['track_data']:
        temp_image_path = 'temp.jpg'
        cv2.imwrite(temp_image_path, data['image'])
        status = put_image(temp_image_path)
        if status.status_code == 200:
            data['image'] = status.json()['image_id']
            os.remove(temp_image_path)
    response['track_data'] = result['track_data']

    return response

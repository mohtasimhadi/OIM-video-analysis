import socket
from fastapi import FastAPI, APIRouter
import os
import cv2
from engines.tracker import YOLOTracker
from utils.request_handler import get_video, put_video, put_image

router = APIRouter()
app = FastAPI()

@router.post('/{video_id}')
async def video_processing(video_id: str):
    response = {}
    video_path = get_video(video_id)
    tracker = YOLOTracker('multiplant.pt', video_path)
    result = tracker.run()

    status = put_video(result['video_path'])
    if status.status_code == 200:
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

app.include_router(router)

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

if __name__ == "__main__":
    import uvicorn
    local_ip = get_local_ip()
    print(f"App is accessible at: http://{local_ip}:5000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
import requests
from utils.config import Config

def get_video(video_id):
    url = Config.CDN_URI + Config.GET_VIDEO + video_id
    output_path = 'input_video.mp4'
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as video_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    video_file.write(chunk)
        return output_path
    return None

def put_video(video_path):
    url = Config.CDN_URI + Config.PUT_VIDEO
    with open(video_path, 'rb') as video_file:
        file = {'file': video_file}
        response = requests.post(url, files=file)
    return {
        'status_code': response.status_code,
        'text': response.text
    }
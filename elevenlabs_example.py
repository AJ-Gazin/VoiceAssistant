import requests
import tempfile
from dotenv import load_dotenv
import os
from playsound import playsound

load_dotenv()

def get_elevenlabs_speech(api_key, text, voice_id):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    print("url is", url)
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to get speech: {response.status_code} {response.text}")

def play_speech(audio_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file_path = temp_audio_file.name

    playsound(temp_audio_file_path)

    os.remove(temp_audio_file_path)

def main():
    api_key = os.getenv("Elevenlabs_API_KEY")
    voice_id = os.getenv("Elevenlabs_VOICE_ID")
    print("VOICE is", voice_id)
    text = "Hello, this is a test message"
    
    try:
        audio_data = get_elevenlabs_speech(api_key, text, voice_id)
        play_speech(audio_data)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
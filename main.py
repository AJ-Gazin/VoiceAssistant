import pyaudio
import whisper
import requests
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import wave
import os

# Configuration
Ollama_API_URL = "http://127.0.0.1:11434/"  # Ollama Llama3 server URL
Elevenlabs_API_KEY = os.getenv("Elevenlabs_API_KEY")
Custom_Voice_ID = os.getenv("Elevenlabs_VOICE_ID")
START_PHRASE = "hey assistant"

def query_llama_model(prompt):
    payload = {"prompt": prompt}
    response = requests.post(Ollama_API_URL, json=payload)
    if response.status_code == 200:
        return response.json()["text"]
    else:
        raise Exception("Error querying Llama model: ", response.text)

def text_to_speech(text, voice_id):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Elevenlabs_API_KEY}"
    }
    payload = {
        "text": text,
        "voice": voice_id
    }
    response = requests.post(Elevenlabs_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.content  # Audio content
    else:
        raise Exception("Error with Elevenlabs TTS: ", response.text)

def list_audio_devices():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']}")
    p.terminate()

def record_audio(device_index, duration=5, filename="temp.wav"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)

    print("* Recording audio...")

    frames = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* Done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(filename="temp.wav"):
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    return result['text']

def main():
    print("Listing available audio devices:")
    list_audio_devices()
    device_index = int(input("Enter the device index for your microphone: "))

    while True:
        record_audio(device_index=device_index, duration=5, filename="start_phrase.wav")
        transcript = transcribe_audio("start_phrase.wav")
        if START_PHRASE in transcript.lower():
            print("Start phrase detected!")
            record_audio(device_index=device_index, duration=10, filename="full_context.wav")  # Adjust duration as needed
            full_transcript = transcribe_audio("full_context.wav")
            print(f"Full context: {full_transcript}")
            response = query_llama_model(full_transcript)
            print(f"Llama response: {response}")
            audio_content = text_to_speech(response, Custom_Voice_ID)
            audio = AudioSegment.from_file(BytesIO(audio_content), format="mp3")
            play(audio)

if __name__ == "__main__":
    main()
import torch
import whisper
import requests
import time
import os
from gtts import gTTS
from pydub import AudioSegment
import simpleaudio as sa
import pyaudio
import wave
import Levenshtein

# Configuration
Ollama_API_URL = "http://127.0.0.1:11434/api/generate"
START_PHRASE = "hey assistant"
DEFAULT_DEVICE_INDEX = 1
LEVENSHTEIN_THRESHOLD = 2  # Maximum distance to consider a match

def query_llama_model(prompt):
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    print(f"Sending prompt to Llama3: {prompt}")
    start_time = time.time()
    response = requests.post(Ollama_API_URL, json=payload)
    response_time = time.time() - start_time
    if response.status_code == 200:
        response_text = response.json().get("response", "")
        print(f"Received response from Llama3 in {response_time:.2f} seconds: {response_text}")
        return response_text
    else:
        print(f"Failed to query Llama3. Status code: {response.status_code}, Response: {response.text}")
        raise Exception("Error querying Llama model: ", response.text)

def text_to_speech(text):
    print(f"Converting text to speech: {text}")
    tts = gTTS(text)
    tts.save('response.mp3')
    if os.path.exists('response.mp3') and os.path.getsize('response.mp3') > 0:
        print("response.mp3 generated successfully.")
    else:
        print("Failed to generate response.mp3.")

def play_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path, format="mp3")
        play_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
        play_obj.wait_done()
        print("Audio playback finished.")
    except Exception as e:
        print(f"Failed to play audio file {file_path}: {e}")

def record_audio(device_index, filename="temp.wav", duration=10):
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
    print(f"Audio recorded and saved to {filename}")

def transcribe_audio(filename="temp.wav"):
    print(f"Transcribing audio file: {filename}")
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for Whisper model.")
        model = whisper.load_model("base").to("cuda")
    else:
        print("CUDA is not available. Using CPU for Whisper model.")
        model = whisper.load_model("base")
    result = model.transcribe(filename)
    print(f"Transcription result: {result['text']}")
    return result['text']

def is_start_phrase_detected(transcript, start_phrase, threshold):
    transcript_words = transcript.lower().split()
    start_phrase_words = start_phrase.split()
    
    for i in range(len(transcript_words) - len(start_phrase_words) + 1):
        phrase_segment = transcript_words[i:i+len(start_phrase_words)]
        distances = [Levenshtein.distance(word, start_phrase_word) for word, start_phrase_word in zip(phrase_segment, start_phrase_words)]
        if all(distance <= threshold for distance in distances):
            return True
    return False

def detect_start_phrase_and_context():
    record_audio(device_index=DEFAULT_DEVICE_INDEX, filename="start_phrase.wav", duration=10)
    transcript = transcribe_audio("start_phrase.wav")
    if is_start_phrase_detected(transcript, START_PHRASE, LEVENSHTEIN_THRESHOLD):
        print("Start phrase detected!")
        return transcript
    else:
        return None

def main():
    print("Using default audio device index:", DEFAULT_DEVICE_INDEX)

    while True:
        full_transcript = detect_start_phrase_and_context()
        if full_transcript:
            print(f"Full context: {full_transcript}")
            response = query_llama_model(full_transcript)
            if response:
                print(f"Llama response: {response}")
                text_to_speech(response)
                if os.path.exists("response.mp3") and os.path.getsize("response.mp3") > 0:
                    print("Playing audio response...")
                    play_audio("response.mp3")
                else:
                    print("Failed to generate response.mp3 or file is empty.")
            else:
                print("No response from Llama3.")
        else:
            print("Failed to detect start phrase. Retrying...")

if __name__ == "__main__":
    main()

import json
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
import io
import re

# Configuration
Ollama_API_URL = "http://127.0.0.1:11434/api/generate"
START_PHRASE = "hey assistant"
DEFAULT_DEVICE_INDEX = 1
LEVENSHTEIN_THRESHOLD = 2  # Maximum distance to consider a match
BUFFER_THRESHOLD = 100  # Minimum number of characters to buffer before yielding
PUNCTUATION_REGEX = re.compile(r'[.!?]')


def query_llama_model(prompt, chunk_size=1024):
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": True,
        "chunk_size": chunk_size
    }
    print(f"Sending prompt to Llama3: {prompt}")
    start_time = time.time()
    response = requests.post(Ollama_API_URL, json=payload, stream=True)
    response_time = time.time() - start_time

    if response.status_code == 200:
        print(f"Received response from Llama3 in {response_time:.2f} seconds.")
        buffer = ""
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:
                json_data = json.loads(chunk)
                response_text = json_data.get("response", "")
                buffer += response_text
                if PUNCTUATION_REGEX.search(buffer) or json_data.get("done", False):
                    last_punctuation = max((loc for loc, val in enumerate(buffer) if PUNCTUATION_REGEX.match(val)), default=-1)
                    if last_punctuation != -1:
                        yield buffer[:last_punctuation + 1]
                        buffer = buffer[last_punctuation + 1:]
                    else:
                        yield buffer
                        buffer = ""
        if buffer:
            yield buffer
    else:
        print(f"Failed to query Llama3. Status code: {response.status_code}, Response: {response.text}")
        raise Exception("Error querying Llama model: ", response.text)

def text_to_speech(text):
    if not text.strip():
        return None
    print(f"Converting text to speech: {text}")
    tts = gTTS(text)
    mp3_data = io.BytesIO()
    tts.write_to_fp(mp3_data)
    mp3_data.seek(0)
    audio = AudioSegment.from_file(mp3_data, format="mp3")
    return audio.raw_data

def play_audio_stream(audio_generator):
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=24000,
                        output=True)

        for chunk in audio_generator:
            if chunk is not None:
                stream.write(chunk)
                #time.sleep(0.1)  # Add a small delay between audio chunks

        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Audio playback finished.")
    except Exception as e:
        print(f"Failed to play audio stream: {e}")


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
            response_generator = query_llama_model(full_transcript)
            audio_generator = (text_to_speech(chunk) for chunk in response_generator)
            print("Playing audio response...")
            play_audio_stream(audio_generator)
        else:
            print("Failed to detect start phrase. Retrying...")

if __name__ == "__main__":
    main()
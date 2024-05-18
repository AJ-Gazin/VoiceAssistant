from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os
load_dotenv()

client = ElevenLabs(
  api_key=os.getenv("Elevenlabs_API_KEY"), # Defaults to ELEVEN_API_KEY
)

response = client.voices.get_all()
audio = client.generate(text="Hello there!", voice=response.voices[0])
print(response.voices)
import requests

def test_ollama_server(ollama_api_url):
    try:
        # Define a simple test prompt
        test_prompt = "Hello, how are you?"
        
        # Prepare the payload
        payload = {
            "model": "llama3",  # Specify the model you are using
            "prompt": test_prompt,
            "stream": False  # Ensure the response is returned as a single object
        }
        
        # Define the endpoint for the Ollama server
        endpoint = "/api/generate"
        
        # Send a request to the Ollama server
        response = requests.post(ollama_api_url + endpoint, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            response_json = response.json()
            # Print the response text
            print("Ollama server response:", response_json.get("response", "No text in response"))
        else:
            # Print an error message if the request was not successful
            print(f"Failed to query Ollama server. Status code: {response.status_code}")
            print("Response:", response.text)
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace with your Ollama server API URL
OLLAMA_API_URL = "http://127.0.0.1:11434"

# Test the Ollama server
test_ollama_server(OLLAMA_API_URL)

import os
from dotenv import load_dotenv
from google import genai
# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents="Explain artificial intelligence in five simple sentences.",
    )
print("\nGemini Response:\n")
print(response.text)


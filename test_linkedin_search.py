"""Test LinkedIn search accuracy."""

from google import genai
from config import Config

client = genai.Client(api_key=Config.GEMINI_API_KEY)

prompt = """Search for Anantha Chandrakasan LinkedIn profile using site:linkedin.com/in

Return ONLY the exact URL if found, otherwise say NOT FOUND."""

response = client.models.generate_content(
    model=Config.MODEL_TEXT,
    contents=prompt,
    config={"tools": [{"google_search": {}}]},
)
print("Search result:", response.text)

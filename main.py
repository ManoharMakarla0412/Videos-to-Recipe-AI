import os
import ssl
import uuid
import whisper
import openai
import yt_dlp
import json
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fix for macOS SSL errors (if any)
ssl._create_default_https_context = ssl._create_unverified_context

# Load Whisper model
model = whisper.load_model("base")
print("Whisper model loaded.")

# OpenAI API Key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

app = FastAPI()
DOWNLOADS_DIR = "downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

def download_audio_from_youtube(url: str) -> str:
    file_id = str(uuid.uuid4())
    output_template = os.path.join(DOWNLOADS_DIR, f"{file_id}.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return os.path.join(DOWNLOADS_DIR, f"{file_id}.mp3")

def extract_recipe(transcript: str) -> dict:
    prompt = f"""
You are a recipe extractor AI. Based on the following cooking video transcript, extract a clean and structured recipe with:

- Recipe Name
- Ingredients (as a list)
- Instructions (as steps)
- Estimated Time (if mentioned)
- Cuisine Type (if clear)

Transcript:
\"\"\"{transcript}\"\"\"

Respond only in valid JSON format like:
{{
  "recipe_name": "",
  "ingredients": ["", ""],
  "instructions": ["", ""],
  "estimated_time": "",
  "cuisine_type": ""
}}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    # Parse the JSON response content into a dictionary
    raw_content = response['choices'][0]['message']['content']
    recipe_dict = json.loads(raw_content)

    return recipe_dict

@app.post("/youtube-to-recipe/")
async def youtube_to_recipe(url: str = Form(...)):
    try:
        audio_path = download_audio_from_youtube(url)
        result = model.transcribe(audio_path)
        transcript = result['text']
        recipe_data = extract_recipe(transcript)

        # Remove the downloaded audio file
        os.remove(audio_path)

        # Clean response with only sorted info
        return JSONResponse(content={
            "recipe_name": recipe_data.get("recipe_name"),
            "estimated_time": recipe_data.get("estimated_time"),
            "cuisine_type": recipe_data.get("cuisine_type"),
            "ingredients": recipe_data.get("ingredients", []),
            "instructions": recipe_data.get("instructions", [])
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

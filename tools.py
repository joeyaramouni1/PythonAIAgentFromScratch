from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import pyttsx3
import openai
import os
from PIL import Image
import requests
from io import BytesIO

def save_to_txt(data: str, filename: str = "output/research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

def speak_text(data: str) -> str:
    """Speak the given text aloud using TTS."""
    engine = pyttsx3.init()
    engine.say(data)
    engine.runAndWait()
    return "Spoken out loud using TTS."

def draw_image(summary: str, filename: str = "output/generated_image.png") -> str:
    """
    Uses OpenAI to generate an image representing the summary of the topic.
    """
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Use summary to prompt image generation
    response = client.images.generate(
        model="dall-e-3",
        prompt=summary,
        size="1024x1024",
        n=1,
    )

    image_url = response.data[0].url

    # Download and save image
    image = Image.open(BytesIO(requests.get(image_url).content))
    image.save(filename)

    return f"Image generated and saved to {filename} based on summary."

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

speak_tool = Tool(
    name="speak_text_aloud",
    func=speak_text,
    description="Reads the research result aloud using text-to-speech.",
)

draw_image_tool = Tool(
    name="draw_image_from_summary",
    func=draw_image,
    description="Generates an image that visually represents the summary of the research topic using DALL·E.",
)

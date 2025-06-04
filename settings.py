from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GEMINI_MODEL_PRO: str = "gemini-2.5-pro-exp-03-25"
    GOOGLE_GEMINI_API_KEY: str = os.getenv("GOOGLE_GEMINI_API_KEY")

    TOGETHER_API_KEY: str = os.getenv("TOGETHERAI_API_KEY")
    TOGETHER_BASE_URL: str = "https://api.together.xyz/v1"

    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK")
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

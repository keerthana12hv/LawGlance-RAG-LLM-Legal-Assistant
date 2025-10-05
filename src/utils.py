# src/utilities.py
import os
from dotenv import load_dotenv

def load_env(dotenv_path=".env"):
    """
    Loads environment variables from a .env file or environment.
    Works in local, Colab, and Render deployments.
    """
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path, override=True)
        print(f"✅ Environment variables loaded from {dotenv_path}")
    elif os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"):
        print("🔗 Environment variables already set in system.")
    else:
        print("⚠️ No .env file found and no API key detected.")

    # Return env variables for optional use
    return {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "HUGGINGFACEHUB_API_TOKEN": os.getenv("HUGGINGFACEHUB_API_TOKEN")
    }

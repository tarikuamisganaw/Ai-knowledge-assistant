import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

# Load environment
load_dotenv()
client = genai.Client()  # Auto-uses GOOGLE_API_KEY

MODEL = "gemini-3-flash-preview"
MAX_HISTORY = 6  # Keep last 6 messages (3 user + 3 AI) to save tokens

print("🤖 Gemini CLI Chat (official style). Type 'quit' to exit.\n")
history = []  # Stores {"role": "user"|"model", "parts": [{"text": "..."}]}

while True:
    user_input = input("You: ").strip()
    
    # Exit
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!"); break
    if not user_input:
        continue

    # Add user message in official format
    history.append({"role": "user", "parts": [{"text": user_input}]})
    
    # Trim history to avoid context limits
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    print("Gemini is thinking...")
    
    try:
        # Send history in the format Gemini expects
        response = client.models.generate_content(
            model=MODEL,
            contents=history  # List of {role, parts} dicts
        )
        
        ai_text = response.text
        print(f"Gemini: {ai_text}\n")
        
        # Save AI response to history
        history.append({"role": "model", "parts": [{"text": ai_text}]})
        
    except ClientError as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            # Try to extract retry delay
            import re
            match = re.search(r"retry in ([\d.]+)s", error_msg)
            wait = float(match.group(1)) + 1 if match else 25
            print(f"Rate limited. Waiting {wait:.0f}s before retry...")
            time.sleep(wait)
            # Remove the failed user turn so history stays clean
            history.pop()
            continue
        else:
            print(f"API Error: {e}")
            history.pop()
    except Exception as e:
        print(f"Error: {e}")
        history.pop()
import os
import openai
from dotenv import load_dotenv

# Load API key from environment variables

ROUTER_API_KEY = "sk-hOxi35qHSFSTxm8RUqylsCWwlBkxSzE+LChKJu637L6sTJG87GMq3D1cCE8bTulpAr3kI9JoKukhhnkkIzUNaVEY0TarPkA15vnYk4pQ9AM="

if ROUTER_API_KEY is None:
    raise ValueError("ROUTER_API_KEY not found. Please check your .env file.")

try:
    # Initialize OpenAI client
    client = openai.OpenAI(
        api_key=ROUTER_API_KEY,
        base_url="https://router.requesty.ai/v1",
        default_headers={"Authorization": f"Bearer {ROUTER_API_KEY}"}
    )

    # Example request
    response = client.chat.completions.create(
        model="google/gemini-2.0-pro-exp-02-05",
        messages=[{"role": "user", "content": "Hello, who are you?"}]
    )

    # Check if the response is successful
    if not response.choices:
        raise Exception("No response choices found.")

    # Print the result
    print(response.choices[0].message.content)

except openai.OpenAIError as e:
    print(f"OpenAI API error: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
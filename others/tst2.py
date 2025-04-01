import os
import sys
from groq import Groq

# Initialize client for conversation responses
client = Groq(
    api_key='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'
)

def main():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
    print("Type 'exit' or 'quit' to end the chat.")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        try:
            # Get response from Groq
            response = client.chat.completions.create(
                messages=messages,
                model="deepseek-r1-distill-qwen-32b",
            )
            bot_response = response.choices[0].message.content

            # Add bot response to history
            messages.append({"role": "assistant", "content": bot_response})

            # Print response
            print(f"\nGroq: {bot_response}")

        except Exception as e:
            print(f"An error occurred: {e}")
            continue

if __name__ == "__main__":
    main()
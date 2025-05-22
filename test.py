import google.generativeai as genai
import os

# Configure the Gemini API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=GOOGLE_API_KEY)

# List available models
print("Available models for your API key:")
for m in genai.list_models():
    print(f"- {m.name} (generation: {getattr(m, 'generation_methods', None)})")

try:
    # Initialize the model
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    
    print("Sending request to Gemini model...")

    # Generate content
    response = model.generate_content("Hello! Write a short, friendly greeting.")

    # Print the response
    print("\n--- Gemini's Response ---")
    print(response.text)
    print("-------------------------\n")

    # Print additional information
    print(f"Model used: {model.model_name}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
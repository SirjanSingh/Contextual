"""Quick test script to check Google API response format."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import google.generativeai as genai

# Get API key from environment variable
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=api_key)

# Test single embedding
print("Testing single text embedding...")
result = genai.embed_content(
    model="models/text-embedding-004",
    content="Hello world",
    task_type="RETRIEVAL_DOCUMENT",
)

print(f"\nResult type: {type(result)}")
print(f"\nIs dict: {isinstance(result, dict)}")
if isinstance(result, dict):
    print(f"Dict keys: {list(result.keys())}")
    if 'embedding' in result:
        print(f"Embedding length: {len(result['embedding'])}")
        print(f"First 5 values: {result['embedding'][:5]}")

# Test batch embedding
print("\n\nTesting batch embedding...")
result_batch = genai.embed_content(
    model="models/text-embedding-004",
    content=["Hello", "World"],
    task_type="RETRIEVAL_DOCUMENT",
)

print(f"\nBatch result type: {type(result_batch)}")
print(f"Is dict: {isinstance(result_batch, dict)}")
if isinstance(result_batch, dict):
    print(f"Dict keys: {list(result_batch.keys())}")

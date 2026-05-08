"""Quick script to list available Gemini models."""
from google import genai
from app.config import get_config

config = get_config()
client = genai.Client(api_key=config.google_api_key)

print("Available Gemini models:\n")
try:
    models = client.models.list()
    for model in models:
        if hasattr(model, 'name'):
            print(f"  - {model.name}")
            if hasattr(model, 'supported_generation_methods'):
                print(f"    Methods: {model.supported_generation_methods}")
        elif hasattr(model, 'model_id'):
            print(f"  - {model.model_id}")
except Exception as e:
    print(f"Error listing models: {e}")
    print("\nTrying alternative approach...")
    
    # Try some common model names
    test_models = [
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-001", 
        "gemini-1.5-flash",
        "gemini-1.5-pro-002",
        "gemini-1.5-pro-001",
        "gemini-1.5-pro",
        "gemini-pro",
    ]
    
    print("\nTesting common model names:")
    for model_name in test_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents="test"
            )
            print(f"  ✓ {model_name} - WORKS")
            break
        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "NOT_FOUND" in error_str:
                print(f"  ✗ {model_name} - NOT FOUND")
            else:
                print(f"  ? {model_name} - ERROR: {e}")

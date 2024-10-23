import os
from openai import OpenAI
import requests
import argparse

def get_openai_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE")
    )

def generate_image(prompt: str, size: str = "1024x1024") -> str:
    client = get_openai_client()
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            n=1,
        )
        return response.data[0].url
    finally:
        client.close()

def save_image(image_url: str, filename: str):
    img_data = requests.get(image_url).content
    with open(filename, 'wb') as handler:
        handler.write(img_data)

def main():
    parser = argparse.ArgumentParser(description="Generate an image using DALL-E 3")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt for the image")
    parser.add_argument("--output", type=str, required=True, help="The output filename")
    args = parser.parse_args()

    image_url = generate_image(args.prompt)
    print(f"Image generated: {image_url}")
    
    save_image(image_url, args.output)
    print(f"Image saved as {args.output}")

if __name__ == "__main__":
    main()

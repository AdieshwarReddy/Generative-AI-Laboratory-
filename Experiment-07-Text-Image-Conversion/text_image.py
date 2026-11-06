
# Generative AI Laboratory
# Experiment 7: Text-to-Image and Image-to-Text

import os
import argparse
import shutil

import torch
from PIL import Image

from diffusers import StableDiffusionPipeline
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)


# ---------------------------------------
# Step 1: Configuration
# ---------------------------------------

os.makedirs("outputs", exist_ok=True)

TEXT_TO_IMAGE_MODEL = (
    "runwayml/stable-diffusion-v1-5"
)

IMAGE_TO_TEXT_MODEL = (
    "Salesforce/blip-image-captioning-base"
)

DEFAULT_PROMPT = (
    "A beautiful futuristic city at sunset, "
    "cinematic lighting, detailed architecture"
)


# ---------------------------------------
# Step 2: Select Device
# ---------------------------------------

device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("GENERATIVE AI LAB - EXPERIMENT 7")
print("Text-to-Image and Image-to-Text Conversion")
print("\nSelected Device:", device)


# ---------------------------------------
# Step 3: Text-to-Image Function
# ---------------------------------------

def text_to_image(prompt):

    print("\nTEXT-TO-IMAGE GENERATION")
    print("Prompt:", prompt)

    print("\nLoading Stable Diffusion model...")

    dtype = (
        torch.float16 if device == "cuda"
        else torch.float32
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        TEXT_TO_IMAGE_MODEL,
        torch_dtype=dtype
    )

    pipe = pipe.to(device)

    pipe.enable_attention_slicing()

    print("Generating image...")

    generator = torch.Generator(
        device=device
    ).manual_seed(42)

    image = pipe(
        prompt=prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        height=512,
        width=512,
        generator=generator
    ).images[0]

    output_path = (
        "outputs/generated_image.png"
    )

    image.save(output_path)

    print("Image saved:", output_path)

    # Release model memory before captioning
    del pipe

    if device == "cuda":
        torch.cuda.empty_cache()

    return output_path


# ---------------------------------------
# Step 4: Image-to-Text Function
# ---------------------------------------

def image_to_text(image_path):

    print("\nIMAGE-TO-TEXT CONVERSION")
    print("Input Image:", image_path)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print("\nLoading BLIP model...")

    processor = BlipProcessor.from_pretrained(
        IMAGE_TO_TEXT_MODEL
    )

    model = (
        BlipForConditionalGeneration.from_pretrained(
            IMAGE_TO_TEXT_MODEL
        )
    )

    model = model.to(device)
    model.eval()

    # Load image
    image = Image.open(
        image_path
    ).convert("RGB")

    # Save a copy of the input image
    input_copy = "outputs/input_image.png"
    image.save(input_copy)

    # Preprocess image
    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    print("Generating caption...")

    # Generate caption
    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=50
        )

    # Convert token IDs into text
    caption = processor.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    print("\nGenerated Caption:")
    print(caption)

    # Save caption
    caption_path = (
        "outputs/generated_caption.txt"
    )

    with open(
        caption_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(caption + "\n")

    print("\nCaption saved:", caption_path)

    return caption


# ---------------------------------------
# Step 5: Main Function
# ---------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Text-to-Image and Image-to-Text "
            "Conversion using Generative AI"
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "text-to-image",
            "image-to-text",
            "both"
        ],
        default="both",
        help="Select execution mode"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text prompt for image generation"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Input image path for captioning"
    )

    args = parser.parse_args()

    # Text-to-image
    if args.mode == "text-to-image":

        text_to_image(args.prompt)

    # Image-to-text
    elif args.mode == "image-to-text":

        if args.image is None:

            parser.error(
                "--image is required for "
                "image-to-text mode"
            )

        image_to_text(args.image)

    # Both tasks
    elif args.mode == "both":

        generated_path = text_to_image(
            args.prompt
        )

        caption_input = (
            args.image
            if args.image is not None
            else generated_path
        )

        image_to_text(caption_input)

    print(
        "\nExperiment 7 completed successfully!"
    )


# ---------------------------------------
# Step 6: Execute Program
# ---------------------------------------

if __name__ == "__main__":
    main()

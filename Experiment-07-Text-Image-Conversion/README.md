
# Experiment 7: Text-to-Image and Image-to-Text Conversion

## Aim

To implement text-to-image generation and image-to-text captioning using Python, PyTorch, and pretrained deep learning models.

## Objectives

1. Understand multimodal Generative AI.
2. Convert natural language text into images.
3. Generate descriptive captions from images.
4. Use pretrained Stable Diffusion and BLIP models.
5. Understand diffusion-based image generation.
6. Understand image captioning using vision-language models.
7. Save and visualize the generated outputs.

## Theory

### What is Multimodal Generative AI?

Multimodal Generative AI refers to AI systems that process or generate information across multiple modalities, such as:

- Text
- Images
- Audio
- Video

This experiment focuses on two tasks:

1. Text-to-Image Generation
2. Image-to-Text Captioning

## 1. Text-to-Image Generation

Text-to-image generation converts a natural language description into an image.

Example:

Input:
"A futuristic city at sunset"

Output:
An AI-generated image representing a futuristic city.

### Model Used

Stable Diffusion v1.5

Stable Diffusion is a latent diffusion model that generates images from text prompts.

### Working Principle

1. Accept a text prompt.
2. Convert the prompt into text embeddings.
3. Initialize random latent noise.
4. Apply iterative denoising guided by the text.
5. Decode the final latent representation into an image.
6. Save the generated image.

## 2. Image-to-Text Conversion

Image-to-text conversion generates a natural language description of an image.

Example:

Input:
An image of a dog playing in a park.

Output:
"A dog playing on the grass."

### Model Used

BLIP (Bootstrapping Language-Image Pre-training)

BLIP is a vision-language model that can generate captions describing image content.

### Working Principle

1. Load an input image.
2. Preprocess the image.
3. Extract visual information using the model.
4. Generate text tokens describing the image.
5. Decode the tokens into a readable caption.
6. Save the generated caption.

## System Architecture

```text
             USER INPUT
                 |
         +-------+-------+
         |               |
         v               v
     TEXT PROMPT      INPUT IMAGE
         |               |
         v               v
   Stable Diffusion      BLIP
         |               |
         v               v
   Diffusion-Based    Image Encoding
   Image Generation        |
         |               v
         |          Text Generation
         v               |
   GENERATED IMAGE        v
                     IMAGE CAPTION
         |               |
         +-------+-------+
                 |
                 v
            SAVE OUTPUTS
```

## Mathematical Concepts

### Diffusion Forward Process

q(x_t | x_0) =
N(sqrt(alpha_bar_t) * x_0,
  (1 - alpha_bar_t) * I)

Where:

- x_0 = Original sample
- x_t = Noisy sample at timestep t
- alpha_bar_t = Cumulative noise schedule
- I = Identity covariance matrix

### Reverse Diffusion

The model progressively estimates and removes noise to generate an image.

A neural network predicts noise using:

epsilon_theta(x_t, t, c)

Where:

- x_t = Current noisy latent
- t = Diffusion timestep
- c = Text conditioning information

### Image Captioning

The captioning model estimates:

P(y | I)

Where:

- I = Input image
- y = Generated caption

It generates text tokens conditioned on the visual information.

## Models Used

### Stable Diffusion

Model:
runwayml/stable-diffusion-v1-5

Purpose:
Text-to-image generation

### BLIP

Model:
Salesforce/blip-image-captioning-base

Purpose:
Image-to-text caption generation

Both models are pretrained. This experiment performs inference and does not train them from scratch.

## Software Requirements

- Python 3.10 or 3.11
- PyTorch
- Diffusers
- Transformers
- Accelerate
- Pillow
- Safetensors

Recommended Hardware:

- NVIDIA GPU with compatible CUDA drivers
- Approximately 8 GB or more GPU VRAM for comfortable execution
- Sufficient disk space for pretrained models

CPU execution is possible but text-to-image generation can be extremely slow.

## Algorithm

Step 1: Import the required Python libraries.

Step 2: Select GPU or CPU.

Step 3: Load the Stable Diffusion pipeline.

Step 4: Accept a text prompt.

Step 5: Generate an image from the prompt.

Step 6: Save the generated image.

Step 7: Load the BLIP image captioning model.

Step 8: Open the generated image or another input image.

Step 9: Preprocess the image.

Step 10: Generate a descriptive caption.

Step 11: Decode the generated tokens.

Step 12: Display and save the caption.

## Installation

```bash
python -m pip install -r requirements.txt
```

For NVIDIA GPU acceleration, install the appropriate PyTorch CUDA build following the official PyTorch installation instructions.

## Execution

Run both tasks:

```bash
python text_image.py --mode both
```

Run only image captioning:

```bash
python text_image.py --mode image-to-text --image sample.jpg
```

Run only text-to-image generation:

```bash
python text_image.py --mode text-to-image --prompt "A beautiful mountain landscape"
```

## Expected Output

The program displays:

- Selected device
- Text prompt
- Generated image path
- Generated image caption
- Caption file path

Generated files:

```text
outputs/generated_image.png
outputs/input_image.png
outputs/generated_caption.txt
```

The input_image.png file is a copy of the image supplied to the captioning model.

## Applications

1. AI image generation
2. Automatic image captioning
3. Accessibility tools
4. Digital content creation
5. Educational applications
6. Visual storytelling
7. Multimodal AI assistants

## Limitations

- Large pretrained models require significant memory.
- Image generation may be slow without a GPU.
- Generated images may contain visual inaccuracies.
- Image captions may omit details or hallucinate objects.
- Model downloads require internet access.
- Results vary depending on prompts and generation settings.

## Result

Text-to-image generation and image-to-text captioning were implemented using pretrained Stable Diffusion and BLIP models.

The program generates images from natural language prompts and produces descriptive captions from images.

## Conclusion

This experiment demonstrates the use of multimodal Generative AI for converting between textual and visual information.

Stable Diffusion enables text-conditioned image generation, while BLIP enables automatic image captioning.

# Judger - Content Moderation System

A comprehensive content moderation system for detecting and filtering inappropriate text and image content. This system provides multiple detection methods using various APIs and models.

## Project Structure

```
jailbreak_diffusion/judger/
├── pre_checker/              # Text content moderation components
│   ├── prompts/              # Prompt templates for text checking
│   ├── NSFW_text_classifier.py      # NSFW text classification
│   ├── NSFW_word_match.py           # Word matching for NSFW detection
│   ├── base.py                      # Base classes for text checkers
│   ├── composite.py                # Composite checker combining multiple methods
│   ├── detoxify.py                  # Detoxify-based toxic content detection
│   ├── distilbert_nsfw_text_checker.py    # DistilBERT implementation
│   ├── distilroberta_nsfw_text_checker.py # DistilRoBERTa implementation
│   ├── google_text_checker.py      # Google Natural Language API integration
│   ├── gpt.py                       # OpenAI GPT-based text checking
│   ├── gpt_4o.py                    # OpenAI GPT-4o text checking
│   ├── gpt_4o_mini.py               # OpenAI GPT-4o Mini text checking
│   ├── llama_guard.py               # Llama Guard-based safety checking
│   ├── nvidia_aegis.py              # NVIDIA NeMo Guardrails/Aegis integration
│   ├── openai_text_moderation.py    # OpenAI moderation API for text
│   ├── azure_text_checker.py       # Azure Content Safety integration
│   └── perspective_api.py         # Google Perspective API integration
└── post_checker/             # Image content moderation components
    ├── checkpoints/           # Model checkpoints for various detectors
    │   ├── finetuned_q16/     # Finetuned Q16 checkpoints
    │   ├── multi-headed/      # Multi-head detector checkpoints
    │   └── q16/               # Q16 checkpoints
    ├── prompts/               # Prompt templates for LLM-based detection
    ├── MultiheadDetector.py   # Multi-category detection system
    ├── Q16.py                 # Q16-based content detection
    ├── SD_safety_checker.py   # Stable Diffusion safety checker
    ├── base.py                # Base classes for image checkers
    ├── aws_image_checker.py   # AWS Rekognition integration
    ├── azure_image_checker.py # Azure Content Safety integration
    ├── google_image_checker.py # Google Vision AI integration
    ├── gpt_4o.py              # OpenAI GPT-4o based image checking
    ├── gpt_4o_mini.py         # OpenAI GPT-4o Mini based image checking
    ├── llava_guard.py         # LLaVA-based safety checking
    ├── lvm_checker.py         # Language-Vision Model checker
    └── openai_image_moderation.py # OpenAI moderation API for images
```

## Overview

Judger is a modular content moderation system designed to detect and filter inappropriate content in both text and images. It provides:

1. **Pre-checker**: Text content moderation before processing
2. **Post-checker**: Image content moderation after generation

The system integrates with multiple commercial APIs and open-source models to provide comprehensive content safety checks.

## Features

### Text Moderation (pre_checker)
- Word-based matching using curated NSFW word lists
- Neural model-based detection:
  - DistilBERT NSFW classifier (eliasalbouzidi/distilbert-nsfw-text-classifier)
  - NSFW text classifier (michellejieli/NSFW_text_classifier)
  - DistilRoBERTa NSFW prompt detector (AdamCodd/distilroberta-nsfw-prompt-stable-diffusion)
  - Detoxify for toxic content detection
- LLM-based detection:
  - OpenAI GPT models (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
  - Llama Guard
- Commercial API integration:
  - Google Natural Language API
  - Azure Content Safety
  - OpenAI Moderation API
  - Google Perspective API
  - NVIDIA Aegis/NeMo Guardrails

### Image Moderation (post_checker)
- Multi-headed detection for various categories:
  - Disturbing content
  - Hateful content
  - Political content
  - Sexual content
  - Violent content
- Model-based detection:
  - Q16 checker
  - Finetuned Q16 checker
  - Stable Diffusion safety checker
  - LLaVA Guard
  - LVM (Language-Vision Model) checker
- Vision-Language Models:
  - OpenAI GPT-4o and GPT-4o-mini with vision capabilities
- Commercial API integration:
  - AWS Rekognition
  - Azure Content Safety for images
  - Google Vision AI SafeSearch
  - OpenAI Moderation API for images

## Installation

The judger system is part of the JailbreakDiffusionBench framework. Please refer to the main [README.md](../../README.md) for installation instructions.

```bash
# Clone the repository
git clone https://github.com/JailbreakDiffusionBench/JailbreakDiffusionBench.git
cd JailbreakDiffusionBench

# Create conda environment
conda env create -f environment.yml
conda activate jdb
```

## Configuration

Create a `.env` file in the project root with the following API keys:

```
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# Azure
AZURE_CONTENT_SAFETY_ENDPOINT=your_azure_endpoint
AZURE_CONTENT_SAFETY_KEY=your_azure_key

# AWS
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region

# Perspective API
PERSPECTIVE_API_KEY=your_perspective_api_key
```

## Usage

### Basic Usage

#### Text Moderation

```python
from jailbreak_diffusion.judger.pre_checker import (
    NSFW_word_match_Checker,
    NSFW_text_classifier_Checker,
    CompositeChecker
)

# Single checker
word_checker = NSFW_word_match_Checker()
result = word_checker.check("Your text to moderate")
print(f"Text flagged: {result}")  # Returns True if NSFW detected

# Composite checker
composite_checker = CompositeChecker(methods=['word_match', 'model'])
result = composite_checker.check("Your text to moderate")
print(f"Text flagged: {result}")
```

#### Image Moderation

```python
from jailbreak_diffusion.judger.post_checker import MultiheadDetector, Q16Detector

# Multihead detector
image_checker = MultiheadDetector(device='cuda')
image_path = "path/to/image.jpg"

# Check single image
result = image_checker.check(image_path, return_scores=True)
print(f"Image flagged: {result['flagged']}, Score: {result['score']:.4f}")

# Check batch of images
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = image_checker.check(images, return_scores=True)
for i, res in enumerate(results):
    print(f"Image {i+1} - Flagged: {res['flagged']}, Score: {res['score']:.4f}")

# Q16 detector
q16_checker = Q16Detector(device='cuda')
result = q16_checker.check(image_path)
print(f"Image flagged: {result}")
```

### Advanced Usage

#### Using with AttackerFactory

The judger system is typically used through the `AttackerFactory`:

```python
from jailbreak_diffusion.attack import AttackerFactory
from jailbreak_diffusion.diffusion_model import DiffusionFactory

# Initialize a diffusion model
model = DiffusionFactory(
    model_name="stable-diffusion-3.5-medium",
    device="cuda"
)

# Initialize an attacker with safety detectors
attacker = AttackerFactory(
    attack_type="MMA",
    target_model=model,
    text_detector={"name": "NSFW_text_classifier"},
    image_detector={"name": "multihead_detector"}
)

# Run attack on a prompt
result = attacker("a landscape photograph")

# Check if attack was successful
print(f"Attack success: {result.success}")
print(f"NSFW text detected: {result.is_text_NSFW}")
print(f"NSFW image detected: {result.is_image_NSFW}")
```

#### Using Cloud APIs

```python
from jailbreak_diffusion.judger.post_checker import (
    AzureContentSafetyDetector,
    OpenAIImageDetector
)

# Azure Content Safety
azure_checker = AzureContentSafetyDetector()
result = azure_checker.check("path/to/image.jpg")
print(f"Image flagged: {result}")

# OpenAI Moderation API
openai_checker = OpenAIImageDetector()
result = openai_checker.check("path/to/image.jpg")
print(f"Image flagged: {result}")
```

#### Custom Text Detector Configuration

```python
from jailbreak_diffusion.judger.pre_checker import GPTChecker

# GPT-4o with custom prompt
gpt_checker = GPTChecker(
    model_name="gpt-4o",
    prompt_template="prompts/simple.md",
    threshold=0.5
)
result = gpt_checker.check("Your text to moderate")
print(f"Text flagged: {result}")
```

## Available Detectors

### Text Detectors (pre_checker)

| Detector Name | Class Name | Description |
|--------------|------------|-------------|
| `NSFW_word_match` | `NSFW_word_match_Checker` | Word-based NSFW detection |
| `NSFW_text_classifier` | `NSFW_text_classifier_Checker` | Neural model-based NSFW detection |
| `distilbert_nsfw_text_checker` | `distilbert_nsfw_text_checker` | DistilBERT-based detection |
| `distilroberta_nsfw_text_checker` | `distilroberta_nsfw_text_checker` | DistilRoBERTa-based detection |
| `gpt_4o` | `GPTChecker` | GPT-4o based detection |
| `gpt_4o_mini` | `GPTChecker` | GPT-4o Mini based detection |
| `llama_guard` | `LlamaGuardChecker` | Llama Guard based detection |
| `openai_text_moderation` | `OpenAITextDetector` | OpenAI Moderation API |
| `azure_text_detector` | `AzureTextDetector` | Azure Content Safety |
| `google_text_checker` | `GoogleTextModerator` | Google Natural Language API |
| `nvidia_aegis` | `NvidiaAegisChecker` | NVIDIA Aegis/NeMo Guardrails |
| `detoxify` | `DetoxifyChecker` | Detoxify-based detection |

### Image Detectors (post_checker)

| Detector Name | Class Name | Description |
|--------------|------------|-------------|
| `multihead_detector` | `MultiheadDetector` | Multi-category detection system |
| `q16_detector` | `Q16Detector` | Q16-based content detection |
| `finetuned_q16_detector` | `FinetunedQ16Detector` | Finetuned Q16 detector |
| `sd_safety_checker` | `SD_SafetyCheckerDetector` | Stable Diffusion safety checker |
| `openai_image_checker` | `OpenAIImageDetector` | OpenAI Moderation API |
| `azure_image_checker` | `AzureContentSafetyDetector` | Azure Content Safety |
| `llava_guard` | `LlavaGuardChecker` | LLaVA-based safety checking |
| `gpt_4o` | `GPT_4o_ImageChecker` | GPT-4o vision-based checking |
| `gpt_4o_mini` | `GPT_4o_mini_ImageChecker` | GPT-4o Mini vision-based checking |

## Model References

This project utilizes several pre-trained models:

- [eliasalbouzidi/distilbert-nsfw-text-classifier](https://huggingface.co/eliasalbouzidi/distilbert-nsfw-text-classifier)
- [michellejieli/NSFW_text_classifier](https://huggingface.co/michellejieli/NSFW_text_classifier)
- [AdamCodd/distilroberta-nsfw-prompt-stable-diffusion](https://huggingface.co/AdamCodd/distilroberta-nsfw-prompt-stable-diffusion)

## API Documentation

### Cloud Service APIs

- [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation)
- [Google Vision SafeSearch](https://cloud.google.com/vision/docs/detecting-safe-search)
- [Google Natural Language API](https://cloud.google.com/natural-language/docs/moderating-text)
- [Azure Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview)
- [AWS Rekognition](https://docs.aws.amazon.com/rekognition/)

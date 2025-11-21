# ASR and STT AI Notebook - Prompts & Outputs

This repository contains prompts and AI-generated outputs exploring Automatic Speech Recognition (ASR) and Speech-to-Text (STT) topics, with a focus on fine-tuning Whisper variants and related models.

## Repository Structure

This is primarily a collection of:
- Research prompts sent to AI assistants
- AI-generated responses and analyses
- Notes and insights from hands-on experimentation
- Technical documentation on ASR/STT model fine-tuning

## Content Organization

- **`prompts/to-run/`**: Queued prompts for future exploration
- **`prompts/processed/`**: Completed prompt-output pairs
- **`notes/`**: Consolidated insights and documentation
- **`data-preparation/`**: Data preparation workflows and scripts
- **`export/`**: Compiled PDF exports for offline reading

## Exported Materials

### Complete Guide
- [STT Fine-Tuning Guide (Complete)](export/first-questions/full/STT-Fine-Tuning-Guide.pdf) - Full compilation of all content

### Split by Topic (4 Parts)
1. [Book 1: Core Concepts](export/first-questions/parts/STT-Fine-Tuning-Guide-BOOK1.pdf)
2. [Book 2: Advanced Techniques](export/first-questions/parts/STT-Fine-Tuning-Guide-BOOK2.pdf)
3. [Book 3: Implementation Details](export/first-questions/parts/STT-Fine-Tuning-Guide-BOOK3.pdf)
4. [Book 4: Practical Applications](export/first-questions/parts/STT-Fine-Tuning-Guide-BOOK4.pdf)

### Podcast Format (NEW!)
Convert the entire notebook into an audio podcast for listening on the go. See [PODCAST-WORKFLOW.md](PODCAST-WORKFLOW.md) for details.

**Quick Start:**
```bash
cd scripts
export OPENROUTER_API_KEY='your-key-here'
./create-podcast.sh
```

Features:
- Natural-sounding TTS with Microsoft Edge neural voices
- SSML-enhanced speech with proper pacing and emphasis
- Individual audio files by topic or single concatenated podcast
- Free TTS (edge-tts), minimal cost for AI conversion (~$1-5)

## AI/Human Collaboration

Content in this repository is generated through:
- Prompts authored by Daniel with AI-generated responses
- Human-written notes refined by AI assistants
- Collaborative exploration of ASR/STT concepts

## Author

**Daniel Rosehill**
[danielrosehill.com](https://danielrosehill.com)

With assistance from Claude (Anthropic)
 
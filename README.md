# Diffusion Music Box

> AI-generated ambient music from noise

Generate endless, evolving ambient music by running a tiny diffusion model on a spectrogram and playing it back as audio.

## Stack
- Language: **python**
- numpy, sounddevice

## Features
- Real-time spectrogram-to-audio inverse transform
- Iterative diffusion on a 2D frequency-time image
- Seamless looping with crossfade between generated segments
- Interactive parameters: noise schedule, diffusion steps, tempo
- Headless CLI with MIDI-like control via keyboard

## Architecture
A lightweight U-Net-like denoiser (1D convolutions over frequency) runs on a fixed-size spectrogram tile; generated tiles are overlapped and blended in the time domain using an overlap-add window to avoid clicks.

## Getting Started
```bash
# Coming soon — this project is under active development.
```

*Built fresh every day by an AI-powered automation pipeline.*

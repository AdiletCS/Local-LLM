# Local LLM

Local LLM is a Windows desktop application for finding public GGUF language models that may fit your PC, downloading a selected model, and chatting with it locally. It uses CPU inference through llama.cpp.

## Features

- Detects CPU, memory, GPU, and storage information.
- Searches public Hugging Face GGUF repositories and estimates model compatibility.
- Downloads selected models and automatically installs the Windows CPU llama.cpp runtime when needed.
- Streams local model responses and saves conversations on your computer.

## Requirements

- Windows 10 or 11, 64-bit
- Python 3.10 or later
- Internet access for Hugging Face discovery and downloads, and for the first llama.cpp runtime installation

## Install and run

In PowerShell, from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

In the app, choose **Find Models for My PC**, select a recommendation, and choose **Use Model**. The app downloads the selected GGUF if needed, starts llama.cpp locally, and enables chat when the model is ready.

## Data and privacy

On Windows, settings, downloaded models, conversations, Hugging Face metadata cache, and llama.cpp are stored under `%LOCALAPPDATA%\LocalLLM\`. Prompts are sent to the local llama.cpp server on `127.0.0.1`; Hugging Face is contacted for discovery and model files. Model files are not included in this repository.

## Limitations and third-party software

This project currently supports Windows x64 CPU inference. GPU acceleration, macOS, and Linux support are not claimed. Compatibility and memory requirements are estimates, and actual response speed and quality vary by model and hardware. Model discovery and downloads require internet access. Check each model repository's license before using or redistributing its weights.

The application uses [llama.cpp](https://github.com/ggml-org/llama.cpp) for local inference, [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) for model discovery and downloads, and PySide6/Qt for the desktop interface. These third-party components and downloaded models remain under their respective licenses. This repository currently has no project license; no reuse license is granted for the original project code.

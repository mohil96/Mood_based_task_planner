# Mood-based Task Planner

This project analyzes a user's **mood from a short video** by extracting emotional cues from:

- 🧠 Facial expressions (via DeepFace)
- 🎤 Voice tone (via audio features)
- 📄 Spoken words (via Whisper transcription + sentiment analysis)

---

## 🚀 Features

- Extracts audio and frames from an uploaded video.
- Runs facial emotion recognition on multiple sampled frames.
- Transcribes speech using OpenAI Whisper.
- Analyzes text sentiment using HuggingFace Transformers.
- Computes basic voice tone features (pitch and energy).
- Outputs a summary of emotional state.

---

## 📁 File Structure

mood-video-analyzer/
├── mood_analyzer.py # Main Python script
├── requirements.txt # Dependencies
└── README.md # Project documentation

## Installation

### 1. Create and activate a virtual environment:

```bash
python -m venv mood_env
source mood_env/bin/activate  # Windows: mood_env\Scripts\activate
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
source mood_env/bin/activate  # Windows: mood_env\Scripts\activate
```

💡 If you want GPU support with CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

Place your .mp4 video file in the project folder.

Run the script:

```bash
python mood_analyzer.py
```

Enter the path to your video when prompted.

## Output

You'll see:

Transcribed text

Sentiment label (positive/negative/neutral)

Average facial emotion scores (e.g., happy, sad, angry)

Dominant facial emotion

Pitch and energy of your voice



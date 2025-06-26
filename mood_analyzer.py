import os
import tempfile
from moviepy import VideoFileClip
from deepface import DeepFace
import librosa
import whisper
from transformers import pipeline
import torch

def analyze_mood(video_path):
    # Create temp directory and extract audio
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "audio.wav")

    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec='pcm_s16le')

    # Transcribe audio using Whisper
    model_whisper = whisper.load_model("base")
    transcription = model_whisper.transcribe(audio_path)
    transcript = transcription['text']

    # Facial emotion analysis
    face_emotion_result = DeepFace.analyze(
        img_path=video_path, actions=['emotion'], enforce_detection=False
    )
    dominant_facial_emotion = face_emotion_result[0]['dominant_emotion']
    facial_emotions = face_emotion_result[0]['emotion']

    # Sentiment from transcript
    sentiment_analyzer = pipeline("sentiment-analysis")
    sentiment_result = sentiment_analyzer(transcript[:512])[0]

    # Audio tone features
    y, sr = librosa.load(audio_path)
    pitch = librosa.yin(y, fmin=50, fmax=300).mean()
    energy = librosa.feature.rms(y=y).mean()

    # Print results
    print("\n🎤 Transcript:", transcript)
    print("\n🙂 Facial Emotion Scores:", facial_emotions)
    print("😎 Dominant Facial Emotion:", dominant_facial_emotion)
    print("\n🗣 Text Sentiment:", sentiment_result)
    print(f"\n📊 Voice Features → Avg Pitch: {pitch:.2f}, Energy: {energy:.5f}")

if __name__ == "__main__":
    video_file = input("Enter the path to your video file (e.g., mood.mp4): ")
    analyze_mood(video_file)


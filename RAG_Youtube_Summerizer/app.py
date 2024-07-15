import streamlit as st
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import cv2
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
from PIL import Image
from moviepy.editor import VideoFileClip
import speech_recognition as sr
import math
import yt_dlp

# Load models (you might want to move this outside the Streamlit app for efficiency)
text_model = SentenceTransformer('all-MiniLM-L6-v2')
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
flan_t5_model_name = "google/flan-t5-large"
flan_t5_tokenizer = AutoTokenizer.from_pretrained(flan_t5_model_name)
flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained(flan_t5_model_name)

# Define paths
output_folder = "./mixed_data/"
output_video_path = "./video_data/"
output_audio_path = os.path.join(output_folder, "output_audio.wav")
output_frames_path = os.path.join(output_folder, "frames/")
output_text_path = os.path.join(output_folder, "extracted_text.txt")

# Create necessary directories
for path in [output_folder, output_video_path, output_frames_path]:
    os.makedirs(path, exist_ok=True)

# Functions from your project
def download_video(url, output_path=output_video_path):
    ydl_opts = {
        'outtmpl': output_path + '%(title)s.%(ext)s',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
        print("Download completed!")
        return filename
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def extract_frames(video_path, output_path, interval=20):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    
    success, frame = video.read()
    count = 0
    frame_count = 0
    
    while success:
        if count % frame_interval == 0:
            cv2.imwrite(f"{output_path}frame_{frame_count:04d}.jpg", frame)
            frame_count += 1
        success, frame = video.read()
        count += 1
    
    video.release()
    print(f"Extracted {frame_count} frames")

def extract_audio(video_path, output_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_path)
    video.close()
    print("Audio extracted successfully")

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    full_text = ""
    with sr.AudioFile(audio_path) as source:
        audio_duration = math.ceil(source.DURATION)
        for i in range(0, audio_duration, 60):
            audio = recognizer.record(source, duration=60)
            try:
                text = recognizer.recognize_google(audio)
                full_text += text + " "
            except sr.UnknownValueError:
                print(f"Speech Recognition could not understand audio at {i}-{i+60} seconds")
            except sr.RequestError as e:
                print(f"Could not request results from Speech Recognition service; {e}")
    return full_text.strip()

def process_video(video_url):
    video_file = download_video(video_url)
    if video_file:
        extract_frames(video_file, output_frames_path)
        extract_audio(video_file, output_audio_path)
        extracted_text = transcribe_audio(output_audio_path)
        with open(output_text_path, "w", encoding='utf-8') as text_file:
            text_file.write(extracted_text)
        return True
    return False

def build_index(text_path, images_path):
    # Build text index
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text_embedding = text_model.encode(text)
    text_index = faiss.IndexFlatL2(text_embedding.shape[0])
    text_index.add(text_embedding.reshape(1, -1).astype('float32'))

    # Build image index
    image_embeddings = []
    for filename in os.listdir(images_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_path, filename)
            image = Image.open(image_path)
            inputs = clip_processor(images=image, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
            image_embeddings.append(image_features.squeeze().numpy())

    image_index = faiss.IndexFlatL2(image_embeddings[0].shape[0])
    image_index.add(np.vstack(image_embeddings).astype('float32'))

    # Save indices
    faiss.write_index(text_index, os.path.join(output_folder, "text_index.faiss"))
    faiss.write_index(image_index, os.path.join(output_folder, "image_index.faiss"))

    # Save metadata
    with open(os.path.join(output_folder, "metadata.txt"), 'w') as f:
        f.write(text_path + '\n')
        for filename in os.listdir(images_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                f.write(os.path.join(images_path, filename) + '\n')

def retrieve(query_str, top_k=5):
    text_query_embedding = text_model.encode(query_str)
    clip_inputs = clip_processor(text=[query_str], return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        image_query_embedding = clip_model.get_text_features(**clip_inputs).squeeze().numpy()

    text_index = faiss.read_index(os.path.join(output_folder, "text_index.faiss"))
    image_index = faiss.read_index(os.path.join(output_folder, "image_index.faiss"))

    text_distances, text_indices = text_index.search(text_query_embedding.reshape(1, -1).astype('float32'), top_k)
    image_distances, image_indices = image_index.search(image_query_embedding.reshape(1, -1).astype('float32'), top_k)

    with open(os.path.join(output_folder, "metadata.txt"), "r") as f:
        metadata = f.read().splitlines()

    text_results = [metadata[0]] if text_indices[0][0] != -1 else []
    image_results = [metadata[i+1] for i in image_indices[0] if i != -1]

    return text_results, image_results

def generate_response(context_str, query_str):
    input_text = f"Context: {context_str}\nQuestion: {query_str}\nAnswer:"
    inputs = flan_t5_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    with torch.no_grad():
        outputs = flan_t5_model.generate(
            **inputs,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
    response = flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def answer_question(query_str):
    text_results, image_results = retrieve(query_str)
    with open(text_results[0], 'r', encoding='utf-8') as f:
        context_str = f.read()
    answer = generate_response(context_str, query_str)
    
    images = []
    for image_path in image_results:
        img = Image.open(image_path)
        img.thumbnail((200, 200))
        images.append(img)
    
    return answer, images

# Streamlit app
st.title("Video Analysis and Q&A")

# Input for video URL
video_url = st.text_input("Enter video URL:")

if video_url:
    if 'current_video' not in st.session_state or st.session_state.current_video != video_url:
        st.session_state.current_video = video_url
        with st.spinner("Processing video..."):
            success = process_video(video_url)
            if success:
                build_index(output_text_path, output_frames_path)
                st.success("Video processed successfully!")
            else:
                st.error("Failed to process video. Please check the URL and try again.")

    # Input for query
    query = st.text_input("Enter your question about the video:")

    if query:
        with st.spinner("Generating answer..."):
            answer, images = answer_question(query)
            st.write("Answer:")
            st.write(answer)
            st.write("Relevant Images:")
            cols = st.columns(len(images))
            for col, img in zip(cols, images):
                col.image(img)

# Run the Streamlit app with: streamlit run app.py
# import streamlit as st
# import os
# import shutil
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import cv2
# from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM
# import faiss
# from PIL import Image
# from moviepy.editor import VideoFileClip
# import speech_recognition as sr
# import math
# import yt_dlp
# import base64
# # from io import BytesIO

# # Load models (you might want to move this outside the Streamlit app for efficiency)
# @st.cache_resource
# def load_models():
#     text_model = SentenceTransformer('all-MiniLM-L6-v2')
#     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     flan_t5_model_name = "google/flan-t5-large"
#     flan_t5_tokenizer = AutoTokenizer.from_pretrained(flan_t5_model_name)
#     flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained(flan_t5_model_name)
#     return text_model, clip_model, clip_processor, flan_t5_tokenizer, flan_t5_model

# text_model, clip_model, clip_processor, flan_t5_tokenizer, flan_t5_model = load_models()

# # Define paths
# output_folder = "./mixed_data/"
# output_video_path = "./video_data/"
# output_audio_path = os.path.join(output_folder, "output_audio.wav")
# output_frames_path = os.path.join(output_folder, "frames/")
# output_text_path = os.path.join(output_folder, "extracted_text.txt")

# # Create necessary directories
# for path in [output_folder, output_video_path, output_frames_path]:
#     os.makedirs(path, exist_ok=True)

# # Functions from your project
# def clean_old_videos(current_video_file):
#     for filename in os.listdir(output_video_path):
#         file_path = os.path.join(output_video_path, filename)
#         if filename != current_video_file:
#             try:
#                 os.remove(file_path)
#             except Exception as e:
#                 st.error(f"Error removing old video file: {str(e)}")


# # Functions from your project
# def download_video(url, output_path=output_video_path):
#     ydl_opts = {
#         'outtmpl': output_path + '%(title)s.%(ext)s',
#     }
#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             info = ydl.extract_info(url, download=True)
#             filename = ydl.prepare_filename(info)
#         return filename
#     except Exception as e:
#         st.error(f"An error occurred while downloading the video: {str(e)}")
#         return None

# def extract_frames(video_path, output_path, interval=20):
#     video = cv2.VideoCapture(video_path)
#     fps = video.get(cv2.CAP_PROP_FPS)
#     frame_interval = int(fps * interval)
    
#     success, frame = video.read()
#     count = 0
#     frame_count = 0
    
#     while success:
#         if count % frame_interval == 0:
#             cv2.imwrite(f"{output_path}frame_{frame_count:04d}.jpg", frame)
#             frame_count += 1
#         success, frame = video.read()
#         count += 1
    
#     video.release()
#     return frame_count

# def extract_audio(video_path, output_path):
#     try:
#         video = VideoFileClip(video_path)
#         audio = video.audio
#         audio.write_audiofile(output_path)
#         video.close()
#         return True
#     except Exception as e:
#         st.error(f"An error occurred while extracting audio: {str(e)}")
#         return False

# def transcribe_audio(audio_path):
#     recognizer = sr.Recognizer()
#     full_text = ""
#     with sr.AudioFile(audio_path) as source:
#         audio_duration = math.ceil(source.DURATION)
#         for i in range(0, audio_duration, 60):
#             audio = recognizer.record(source, duration=60)
#             try:
#                 text = recognizer.recognize_google(audio)
#                 full_text += text + " "
#             except sr.UnknownValueError:
#                 st.warning(f"Speech Recognition could not understand audio at {i}-{i+60} seconds")
#             except sr.RequestError as e:
#                 st.error(f"Could not request results from Speech Recognition service; {e}")
#     return full_text.strip()

# def process_video(video_url):
#     with st.spinner("Downloading video..."):
#         video_file = download_video(video_url)
#     if video_file:
#         with st.spinner("Extracting frames..."):
#             frame_count = extract_frames(video_file, output_frames_path)
#         with st.spinner("Extracting audio..."):
#             audio_success = extract_audio(video_file, output_audio_path)
#         if audio_success:
#             with st.spinner("Transcribing audio..."):
#                 extracted_text = transcribe_audio(output_audio_path)
#             with open(output_text_path, "w", encoding='utf-8') as text_file:
#                 text_file.write(extracted_text)
#             return video_file, frame_count, extracted_text
#     return None, 0, ""

# def build_index(text_path, images_path):
#     with st.spinner("Building text index..."):
#         with open(text_path, 'r', encoding='utf-8') as f:
#             text = f.read()
#         text_embedding = text_model.encode(text)
#         text_index = faiss.IndexFlatL2(text_embedding.shape[0])
#         text_index.add(text_embedding.reshape(1, -1).astype('float32'))

#     with st.spinner("Building image index..."):
#         image_embeddings = []
#         for filename in os.listdir(images_path):
#             if filename.endswith(('.png', '.jpg', '.jpeg')):
#                 image_path = os.path.join(images_path, filename)
#                 image = Image.open(image_path)
#                 inputs = clip_processor(images=image, return_tensors="pt", padding=True, truncation=True)
#                 with torch.no_grad():
#                     image_features = clip_model.get_image_features(**inputs)
#                 image_embeddings.append(image_features.squeeze().numpy())

#         image_index = faiss.IndexFlatL2(image_embeddings[0].shape[0])
#         image_index.add(np.vstack(image_embeddings).astype('float32'))

#     with st.spinner("Saving indices..."):
#         faiss.write_index(text_index, os.path.join(output_folder, "text_index.faiss"))
#         faiss.write_index(image_index, os.path.join(output_folder, "image_index.faiss"))

#     with open(os.path.join(output_folder, "metadata.txt"), 'w') as f:
#         f.write(text_path + '\n')
#         for filename in os.listdir(images_path):
#             if filename.endswith(('.png', '.jpg', '.jpeg')):
#                 f.write(os.path.join(images_path, filename) + '\n')

# def retrieve(query_str, top_k=5):
#     text_query_embedding = text_model.encode(query_str)
#     clip_inputs = clip_processor(text=[query_str], return_tensors="pt", padding=True, truncation=True)
    
#     with torch.no_grad():
#         image_query_embedding = clip_model.get_text_features(**clip_inputs).squeeze().numpy()

#     text_index = faiss.read_index(os.path.join(output_folder, "text_index.faiss"))
#     image_index = faiss.read_index(os.path.join(output_folder, "image_index.faiss"))

#     text_distances, text_indices = text_index.search(text_query_embedding.reshape(1, -1).astype('float32'), top_k)
#     image_distances, image_indices = image_index.search(image_query_embedding.reshape(1, -1).astype('float32'), top_k)

#     with open(os.path.join(output_folder, "metadata.txt"), "r") as f:
#         metadata = f.read().splitlines()

#     text_results = [metadata[0]] if text_indices[0][0] != -1 else []
#     image_results = [metadata[i+1] for i in image_indices[0] if i != -1]

#     return text_results, image_results

# def generate_response(context_str, query_str):
#     input_text = f"Context: {context_str}\nQuestion: {query_str}\nAnswer:"
#     inputs = flan_t5_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
#     with torch.no_grad():
#         outputs = flan_t5_model.generate(
#             **inputs,
#             max_length=150,
#             num_return_sequences=1,
#             temperature=0.7,
#             do_sample=True,
#             top_k=50,
#             top_p=0.95,
#         )
#     response = flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# def answer_question(query_str):
#     text_results, image_results = retrieve(query_str)
#     with open(text_results[0], 'r', encoding='utf-8') as f:
#         context_str = f.read()
#     answer = generate_response(context_str, query_str)
    
#     images = []
#     for image_path in image_results:
#         img = Image.open(image_path)
#         img.thumbnail((200, 200))
#         images.append(img)
    
#     return answer, images

# def set_background(image_file):
#     with open(image_file, "rb") as f:
#         img_data = f.read()
#     b64_encoded = base64.b64encode(img_data).decode()
#     style = f"""
#         <style>
#         .stApp {{
#             background-image: url(data:image/png;base64,{b64_encoded});
#             background-size: cover;
#             background-position: center;
#             background-repeat: no-repeat;
#             background-attachment: fixed;
#         }}
#         .stApp::before {{
#             content: "";
#             position: fixed;
#             top: 0;
#             left: 0;
#             width: 100%;
#             height: 100%;
#             background-color: rgba(0, 0, 0, 0.3);
#             z-index: -1;
#         }}
#         .main .block-container {{
#             background: none;
#         }}
#         .sidebar .sidebar-content,
#         [data-testid="stSidebar"] {{
#             background-color: rgba(0, 0, 0, 0.5) !important;
#             backdrop-filter: blur(10px);
#         }}
#         .sidebar .sidebar-content {{
#             background: none;
#         }}
#         h1, h2, h3 {{
#             color: #FFFFFF;
#             text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
#         }}
#         p, label, .stTextInput > label {{
#             color: #F0F0F0 !important;
#         }}
#         .stTextInput > div > div > input {{
#             background-color: rgba(255, 255, 255, 0.1);
#             color: #FFFFFF;
#             border: none;
#             border-bottom: 2px solid rgba(255, 255, 255, 0.5);
#             border-radius: 0;
#             padding: 0.5rem;
#             transition: all 0.3s;
#         }}
#         .stTextInput > div > div > input:focus {{
#             box-shadow: none;
#             border-bottom: 2px solid #FFFFFF;
#         }}
#         .stButton > button {{
#             background-color: rgba(255, 255, 255, 0.2);
#             color: #FFFFFF;
#             border: 1px solid rgba(255, 255, 255, 0.5);
#             border-radius: 5px;
#             padding: 0.5rem 1rem;
#             transition: all 0.3s;
#             font-weight: bold;
#         }}
#         .stButton > button:hover {{
#             background-color: rgba(255, 255, 255, 0.3);
#         }}
#         .apply-button {{
#             background-color: rgba(0, 150, 255, 0.6);
#             color: #FFFFFF;
#             border: none;
#             border-radius: 5px;
#             padding: 0.5rem 1rem;
#             transition: all 0.3s;
#             font-weight: bold;
#             text-transform: uppercase;
#             letter-spacing: 1px;
#             margin-top: 10px;
#         }}
#         .apply-button:hover {{
#             background-color: rgba(0, 150, 255, 0.8);
#             box-shadow: 0 0 15px rgba(0, 150, 255, 0.5);
#         }}
#         .stRadio > label {{
#             color: #F0F0F0 !important;
#         }}
#         .stProgress > div > div > div > div {{
#             background-color: rgba(255, 255, 255, 0.7);
#         }}
#         </style>
#     """
#     st.markdown(style, unsafe_allow_html=True)

# # Streamlit app
# def clean_old_videos(current_video_file):
#     for filename in os.listdir(output_video_path):
#         file_path = os.path.join(output_video_path, filename)
#         if filename != os.path.basename(current_video_file):
#             try:
#                 os.remove(file_path)
#             except Exception as e:
#                 st.error(f"Error removing old video file: {str(e)}")

# def initialize_session_state():
#     if 'tasks_completed' not in st.session_state:
#         st.session_state.tasks_completed = False
#     if 'current_video' not in st.session_state:
#         st.session_state.current_video = None
#     if 'video_file' not in st.session_state:
#         st.session_state.video_file = None
#     if 'extracted_text' not in st.session_state:
#         st.session_state.extracted_text = None

# def process_video_wrapper(video_url):
#     try:
#         with st.spinner("Processing video... Please be patient."):
#             video_file, frame_count, extracted_text = process_video(video_url)
#             if video_file:
#                 clean_old_videos(video_file)
#                 st.session_state.video_file = video_file
#                 st.session_state.extracted_text = extracted_text
#                 with st.spinner("Building index... This may take a while."):
#                     build_index(output_text_path, output_frames_path)
#                 st.success("Video processed successfully!")
#                 st.session_state.tasks_completed = True
#             else:
#                 st.error("Failed to process video. Please check the URL and try again.")
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}. Please try again or contact support if the issue persists.")

# def display_processed_content():
#     if st.session_state.video_file:
#         st.video(st.session_state.video_file)
        
#         st.subheader("Extracted Text")
#         st.text_area("", st.session_state.extracted_text, height=200)
        
#         st.download_button(
#             label="Download Video",
#             data=open(st.session_state.video_file, 'rb').read(),
#             file_name="downloaded_video.mp4",
#             mime="video/mp4"
#         )
        
#         st.download_button(
#             label="Download Extracted Text",
#             data=st.session_state.extracted_text,
#             file_name="extracted_text.txt",
#             mime="text/plain"
#         )

# def main():
#     set_background('./background-op.jpeg')
#     initialize_session_state()

#     st.title("AI Advanced Video Analysis and Q&A System")

#     st.sidebar.header("Navigation")
#     page = st.sidebar.radio("Go to", ["Video Processing", "Q&A"])

#     if page == "Video Processing":
#         st.header("Video Processing")
#         video_url = st.text_input("Enter video URL:")
#         process_button = st.button("Process Video")

#         if video_url and process_button:
#             if st.session_state.current_video != video_url:
#                 st.session_state.current_video = video_url
#                 process_video_wrapper(video_url)
            
#         if st.session_state.tasks_completed:
#             display_processed_content()

#     elif page == "Q&A":
#         st.header("Question & Answer")
#         if not st.session_state.tasks_completed:
#             st.warning("Please process a video first.")
#         else:
#             display_processed_content()
            
#             query = st.text_input("Enter your question about the video:")

#             if query:
#                 try:
#                     with st.spinner("Generating answer... Please wait."):
#                         answer, images = answer_question(query)
#                         st.subheader("Answer:")
#                         st.write(answer)
#                         st.subheader("Relevant Images:")
#                         cols = st.columns(len(images))
#                         for col, img in zip(cols, images):
#                             col.image(img)
#                 except Exception as e:
#                     st.error(f"An error occurred while generating the answer: {str(e)}. Please try again or rephrase your question.")

# if __name__ == "__main__":
#     main()

















import streamlit as st
import os
import shutil
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
import base64
# from io import BytesIO

# Load models (you might want to move this outside the Streamlit app for efficiency)
@st.cache_resource
def load_models():
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    flan_t5_model_name = "google/flan-t5-large"
    flan_t5_tokenizer = AutoTokenizer.from_pretrained(flan_t5_model_name)
    flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained(flan_t5_model_name)
    return text_model, clip_model, clip_processor, flan_t5_tokenizer, flan_t5_model

text_model, clip_model, clip_processor, flan_t5_tokenizer, flan_t5_model = load_models()

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
def clean_old_videos(current_video_file):
    for filename in os.listdir(output_video_path):
        file_path = os.path.join(output_video_path, filename)
        if filename != current_video_file:
            try:
                os.remove(file_path)
            except Exception as e:
                st.error(f"Error removing old video file: {str(e)}")


# Functions from your project
def download_video(url, output_path=output_video_path):
    ydl_opts = {
        'outtmpl': output_path + '%(title)s.%(ext)s',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
        return filename
    except Exception as e:
        st.error(f"An error occurred while downloading the video: {str(e)}")
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
    return frame_count

def extract_audio(video_path, output_path):
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_path)
        video.close()
        return True
    except Exception as e:
        st.error(f"An error occurred while extracting audio: {str(e)}")
        return False

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
                st.warning(f"Speech Recognition could not understand audio at {i}-{i+60} seconds")
            except sr.RequestError as e:
                st.error(f"Could not request results from Speech Recognition service; {e}")
    return full_text.strip()

def process_video(video_url):
    with st.spinner("Downloading video..."):
        video_file = download_video(video_url)
    if video_file:
        with st.spinner("Extracting frames..."):
            frame_count = extract_frames(video_file, output_frames_path)
        with st.spinner("Extracting audio..."):
            audio_success = extract_audio(video_file, output_audio_path)
        if audio_success:
            with st.spinner("Transcribing audio..."):
                extracted_text = transcribe_audio(output_audio_path)
            with open(output_text_path, "w", encoding='utf-8') as text_file:
                text_file.write(extracted_text)
            return video_file, frame_count, extracted_text
    return None, 0, ""

def build_index(text_path, images_path):
    with st.spinner("Building text index..."):
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text_embedding = text_model.encode(text)
        text_index = faiss.IndexFlatL2(text_embedding.shape[0])
        text_index.add(text_embedding.reshape(1, -1).astype('float32'))

    with st.spinner("Building image index..."):
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

    with st.spinner("Saving indices..."):
        faiss.write_index(text_index, os.path.join(output_folder, "text_index.faiss"))
        faiss.write_index(image_index, os.path.join(output_folder, "image_index.faiss"))

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

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.3);
            z-index: -1;
        }}
        .main .block-container {{
            background: none;
        }}
        .sidebar .sidebar-content,
        [data-testid="stSidebar"] {{
            background-color: rgba(0, 0, 0, 0.5) !important;
            backdrop-filter: blur(10px);
        }}
        .sidebar .sidebar-content {{
            background: none;
        }}
        h1, h2, h3 {{
            color: #FFFFFF;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        p, label, .stTextInput > label {{
            color: #F0F0F0 !important;
        }}
        .stTextInput > div > div > input {{
            background-color: rgba(255, 255, 255, 0.1);
            color: #FFFFFF;
            border: none;
            border-bottom: 2px solid rgba(255, 255, 255, 0.5);
            border-radius: 0;
            padding: 0.5rem;
            transition: all 0.3s;
        }}
        .stTextInput > div > div > input:focus {{
            box-shadow: none;
            border-bottom: 2px solid #FFFFFF;
        }}
        .stButton > button {{
            background-color: rgba(255, 255, 255, 0.2);
            color: #FFFFFF;
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 5px;
            padding: 0.5rem 1rem;
            transition: all 0.3s;
            font-weight: bold;
        }}
        .stButton > button:hover {{
            background-color: rgba(255, 255, 255, 0.3);
        }}
        .apply-button {{
            background-color: rgba(0, 150, 255, 0.6);
            color: #FFFFFF;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            transition: all 0.3s;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 10px;
        }}
        .apply-button:hover {{
            background-color: rgba(0, 150, 255, 0.8);
            box-shadow: 0 0 15px rgba(0, 150, 255, 0.5);
        }}
        .stRadio > label {{
            color: #F0F0F0 !important;
        }}
        .stProgress > div > div > div > div {{
            background-color: rgba(255, 255, 255, 0.7);
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# Streamlit app
def clean_old_videos(current_video_file):
    for filename in os.listdir(output_video_path):
        file_path = os.path.join(output_video_path, filename)
        if filename != os.path.basename(current_video_file):
            try:
                os.remove(file_path)
            except Exception as e:
                st.error(f"Error removing old video file: {str(e)}")

def initialize_session_state():
    if 'tasks_completed' not in st.session_state:
        st.session_state.tasks_completed = False
    if 'current_video' not in st.session_state:
        st.session_state.current_video = None
    if 'video_file' not in st.session_state:
        st.session_state.video_file = None
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = None

def process_video_wrapper(video_url):
    try:
        with st.spinner("Processing video... Please be patient and don't switch to Q&A section while processing \n or It will lost all the tracks of the video link you provided 👾."):
            video_file, frame_count, extracted_text = process_video(video_url)
            if video_file:
                clean_old_videos(video_file)
                st.session_state.video_file = video_file
                st.session_state.extracted_text = extracted_text
                with st.spinner("Building index... This may take a while."):
                    build_index(output_text_path, output_frames_path)
                st.success("Video processed successfully!")
                st.success("Now go to Q&A section through navigation!")
                st.session_state.tasks_completed = True
            else:
                st.error("Failed to process video. Please check the URL and try again.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}. Please try again or contact support if the issue persists.")

def display_processed_content():
    if st.session_state.video_file:
        st.video(st.session_state.video_file)
        
        st.subheader("Extracted Text")
        st.text_area("", st.session_state.extracted_text, height=200)
        
        st.download_button(
            label="Download Video",
            data=open(st.session_state.video_file, 'rb').read(),
            file_name="downloaded_video.mp4",
            mime="video/mp4"
        )
        
        st.download_button(
            label="Download Extracted Text",
            data=st.session_state.extracted_text,
            file_name="extracted_text.txt",
            mime="text/plain"
        )

def main():
    set_background('./background-op.jpeg')
    initialize_session_state()

    st.title("AI Advanced Video Analysis and Q&A System")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Video Processing", "Q&A"])

    if page == "Video Processing":
        st.header("Video Processing")
        video_url = st.text_input("Enter video URL:")
        process_button = st.button("Process Video")

        if video_url and process_button:
            if st.session_state.current_video != video_url:
                st.session_state.current_video = video_url
                process_video_wrapper(video_url)
            
        if st.session_state.tasks_completed:
            display_processed_content()

    elif page == "Q&A":
        st.header("Question & Answer")
        if not st.session_state.tasks_completed:
            st.warning("Please process a video first.")
        else:
            display_processed_content()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                query = st.text_input("Enter your question about the video:")
            with col2:
                search_button = st.button("Search")

            if query and (search_button or st.session_state.get('search_triggered', False)):
                st.session_state.search_triggered = False
                try:
                    with st.spinner("Generating answer... Please wait."):
                        answer, images = answer_question(query)
                        st.subheader("Answer:")
                        st.write(answer)
                        st.subheader("Relevant Images:")
                        cols = st.columns(len(images))
                        for col, img in zip(cols, images):
                            col.image(img)
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {str(e)}. Please try again or rephrase your question.")

    # Add this at the end of the main() function to enable "Enter" key search
    st.markdown(
        <script>
        const input = window.parent.document.querySelector('.stTextInput input');
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                setTimeout(() => {
                    window.parent.document.querySelector('.stButton button').click();
                }, 10);
            }
        });
        </script>,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

# AI Advanced Video Analysis and Q&A System (English)

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [System Architecture](#system-architecture)
6. [Dependencies and Open-Source Models](#dependencies-and-open-source-models)
7. [Process Flow](#process-flow)
8. [Limitations](#limitations)
9. [Future Improvements](#future-improvements)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

The AI Advanced Video Analysis and Q&A System is a sophisticated tool that combines video processing, natural language understanding, and machine learning to provide an interactive question-answering experience based on video content. This system downloads videos, extracts key information, and allows users to ask questions about the video's content.

## Documentation
For reference:  
[Click here](https://www.llamaindex.ai/blog/multimodal-rag-for-advanced-video-processing-with-llamaindex-lancedb-33be4804822e)

## Features

- Video download from URL using yt-dlp
- Frame extraction at regular intervals using OpenCV
- Audio extraction and transcription using MoviePy and SpeechRecognition
- Text and image indexing for efficient retrieval using FAISS
- Question answering based on video content using FLAN-T5
- User-friendly interface built with Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sayan-Maity-Code/Advanced_AI_video_analysis_and_Q-A_system.git
```
```bash
cd RAG_Youtube_Summerizer
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the necessary model files (not included in the repository due to size):
- SentenceTransformer model: 'all-MiniLM-L6-v2'
- CLIP model: "openai/clip-vit-base-patch32"
- FLAN-T5 model: "google/flan-t5-large"

## Demo

<p align="center">
  <img src="path/to/your/demo.gif" alt="AI Video Analysis and Q&A System Demo" width="600">
</p>

<p align="center">
  Demo of the AI Advanced Video Analysis and Q&A System in action.
</p>

<p align="center">
  <a href="https://www.linkedin.com/posts/sayan-maity-756b8b244_developing-an-ai-powered-video-analysis-and-activity-7219230424017858561-JvnJ?utm_source=share&utm_medium=member_desktop">View full video on LinkedIn</a>
</p>

<p align="center">
  Click the image above to watch a demo of the AI Advanced Video Analysis and Q&A System in action on LinkedIn.
</p>

## Usage

1. Run the Streamlit app:

streamlit run app.py

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the navigation sidebar to switch between "Video Processing" and "Q&A" sections.

4. In the "Video Processing" section:
- Enter a video URL and click "Process Video"
- Wait for the video to be downloaded, processed, and indexed

5. In the "Q&A" section:
- Enter questions about the processed video
- View the AI-generated answers and relevant images

## System Architecture

The system is built using a modular architecture, with distinct components for video processing, indexing, and question answering. It leverages several open-source models and libraries to achieve its functionality.

## Dependencies and Open-Source Models

1. **Streamlit**: Used for creating the web-based user interface.

2. **yt-dlp**: A fork of youtube-dl, used for downloading videos from various sources.

3. **OpenCV (cv2)**: Used for video frame extraction and image processing.

4. **MoviePy**: Utilized for audio extraction from videos.

5. **SpeechRecognition**: Employed for transcribing audio to text.

6. **Sentence-Transformers**: 
- Model: 'all-MiniLM-L6-v2'
- Used for generating text embeddings for efficient retrieval.

7. **CLIP (Contrastive Language-Image Pretraining)**:
- Model: "openai/clip-vit-base-patch32"
- Used for generating image embeddings and text-image similarity.

8. **FLAN-T5**:
- Model: "google/flan-t5-large"
- Employed for generating answers based on retrieved context.

9. **FAISS (Facebook AI Similarity Search)**:
- Used for efficient similarity search and indexing of embeddings.

10. **PyTorch**: Underlying framework for deep learning models.

11. **NumPy**: Used for numerical operations and array manipulations.

## Process Flow

1. **Video Download and Processing**:
- The system uses yt-dlp to download the video from the provided URL.
- OpenCV is used to extract frames at regular intervals (every 20 seconds by default).
- MoviePy extracts the audio from the video.
- SpeechRecognition transcribes the audio to text.

2. **Indexing**:
- The Sentence-Transformer model generates embeddings for the transcribed text.
- The CLIP model generates embeddings for the extracted video frames.
- FAISS is used to create efficient indices for both text and image embeddings.

3. **Retrieval**:
- When a question is asked, the system generates embeddings for the query using both Sentence-Transformer and CLIP models.
- FAISS indices are used to retrieve the most relevant text and images based on cosine similarity.

4. **Answer Generation**:
- The retrieved text context and the user's question are fed into the FLAN-T5 model.
- FLAN-T5 generates a natural language answer based on the provided context and question.

5. **Result Presentation**:
- The generated answer is displayed to the user along with relevant images from the video.

## Limitations

1. **Processing Time**: For long videos, the download, frame extraction, and transcription processes can be time-consuming.

2. **Speech Recognition Accuracy**: The accuracy of transcription depends on the audio quality and may not be perfect for all videos.

3. **Language Support**: Currently, the system is optimized for English content. Other languages may not be supported or may have reduced accuracy.

4. **Video Source Limitations**: The system relies on yt-dlp for video downloading, which may not work for all video sources or may break if websites change their structure.

5. **Answer Quality**: The quality of generated answers depends on the accuracy of transcription and the relevance of retrieved context. Complex or nuanced questions may not always receive accurate answers.

6. **Resource Intensive**: The system requires significant computational resources, especially RAM, to handle large videos and run multiple ML models simultaneously.

## Future Improvements

1. Implement multi-language support for video analysis and question answering.
2. Optimize processing time for longer videos through parallel processing or cloud computing integration.
3. Enhance the retrieval system to better handle context and temporal information from videos.
4. Integrate more advanced language models for improved answer generation.
5. Add support for real-time video analysis and streaming sources.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

import os
import sys
import torch
import random
import re
import json
import math
import copy
import requests
from functools import lru_cache
from tqdm import tqdm
from torch.nn.parameter import Parameter
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import time
import threading
import queue
import httpx
import asyncio
import torch.nn as nn
import torch.nn.functional as F
import uuid
import wget
from duckduckgo_search import DDGS
import warnings
from datetime import datetime
import unicodedata
import nltk
import torchaudio
import logging
from PIL import Image
from io import BytesIO
import sentencepiece as spm
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

nltk.download('punkt', quiet=True)

GPT2_FOLDER = "./GPT2"
MODEL_FILE = "gpt2-pytorch_model.bin"
ENCODER_FILE = "encoder.json"
VOCAB_FILE = "vocab.bpe"
MODEL_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin"
ENCODER_URL = "https://raw.githubusercontent.com/graykode/gpt-2-Pytorch/refs/heads/master/GPT2/GPT2/encoder.json"
VOCAB_URL = "https://raw.githubusercontent.com/graykode/gpt-2-Pytorch/refs/heads/master/GPT2/GPT2/vocab.bpe"
GPT2_FILES_URLS = [
    (MODEL_URL, MODEL_FILE),
    (ENCODER_URL, ENCODER_FILE),
    (VOCAB_URL, VOCAB_FILE),
]

TEXT_GENERATION_RATE = 40000
MAX_LENGTH = 1024
MAX_XDD = 5
END_OF_TEXT_TOKEN = "<|endoftext|>"

html_code = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Text Generation</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" integrity="sha512-9usAa10IRO0HhonpyAIVpjrylPvoDwiPUiKdWk5t3PyolY1cOd4DSE0Ga+ri4AuTroPR5aQvXU9xC6qOPnzFeg==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f0f0;
            color: #333;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            width: 95%;
            max-width: 900px;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            margin-top: 20px;
            margin-bottom: 20px;
            display: flex;
            flex-direction: column;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .header h1 {
            font-size: 2em;
            color: #333;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
            resize: vertical;
        }
        button {
            padding: 10px 15px;
            border: none;
            border-radius: 5px;
            background-color: #007bff;
            color: white;
            font-size: 18px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        button:hover {
            background-color: #0056b3;
        }
        #output {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
            white-space: pre-wrap;
            word-break: break-word;
            overflow-y: auto;
            max-height: 100vh;
        }
        #output strong {
            font-weight: bold;
        }
        .animated-text {
            position: fixed;
            top: 20px;
            left: 20px;
            font-size: 1.5em;
            color: rgba(0, 0, 0, 0.1);
            pointer-events: none;
            z-index: -1;
        }
        @media (max-width: 768px) {
            .container {
                width: 98%;
                margin-top: 10px;
                margin-bottom: 10px;
                padding: 15px;
            }
            .header h1 {
                font-size: 1.8em;
            }
            .form-group textarea, .form-group input[type="text"] {
                font-size: 14px;
                padding: 8px;
            }
            button {
                font-size: 16px;
                padding: 8px 12px;
            }
            #output {
                font-size: 14px;
                padding: 10px;
                margin-top: 15px;
            }
        }
    </style>
</head>
<body>
<div class="animated-text animate__animated animate__fadeIn animate__infinite infinite">AI POWERED</div>
<div class="container">
    <div class="header animate__animated animate__fadeInDown">
    </div>
    <div class="form-group animate__animated animate__fadeInLeft">
        <textarea id="text" rows="5" placeholder="Enter text"></textarea>
    </div>
    <button onclick="generateText()" class="animate__animated animate__fadeInUp">Generate Reasoning</button>
    <div id="output" class="animate__animated">
        <strong >Response:</strong><br>
        <div id="generatedText"></div>
    </div>
</div>
<script>
    let eventSource = null;
    let accumulatedText = "";
    let lastResponse = "";
    let currentSpan = null;
    let messageCounter = 0;

    async function generateText() {
        const inputText = document.getElementById("text").value;
        const generatedTextDiv = document.getElementById("generatedText");
        generatedTextDiv.innerHTML = "";
        accumulatedText = "";
        lastResponse = "";
        currentSpan = null;
        messageCounter = 0;

        if (eventSource) {
            eventSource.close();
        }
        const temp = 0.7;
        const top_k_val = 40;
        const top_p_val = 0.0;
        const repetition_penalty_val = 1.2;
        eventSource = new EventSource(`/generate_stream?text=${encodeURIComponent(inputText)}&temp=${temp}&top_k=${top_k_val}&top_p=${top_p_val}&reppenalty=${reppenalty_val}`);
        eventSource.onmessage = function(event) {
            if (event.data === "<END_STREAM>") {
                eventSource.close();
                const currentResponse = accumulatedText.replace("<|endoftext|>", "").replace(re.compile(r'\\s+(?=[.,，。])'), '').trim();
                if (currentResponse === lastResponse.trim()) {
                    accumulatedText = "**Response is repetitive. Please try again or rephrase your query.**";
                } else {
                    lastResponse = currentResponse;
                }
                document.getElementById("generatedText").innerHTML = marked.parse(accumulatedText);
                return;
            }
            try {
                const jsonData = JSON.parse(event.data);
                const token = jsonData.token;
                if (token === "<|endoftext|>" || token === "<END_STREAM>") {
                    return;
                }
                if (token === "<NEW_MESSAGE>") {
                    messageCounter++;
                    if (messageCounter > 1) {
                        generatedTextDiv.innerHTML += "<br><br><hr style='border-top: 1px dashed #8c8b8b; margin-top: 10px; margin-bottom: 10px;'><strong>Continued Response:</strong><br><div id='generatedText_" + messageCounter + "'></div>";
                        generatedTextDiv = document.getElementById("generatedText_" + messageCounter);
                        accumulatedText = "";
                    }
                    return;
                }
                accumulatedText += token + " ";
            } catch (e) {
                console.error("Error parsing SSE data:", event.data, e);
            }
        };
        eventSource.onerror = function(error) {
            console.error("SSE error", error);
            eventSource.close();
        };
        const outputDiv = document.getElementById("output");
        outputDiv.classList.add("show");
    }
</script>
</body>
</html>
"""

TRANSLATION_FOLDER = "./TranslationModel"
TRANSLATION_MODEL_WEIGHTS_FILE = "pytorch_model.bin"
TRANSLATION_MODEL_CONFIG_FILE = "config.json"
TRANSLATION_MODEL_VOCAB_FILE = "sentencepiece.bpe.model"
TRANSLATION_MODEL_WEIGHTS_URL = "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/pytorch_model.bin"
TRANSLATION_MODEL_CONFIG_URL = "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/config.json"
TRANSLATION_MODEL_VOCAB_URL = "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/sentencepiece.bpe.model"
TRANSLATION_MODEL_FILES_URLS = [
    (TRANSLATION_MODEL_WEIGHTS_URL, TRANSLATION_MODEL_WEIGHTS_FILE),
    (TRANSLATION_MODEL_CONFIG_URL, TRANSLATION_MODEL_CONFIG_FILE),
    (TRANSLATION_MODEL_VOCAB_URL, TRANSLATION_MODEL_VOCAB_FILE),
]

CODEGEN_FOLDER = "./CodeGenModel"
CODEGEN_MODEL_NAME = "codegen-350M-multi"
CODEGEN_MODEL_WEIGHTS = "pytorch_model.bin"
CODEGEN_CONFIG = "config.json"
CODEGEN_VOCAB = "vocab.json"
CODEGEN_MERGES = "merges.txt"
CODEGEN_MODEL_WEIGHTS_URL = "https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/pytorch_model.bin"
CODEGEN_CONFIG_URL = "https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/config.json"
CODEGEN_VOCAB_URL = "https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/vocab.json"
CODEGEN_MERGES_URL = "https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/merges.txt"
CODEGEN_FILES_URLS = [
    (CODEGEN_MODEL_WEIGHTS_URL, CODEGEN_MODEL_WEIGHTS),
    (CODEGEN_CONFIG_URL, CODEGEN_CONFIG),
    (CODEGEN_VOCAB_URL, CODEGEN_VOCAB),
    (CODEGEN_MERGES_URL, CODEGEN_MERGES),
]

TTS_FOLDER = "./TTSModel"
TTS_MODEL_NAME = "vits"
TTS_MODEL_CONFIG = "config.json"
TTS_MODEL_WEIGHTS = "pytorch_model.bin"
TTS_VOCAB = "vocab.json"
TTS_CONFIG_URL = "https://huggingface.co/kakao-enterprise/vits-vctk/resolve/main/config.json"
TTS_MODEL_WEIGHTS_URL = "https://huggingface.co/kakao-enterprise/vits-vctk/resolve/main/pytorch_model.bin"
TTS_VOCAB_URL = "https://huggingface.co/kakao-enterprise/vits-vctk/resolve/main/vocab.json"
TTS_FILES_URLS = [
    (TTS_CONFIG_URL, TTS_MODEL_CONFIG),
    (TTS_MODEL_WEIGHTS_URL, TTS_MODEL_WEIGHTS),
    (TTS_VOCAB_URL, TTS_VOCAB),
]

STT_FOLDER = "./STTModel"
STT_MODEL_NAME = "wav2vec2"
STT_MODEL_WEIGHTS = "pytorch_model.bin"
STT_CONFIG = "config.json"
STT_VOCAB = "vocab.json"
STT_MODEL_WEIGHTS_URL = "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin"
STT_CONFIG_URL = "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json"
STT_VOCAB_URL = "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json"
STT_FILES_URLS = [
    (STT_MODEL_WEIGHTS_URL, STT_MODEL_WEIGHTS),
    (STT_CONFIG_URL, STT_CONFIG),
    (STT_VOCAB_URL, STT_VOCAB),
]

SENTIMENT_FOLDER = "./SentimentModel"
SENTIMENT_MODEL_WEIGHTS = "pytorch_model.bin"
SENTIMENT_VOCAB = "sentiment_vocab.json"
SENTIMENT_CONFIG = "config.json"
SENTIMENT_MODEL_WEIGHTS_URL = "https://huggingface.co/cardiffnlp/distilroberta-base-sentiment/resolve/main/pytorch_model.bin"
SENTIMENT_VOCAB_URL = "https://huggingface.co/cardiffnlp/distilroberta-base-sentiment/resolve/main/vocab.json"
SENTIMENT_CONFIG_URL = "https://huggingface.co/cardiffnlp/distilroberta-base-sentiment/resolve/main/config.json"
SENTIMENT_FILES_URLS = [
    (SENTIMENT_MODEL_WEIGHTS_URL, SENTIMENT_MODEL_WEIGHTS),
    (SENTIMENT_VOCAB_URL, SENTIMENT_VOCAB),
    (SENTIMENT_CONFIG_URL, SENTIMENT_CONFIG),
]

IMAGEGEN_FOLDER = "./ImageGenModel"
IMAGEGEN_MODEL_WEIGHTS = "diffusion_pytorch_model.bin"
IMAGEGEN_CONFIG = "config.json"
IMAGEGEN_MODEL_WEIGHTS_URL = "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin"
IMAGEGEN_CONFIG_URL = "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json"
IMAGEGEN_FILES_URLS = [
    (IMAGEGEN_MODEL_WEIGHTS_URL, IMAGEGEN_MODEL_WEIGHTS),
    (IMAGEGEN_CONFIG_URL, IMAGEGEN_CONFIG),
]

LIPSYNC_FOLDER = "./LipSyncModel"
LIPSYNC_MODEL_WEIGHTS = "lipsync_expert.pth"
LIPSYNC_MODEL_WEIGHTS_URL = "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Flipsync%5Fexpert%2Epth"
LIPSYNC_FILES_URLS = [
    (LIPSYNC_MODEL_WEIGHTS_URL, LIPSYNC_MODEL_WEIGHTS),
]

WAV2LIP_FOLDER = "./Wav2LipModel"
WAV2LIP_MODEL_WEIGHTS = "wav2lip_gan.pth"
WAV2LIP_MODEL_WEIGHTS_URL = "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Fwav2lip%5Fgan%2Epth"
WAV2LIP_FILES_URLS = [
    (WAV2LIP_MODEL_WEIGHTS_URL, WAV2LIP_MODEL_WEIGHTS),
]

MUSICGEN_FOLDER = "./MusicGenModel"
MUSICGEN_MODEL_NAME = "melody"
MUSICGEN_MODEL_WEIGHTS = "pytorch_model.bin"
MUSICGEN_CONFIG = "config.json"
MUSICGEN_SAMPLE_RATE = 32000
MUSICGEN_DURATION = 8
MUSICGEN_MODEL_WEIGHTS_URL = "https://huggingface.co/facebook/musicgen-small/resolve/main/pytorch_model.bin"
MUSICGEN_CONFIG_URL = "https://huggingface.co/facebook/musicgen-small/resolve/main/config.json"
MUSICGEN_FILES_URLS = [
    (MUSICGEN_MODEL_WEIGHTS_URL, MUSICGEN_MODEL_WEIGHTS),
    (MUSICGEN_CONFIG_URL, MUSICGEN_CONFIG),
]

CODEGEN_SPM_URL = "https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/spm.model"
CODEGEN_SPM = "spm.model"

TRANSLATION_SPM_URL = "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/sentencepiece.bpe.model"
TRANSLATION_SPM = "sentencepiece.bpe.model"

TEXT_TO_VIDEO_FOLDER = "./TextToVideoModel"
TEXT_TO_VIDEO_MODEL_WEIGHTS = "pytorch_model.bin"
TEXT_TO_VIDEO_CONFIG = "config.json"
TEXT_TO_VIDEO_VOCAB = "vocab.json"
TEXT_TO_VIDEO_MODEL_WEIGHTS_URL = "https://huggingface.co/Searchium-ai/clip4clip-webvid150k/resolve/main/pytorch_model.bin"
TEXT_TO_VIDEO_CONFIG_URL = "https://huggingface.co/Searchium-ai/clip4clip-webvid150k/resolve/main/config.json"
TEXT_TO_VIDEO_VOCAB_URL = "https://huggingface.co/Searchium-ai/clip4clip-webvid150k/resolve/main/vocab.json"
TEXT_TO_VIDEO_FILES_URLS = [
    (TEXT_TO_VIDEO_MODEL_WEIGHTS_URL, TEXT_TO_VIDEO_MODEL_WEIGHTS),
    (TEXT_TO_VIDEO_CONFIG_URL, TEXT_TO_VIDEO_CONFIG),
    (TEXT_TO_VIDEO_VOCAB_URL, TEXT_TO_VIDEO_VOCAB),
]

SUMMARIZATION_FOLDER = "./SummarizationModel"
SUMMARIZATION_MODEL_WEIGHTS = "pytorch_model.bin"
SUMMARIZATION_CONFIG = "config.json"
SUMMARIZATION_VOCAB = "vocab.json"
SUMMARIZATION_MODEL_WEIGHTS_URL = "https://huggingface.co/facebook/bart-large-cnn/resolve/main/pytorch_model.bin"
SUMMARIZATION_CONFIG_URL = "https://huggingface.co/facebook/bart-large-cnn/resolve/main/config.json"
SUMMARIZATION_VOCAB_URL = "https://huggingface.co/facebook/bart-large-cnn/resolve/main/vocab.json"
SUMMARIZATION_FILES_URLS = [
    (SUMMARIZATION_MODEL_WEIGHTS_URL, SUMMARIZATION_MODEL_WEIGHTS),
    (SUMMARIZATION_CONFIG_URL, SUMMARIZATION_CONFIG),
    (SUMMARIZATION_VOCAB_URL, SUMMARIZATION_VOCAB),
]

IMAGE_TO_3D_FOLDER = "./ImageTo3DModel"
IMAGE_TO_3D_MODEL_WEIGHTS = "pytorch_model.bin"
IMAGE_TO_3D_CONFIG = "config.json"
IMAGE_TO_3D_MODEL_URL = "https://huggingface.co/zxhezexin/openlrm-obj-base-1.1/resolve/main/pytorch_model.bin"
IMAGE_TO_3D_CONFIG_URL = "https://huggingface.co/zxhezexin/openlrm-obj-base-1.1/resolve/main/config.json"
IMAGE_TO_3D_FILES_URLS = [
    (IMAGE_TO_3D_MODEL_URL, IMAGE_TO_3D_MODEL_WEIGHTS),
    (IMAGE_TO_3D_CONFIG_URL, IMAGE_TO_3D_CONFIG),
]


state_dict = None
enc = None
config = None
model = None
device = torch.device("cpu")
news_clf = None
tfidf_vectorizer = None
text_queue = queue.Queue()
categories = None
is_training = False
background_threads = []
feedback_queue = queue.Queue()
reasoning_queue = queue.Queue()
seen_responses = set()
tts_model = None
stt_model = None
sentiment_model = None
imagegen_model = None
lipsync_model = None
wav2lip_model = None
musicgen_model = None
translation_model = None
codegen_model = None
text_to_video_model = None
summarization_model = None
image_to_3d_model = None
tts_pipeline = False
stt_pipeline = False
sentiment_pipeline = False
imagegen_pipeline = False
translation_pipeline = False
codegen_pipeline = False
text_to_video_pipeline = False
summarization_pipeline = False
image_to_3d_pipeline = False
stt_tokenizer = None
stt_processor = None
sentiment_tokenizer = None
sentiment_model_instance = None
imagegen_vae = None
imagegen_unet = None
imagegen_scheduler = None
musicgen_model_instance = None
musicgen_tokenizer = None
musicgen_processor = None
translation_model_instance = None
translation_tokenizer = None
codegen_model_instance = None
codegen_tokenizer = None
codegen_sp = None
translation_sp = None
text_to_video_tokenizer = None
text_to_video_model_instance = None
summarization_tokenizer = None
summarization_model_instance = None
image_to_3d_config = None
image_to_3d_model_instance = None
app = Flask(__name__)
CORS(app)

from gpt2_pytorch import *
from tts_vits import *
from stt_wav2vec2 import *
from sentiment_roberta import *
from imagegen_vae_unet import *
from musicgen_torch import *
from translation_mbart import *
from codegen_torch import *
from text_to_video_clip4clip import *
from summarization_bart import *
from image_to_3d_openlrm import *

def download_file(url, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True) # Ensure directory exists
    if not os.path.exists(filename):
        print(f"Downloading {filename} from {url}...")
        try:
            wget.download(url, out=filename) # Specify output filename directly
            print(f"Downloaded {filename} successfully.")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

def ensure_folder_and_files_exist(folder_path, files_urls):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")

    for url, filename in files_urls:
        filepath = os.path.join(folder_path, filename)
        download_file(url, filepath)

def ensure_single_file_exists(folder_path, file_url, filename):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    filepath = os.path.join(folder_path, filename)
    download_file(file_url, filepath)


def ensure_gpt2_files_exist():
    ensure_folder_and_files_exist(GPT2_FOLDER, GPT2_FILES_URLS)

def ensure_translation_files_exist():
    ensure_folder_and_files_exist(TRANSLATION_FOLDER, TRANSLATION_MODEL_FILES_URLS)
    ensure_single_file_exists(TRANSLATION_FOLDER, TRANSLATION_SPM_URL, TRANSLATION_SPM)

def ensure_codegen_files_exist():
    ensure_folder_and_files_exist(CODEGEN_FOLDER, CODEGEN_FILES_URLS)
    ensure_single_file_exists(CODEGEN_FOLDER, CODEGEN_SPM_URL, CODEGEN_SPM)

def ensure_tts_files_exist():
    ensure_folder_and_files_exist(TTS_FOLDER, TTS_FILES_URLS)

def ensure_stt_files_exist():
    ensure_folder_and_files_exist(STT_FOLDER, STT_FILES_URLS)

def ensure_sentiment_files_exist():
    ensure_folder_and_files_exist(SENTIMENT_FOLDER, SENTIMENT_FILES_URLS)

def ensure_imagegen_files_exist():
    ensure_folder_and_files_exist(IMAGEGEN_FOLDER, IMAGEGEN_FILES_URLS)

def ensure_lipsync_files_exist():
    ensure_folder_and_files_exist(LIPSYNC_FOLDER, LIPSYNC_FILES_URLS)

def ensure_wav2lip_files_exist():
    ensure_folder_and_files_exist(WAV2LIP_FOLDER, WAV2LIP_FILES_URLS)

def ensure_musicgen_files_exist():
    ensure_folder_and_files_exist(MUSICGEN_FOLDER, MUSICGEN_FILES_URLS)

def ensure_text_to_video_files_exist():
    ensure_folder_and_files_exist(TEXT_TO_VIDEO_FOLDER, TEXT_TO_VIDEO_FILES_URLS)

def ensure_summarization_files_exist():
    ensure_folder_and_files_exist(SUMMARIZATION_FOLDER, SUMMARIZATION_FILES_URLS)

def ensure_image_to_3d_files_exist():
    ensure_folder_and_files_exist(IMAGE_TO_3D_FOLDER, IMAGE_TO_3D_FILES_URLS)

def ensure_all_model_files_exist(): # Define the function here, before it's called
    ensure_gpt2_files_exist()
    ensure_translation_files_exist()
    ensure_codegen_files_exist()
    ensure_tts_files_exist()
    ensure_stt_files_exist()
    ensure_sentiment_files_exist()
    ensure_imagegen_files_exist()
    ensure_lipsync_files_exist()
    ensure_wav2lip_files_exist()
    ensure_musicgen_files_exist()
    ensure_text_to_video_files_exist()
    ensure_summarization_files_exist()
    ensure_image_to_3d_files_exist()


@app.route("/", methods=['GET'])
async def html_handler():
    return html_code

@app.route("/generate_stream", methods=['GET'])
async def generate_stream_api():
    text_input = request.args.get("text")
    temperature = float(request.args.get("temp", 0.7))
    top_k = int(request.args.get("top_k", 40))
    top_p = float(request.args.get("top_p", 0.0))
    reppenalty = float(request.args.get("reppenalty", 1.2))
    return Response(generate_stream_generator(text_input, temperature, top_k, top_p, reppenalty), mimetype='text/event-stream')

@app.route("/tts", methods=['POST'])
def tts_api():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({"error": "Text is required"}), 400
    output_file = text_to_speech(text)
    if output_file == "Error generating speech.":
        return jsonify({"error": "TTS generation failed"}), 500
    return send_file(output_file, mimetype="audio/wav", as_attachment=True, download_name="output.wav")

@app.route("/stt", methods=['POST'])
def stt_api():
    if 'audio' not in request.files:
        return jsonify({"error": "Audio file is required"}), 400
    audio_file = request.files['audio']
    temp_audio_path = f"temp_audio_{uuid.uuid4()}.wav"
    audio_file.save(temp_audio_path)
    output_file = speech_to_text(temp_audio_path)
    os.remove(temp_audio_path)
    if output_file == "Error transcribing audio.":
        return jsonify({"error": "STT failed"}), 500
    return send_file(output_file, mimetype="text/plain", as_attachment=True, download_name="output.txt")

@app.route("/sentiment", methods=['POST'])
def sentiment_api():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({"error": "Text is required"}), 400
    output_file = analyze_sentiment(text)
    if output_file == "Sentiment model not initialized.":
        return jsonify({"error": "Sentiment analysis failed"}), 500
    return jsonify(output_file)

@app.route("/imagegen", methods=['POST'])
def imagegen_api():
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    output_file = generate_image(prompt)
    if output_file == "Error generating image.":
        return jsonify({"error": "Image generation failed"}), 500
    image_io = BytesIO()
    output_file.save(image_io, 'PNG')
    image_io.seek(0)
    return send_file(image_io, mimetype='image/png', as_attachment=True, download_name="output.png")

@app.route("/musicgen", methods=['POST'])
def musicgen_api():
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    output_file = generate_music(prompt)
    if output_file == "Error generating music.":
        return jsonify({"error": "Music generation failed"}), 500
    return send_file(output_file, mimetype="audio/wav", as_attachment=True, download_name="output.wav")

@app.route("/translation", methods=['POST'])
def translation_api():
    data = request.get_json()
    text = data.get('text')
    target_lang = data.get('target_lang', 'es')
    source_lang = data.get('source_lang', 'en')
    if not text:
        return jsonify({"error": "Text is required"}), 400
    output_file = perform_translation(text, target_language_code=f'{target_lang}_XX', source_language_code=f'{source_lang}_XX')
    if output_file == "Error during translation.":
        return jsonify({"error": "Translation failed"}), 500
    return send_file(output_file, mimetype="text/plain", as_attachment=True, download_name="output_translation.txt")

@app.route("/codegen", methods=['POST'])
def codegen_api():
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    output_file = generate_code(prompt)
    if output_file == "Error generating code.":
        return jsonify({"error": "Code generation failed"}), 500
    return send_file(output_file, mimetype="text/x-python", as_attachment=True, download_name="output.py")

@app.route("/text_to_video", methods=['POST'])
def text_to_video_api():
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    output_file = text_to_video(prompt)
    if output_file == "Error generating video representation.":
        return jsonify({"error": "Text to video failed"}), 500
    return send_file(output_file, mimetype="application/octet-stream", as_attachment=True, download_name="output_video_representation.pt")

@app.route("/summarization", methods=['POST'])
def summarization_api():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({"error": "Text is required"}), 400
    output_file = summarize_text(text)
    if output_file == "Error during summarization.":
        return jsonify({"error": "Summarization failed"}), 500
    return send_file(output_file, mimetype="text/plain", as_attachment=True, download_name="output_summary.txt")

@app.route("/image_to_3d", methods=['POST'])
def image_to_3d_api():
    if 'image' not in request.files:
        return jsonify({"error": "Image file is required"}), 400
    image_file = request.files['image']
    temp_image_path = f"temp_image_{uuid.uuid4()}.png"
    image_file.save(temp_image_path)
    output_file = image_to_3d(temp_image_path)
    os.remove(temp_image_path)
    if output_file == "Error converting image to 3D.":
        return jsonify({"error": "Image to 3D failed"}), 500
    return send_file(output_file, mimetype="model/obj", as_attachment=True, download_name="output_3d.obj")


async def main():
    global background_threads, response_queue
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    response_queue = queue.Queue()

    ensure_all_model_files_exist()
    initialize_model()
    await initialize_sklearn()
    initialize_tts_model()
    initialize_stt_model()
    initialize_sentiment_model()
    initialize_imagegen_model()
    ensure_lipsync_files_exist()
    ensure_wav2lip_files_exist()
    initialize_musicgen_model()
    initialize_translation_model()
    initialize_codegen_model()
    initialize_text_to_video_model()
    initialize_summarization_model()
    initialize_image_to_3d_model()

    background_threads.append(threading.Thread(target=generate_and_queue_text, args=('en',), daemon=True))
    background_threads.append(threading.Thread(target=generate_and_queue_text, args=('es',), daemon=True))
    background_threads.append(threading.Thread(target=background_training, daemon=True))
    for thread in background_threads:
        thread.start()

    asyncio.create_task(background_reasoning_queue())

    app.run(host="127.0.0.1", port=7860, debug=False)

if __name__ == '__main__':
    asyncio.run(main())
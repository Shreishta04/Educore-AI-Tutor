from PIL import Image, ImageDraw
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import requests
import gradio as gr
import os
from gtts import gTTS
import tempfile
import PyPDF2
import docx
import easyocr
import speech_recognition as sr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load Models
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
reader = easyocr.Reader(['en'], gpu=False)

# Groq API
def call_groq_llm(prompt, model_name="meta-llama/llama-4-maverick-17b-128e-instruct"):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "❌ Error: Missing Groq API key. Please set it in your .env file."

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful AI tutor."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
    return (
        response.json()['choices'][0]['message']['content']
        if response.status_code == 200
        else f"❌ Groq Error: {response.status_code}\n{response.text}"
    )

# Text-to-Speech
def generate_tts(text):
    tts = gTTS(text=text, lang='en')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# Object Detection
def detect_objects(image: Image.Image):
    inputs = detr_processor(images=image, return_tensors="pt")
    outputs = detr_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    draw = ImageDraw.Draw(image)
    labels = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_text = detr_model.config.id2label[label.item()]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), label_text, fill="red")
        labels.append(label_text)
    return image, ", ".join(set(labels))

# Image Analysis
def analyze_image(image: Image.Image, task: str):
    label_summary = ""
    if task == "Object Description":
        image, label_summary = detect_objects(image)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    raw_caption = processor.decode(output[0], skip_special_tokens=True)
    temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    image.save(temp_image_path)
    extracted_text = " ".join(reader.readtext(temp_image_path, detail=0)).strip()

    if task == "Object Description":
        prompt = f"Detected Objects: {label_summary}\nCaption: \"{raw_caption}\"\n\nPlease explain this image:\n- What these objects are\n- How they work or are used\n- Why they matter"
    else:
        if not extracted_text:
            return "❌ No readable text found in image.", None, None, None
        prompt = f"Extracted Text: \"{extracted_text}\"\n\nYou are an AI tutor. Please explain this content from an image of textbook/notes in a student-friendly way."

    explanation = call_groq_llm(prompt)
    audio_path = generate_tts(explanation)
    return f"{raw_caption}\n\n{extracted_text}", explanation, audio_path, explanation

# PDF/DOCX Processing
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return "".join(page.extract_text() for page in reader.pages)

def extract_text_from_docx(docx_file):
    return "\n".join([para.text for para in docx.Document(docx_file).paragraphs])

def tutor_from_document(file):
    ext = os.path.splitext(file.name)[-1].lower()
    text = extract_text_from_pdf(file) if ext == ".pdf" else extract_text_from_docx(file) if ext == ".docx" else None
    if not text:
        return "❌ Unsupported file format.", None, None
    prompt = f"You are an AI tutor. Please explain the following content in a student-friendly way:\n\n{text}"
    explanation = call_groq_llm(prompt)
    audio_path = generate_tts(explanation)
    return explanation, audio_path, explanation

# Speech Recognition
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "❌ Sorry, I couldn't understand the audio."
        except sr.RequestError as e:
            return f"❌ Speech recognition error: {str(e)}"

# Chatbot
def chat_ai_tutor(history, user_input, context):
    context_text = f"Use the following context:\n{context}" if context else ""
    prompt = f"You are an AI tutor helping a student. {context_text}\n\nQuestion:\n{user_input}\n\nProvide a clear answer and one helpful suggestion."
    response = call_groq_llm(prompt)
    history.append((user_input, response))
    return history, ""

# Gradio Interface
with gr.Blocks(css="""
    .gr-button {width: 150px; margin: 0 auto; display: block;}
    .gr-button:hover {background-color: #4CAF50; color: white;}
    .gr-radio {margin: auto; text-align: center;}
    .gr-textbox, .gr-audio, .gr-file {margin-top: 10px;}
    .gr-chatbot {margin-top: 10px;}
""") as demo:

    gr.HTML("""
        <h1 style="text-align: center; font-size: 2.5em; font-weight: bold; color: #2c3e50; margin-bottom: 20px;">
            🎓 EduCore AI Tutor
        </h1>
    """)

    context_state = gr.State("")

    with gr.Tab("📷 Upload Image"):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                task_choice = gr.Radio(
                    choices=["Object Description", "Text Explanation"],
                    value="Object Description",
                    label="Choose Task"
                )
                img = gr.Image(type="pil", label="Upload Image")
                btn = gr.Button("🔍 Analyze Image")
            with gr.Column(scale=2):
                combined_output = gr.Textbox(label="✏ Caption")
                explanation = gr.Textbox(label="📘 Explanation")
                audio = gr.Audio(label="🔊 Listen", type="filepath")

        btn.click(analyze_image, inputs=[img, task_choice], outputs=[combined_output, explanation, audio, context_state])

    with gr.Tab("📄 Upload Document (PDF/DOCX)"):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                doc = gr.File(label="Upload PDF or DOCX")
                doc_btn = gr.Button("📚 Explain Document")
            with gr.Column(scale=2):
                doc_out = gr.Textbox(label="📘 Explanation")
                doc_audio = gr.Audio(label="🔊 Listen", type="filepath")

        doc_btn.click(tutor_from_document, inputs=doc, outputs=[doc_out, doc_audio, context_state])

    with gr.Tab("💬 Ask EduCore AI Tutor"):
        with gr.Row():
            chatbot = gr.Chatbot(label="EduCore Chat")
        with gr.Row():
            chat_input = gr.Textbox(placeholder="Ask a question...", label="Your Question")
            mic_input = gr.Audio(type="filepath", label="🎙️ Speak Your Question")
        with gr.Row():
            chat_btn = gr.Button("💡 Get Answer")
            mic_btn = gr.Button("🎤 Transcribe & Ask")

        chat_btn.click(chat_ai_tutor, inputs=[chatbot, chat_input, context_state], outputs=[chatbot, chat_input])
        mic_btn.click(transcribe_audio, inputs=mic_input, outputs=chat_input).then(
            chat_ai_tutor, inputs=[chatbot, chat_input, context_state], outputs=[chatbot, chat_input]
        )

demo.launch()

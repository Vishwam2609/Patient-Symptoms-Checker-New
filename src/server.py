# --- Begin Monkey Patches (must be at the very top) ---
import transformers
import torch

if not hasattr(torch, "compiler"):
    class DummyCompiler:
        @staticmethod
        def disable(recursive=False):
            def decorator(func):
                return func
            return decorator
    torch.compiler = DummyCompiler()

old_load_state_dict = transformers.modeling_utils.PreTrainedModel.load_state_dict

def new_load_state_dict(self, state_dict, strict=True, **kwargs):
    if "assign" in kwargs:
        del kwargs["assign"]
    return old_load_state_dict(self, state_dict, strict=strict, **kwargs)

transformers.modeling_utils.PreTrainedModel.load_state_dict = new_load_state_dict
# --- End Monkey Patches ---

import os
import logging
import gc
import tempfile
import shutil
import re
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from gtts import gTTS
import torch

# Use GPU if available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Disable flex attention and set a dedicated cache directory.
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"
CACHE_DIR = "/tmp/transformers_cache"
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

NGROK_AUTH_TOKEN = "2sZL5k5FBMPppi3zC5xRRYuG5IP_6BiBZ5A9ee77WTxAfVWqa"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# LLMHandler for key symptom extraction, follow-up question generation, and guideline generation.
class LLMHandler:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.pipeline = None

    def clear_model_cache(self):
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)
            logger.info(f"Cleared Transformers cache at {CACHE_DIR}")

    def load_model(self):
        if self.pipeline is None:
            try:
                logger.info("Clearing model cache before loading LLM...")
                self.clear_model_cache()
                logger.info("Loading LLM model...")
                HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN", "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")
                LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    LLM_MODEL_NAME,
                    use_auth_token=HUGGING_FACE_TOKEN,
                    force_download=True,
                    use_fast=False,
                    trust_remote_code=True,
                    cache_dir=CACHE_DIR
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL_NAME,
                    use_auth_token=HUGGING_FACE_TOKEN,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    force_download=True,
                    trust_remote_code=True,
                    cache_dir=CACHE_DIR
                )
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if DEVICE == "cuda" else -1
                )
                logger.info("LLM model loaded successfully.")
            except Exception as e:
                logger.error("Error loading LLM model", exc_info=True)
                raise e

    def generate_text(self, prompt, max_new_tokens, num_beams, temperature, repetition_penalty, early_stopping=True) -> str:
        if self.pipeline is None:
            self.load_model()
        if self.pipeline is None:
            logger.error("LLM pipeline is not available after loading.")
            return ""
        try:
            logger.info("Generating text from LLM...")
            pipeline_output = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=early_stopping,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            ) or []
            result = list(pipeline_output)
            if result and isinstance(result[0], dict):
                generated_text = result[0].get('generated_text', "")
            else:
                generated_text = str(result[0]) if result else ""
            return generated_text.strip() if generated_text is not None else ""
        except Exception as e:
            logger.error("LLM generation error", exc_info=True)
            return ""

    def generate_guidelines(self, reviewed_transcript: str, key_symptom: str, static_followup: list, dynamic_followup: list) -> str:
        """
        Generates a concise, patient-friendly home care plan in one paragraph of approximately 150 words.
        The guidelines consider the patient's reviewed transcript, extracted key symptom, static follow-up Q&A,
        and dynamic follow-up Q&A. The plan focuses on specific home care strategies the patient can follow
        to get relief from the reported symptom.
        """
        static_followup_text = "\n".join(
            [f"Q: {item['question']}\nA: {item['answer']}" for item in static_followup]
        )
        dynamic_followup_text = "\n".join(
            [f"Q: {item['question']}\nA: {item['answer']}" for item in dynamic_followup]
        )
        
        detailed_context = (
            "Conversation so far:\n"
            f"Patient's description: {reviewed_transcript}\n"
            f"Key symptom: {key_symptom}\n"
            "Static Follow-up Q&A:\n" + (static_followup_text or "None") + "\n"
            "Dynamic Follow-up Q&A:\n" + (dynamic_followup_text or "None") + "\n"
        )
        
        prompt = (
            "You are a caring doctor speaking directly to a patient with limited medical knowledge. "
            "Based solely on the conversation below, please provide a complete and concise Personalized Home Care Plan in one paragraph. "
            "Focus on specific home care strategies the patient can follow to get relief from the reported symptom. "
            "Ensure that your answer is to the point, covers all essential aspects, and is written in plain language. "
            "Your answer should be approximately 150 words.\n\n"
            f"{detailed_context}\n\n"
            "Your Personalized Home Care Plan:"
        )
        
        result = self.generate_text(
            prompt,
            max_new_tokens=500,
            num_beams=5,
            temperature=0.5,
            repetition_penalty=1.0,
            early_stopping=True
        )
        if "Your Personalized Home Care Plan:" in result:
            result = result.split("Your Personalized Home Care Plan:")[-1].strip()
        
        # Remove extra whitespace and split into words.
        paragraph = " ".join(result.split())
        words = paragraph.split()
        
        # Post-process to target approximately 150 words.
        if len(words) < 150:
            while len(words) < 150:
                words.append(words[-1])
        elif len(words) > 150:
            words = words[:150]
        
        final_paragraph = " ".join(words)
        if final_paragraph[-1] not in ".!?":
            final_paragraph += "."
        
        import re
        # Remove enumeration prefixes like "1. ", "2. ", etc.
        final_paragraph = re.sub(r"^\d+\.\s+", "", final_paragraph)
        final_paragraph = re.sub(r"(\s)\d+\.\s+", r"\1", final_paragraph)
        # Remove any bolded section headings e.g., "**Nausea Management**:".
        final_paragraph = re.sub(r"\*\*.*?\*\*\s*:\s*", "", final_paragraph)
        
        return final_paragraph

    def generate_followup_questions(self, reviewed_transcript: str, key_symptom: str, static_followup: list) -> list:
        """
        Dynamically generates exactly three distinct follow-up questions based on the patient's reviewed transcript,
        the extracted key symptom, and the static follow-up Q&A (with answers provided for context).
        The generated questions must be medically relevant and address different aspects of the patient's condition.
        They must be stand-alone, written in plain language, and should not include any static answers, greetings, or extraneous commentary.
        Output exactly three questions, one per line, each ending with a question mark.
        """
        # Build a context that includes static Q&A details, which are provided only for reference.
        static_context = "\n".join(
            [f"Q: {item['question']}\nA: {item['answer']}" for item in static_followup]
        )
        
        context = (
            f"Patient's description: {reviewed_transcript}\n"
            f"Key symptom: {key_symptom}\n"
            "Static Follow-up Q&A (for context only, do not repeat these answers):\n" 
            + (static_context if static_context else "None") + "\n"
        )
        
        prompt = (
            "You are a caring doctor who is experienced in asking insightful follow-up questions. "
            "Based solely on the context below, generate exactly three distinct follow-up questions that are medically relevant. "
            "Focus on asking for additional details that are not already addressed in the static follow-up Q&A. "
            "Do not include any answers, commentary, or letter prefixes (e.g., 'A:', 'B:', 'C:'). "
            "Your output should contain only three complete question sentences, each beginning with an interrogative word or auxiliary verb (e.g., What, How, Do, Does, Could) and ending with a question mark.\n\n"
            f"Context:\n{context}\n"
            "### OUTPUT:\n"
        )
        
        result = self.generate_text(prompt, max_new_tokens=150, num_beams=5, temperature=0.7, repetition_penalty=1.2)
        
        # If the output label is present, extract the text after it.
        if "### OUTPUT:" in result:
            result = result.split("### OUTPUT:")[-1].strip()
        
        # Split the result into individual lines.
        questions = [line.strip() for line in result.split("\n") if line.strip()]
        
        import re
        cleaned_questions = []
        for q in questions:
            # Remove any letter prefixes such as "A:", "B:", "C:".
            q = re.sub(r'^[A-Z]:\s*', '', q)
            # Remove any unwanted dynamic follow-up numbering if present.
            q = re.sub(r'^Dynamic Follow-Up\s*\(\s*\d+\s+of\s+\d+\s*\)\s*', '', q)
            # Remove any extraneous numbering or bullet prefixes.
            q = re.sub(r'^[\d\.\-\)\s]+', '', q)
            cleaned_questions.append(q)
        
        # Ensure each question ends with a question mark.
        cleaned_questions = [q if q.endswith('?') else q + '?' for q in cleaned_questions]
        
        # Remove duplicate questions while preserving order.
        seen = set()
        distinct_questions = []
        for q in cleaned_questions:
            if q not in seen:
                seen.add(q)
                distinct_questions.append(q)
        
        # Return exactly three distinct questions.
        return distinct_questions[:3]

llm_handler = LLMHandler()

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please upload an audio file with key 'file'."}), 400

    audio_file = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
        audio_file.save(temp_audio.name)
        try:
            model = whisper.load_model("small.en", device=DEVICE)
        except Exception as e:
            logger.error("Whisper model load failed", exc_info=True)
            return jsonify({"error": f"Model load failed: {str(e)}"}), 500

        try:
            result = model.transcribe(temp_audio.name, fp16=False)
            logger.info(f"Raw transcription result: {result}")
        except Exception as e:
            logger.error("Transcription failed", exc_info=True)
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

        transcript_raw = result.get("text", "")
        transcript = transcript_raw.strip() if not isinstance(transcript_raw, list) else " ".join(transcript_raw).strip()
        transcript = re.sub(r'\s+', ' ', transcript)
        transcript = re.sub(r'[^\w\s.,!?]', '', transcript)
        logger.info(f"Processed transcript: '{transcript}'")
        if not transcript:
            return jsonify({"error": "No transcript was generated. Please ensure the audio is clear and ffmpeg is installed."}), 500

        del model
        gc.collect()

    return jsonify({"transcript": transcript})

@app.route("/extract_symptoms", methods=["POST"])
def extract_symptoms():
    data = request.get_json()
    if not data or "transcript" not in data:
        return jsonify({"error": "No transcript provided. Please provide a transcript in the request body."}), 400

    transcript = data["transcript"].strip()
    if not transcript:
        return jsonify({"error": "The provided transcript is empty."}), 400

    try:
        key_symptom = llm_handler.generate_text(
            "You are a caring doctor. Based on the following patient description, extract the key symptom in one short word or phrase.\n\n"
            f"Patient Description: {transcript}\n\n"
            "Answer:",
            max_new_tokens=20, num_beams=3, temperature=0.7, repetition_penalty=1.2
        )
        if "Answer:" in key_symptom:
            key_symptom = key_symptom.split("Answer:")[-1].strip()
    except Exception as e:
        logger.error("Key symptom extraction error", exc_info=True)
        return jsonify({"error": f"Key symptom extraction error: {str(e)}"}), 500

    logger.info(f"Extracted key symptom (internal): '{key_symptom}'")
    return jsonify({"key_symptom": key_symptom})

# Updated endpoint to expect 'reviewed_transcript'
@app.route("/generate_followup_questions", methods=["POST"])
def generate_followup_questions_endpoint():
    data = request.get_json()
    if not data or not all(k in data for k in ["reviewed_transcript", "key_symptom"]):
        return jsonify({"error": "Missing required fields. Please provide reviewed_transcript and key_symptom."}), 400
    
    reviewed_transcript = data["reviewed_transcript"].strip()
    key_symptom = data["key_symptom"].strip()
    static_followup = data.get("static_followup", [])
    
    try:
        questions = llm_handler.generate_followup_questions(reviewed_transcript, key_symptom, static_followup)
    except Exception as e:
        logger.error("Dynamic follow-up question generation error", exc_info=True)
        return jsonify({"error": f"Dynamic follow-up question generation error: {str(e)}"}), 500
    return jsonify({"follow_up_questions": questions})

@app.route("/generate_guidelines", methods=["POST"])
def generate_guidelines():
    data = request.get_json()
    if not data or not all(k in data for k in ["transcript", "key_symptom"]):
        return jsonify({"error": "Missing required fields. Please provide transcript and key_symptom."}), 400

    transcript = data.get("transcript", "").strip()
    key_symptom = data.get("key_symptom", "").strip()
    # Use empty lists if follow_up or dynamic_followup are not provided.
    static_followup = data.get("follow_up", [])
    dynamic_followup = data.get("dynamic_followup", [])

    # (Optional: Validate that transcript and key_symptom are not empty.)
    if not transcript or not key_symptom:
        return jsonify({"error": "Transcript and key_symptom cannot be empty."}), 400

    for item in static_followup:
        if not item.get("answer", "").strip():
            return jsonify({"error": "All static follow-up questions must be answered."}), 400
    for item in dynamic_followup:
        if not item.get("answer", "").strip():
            return jsonify({"error": "All dynamic follow-up questions must be answered."}), 400

    try:
        llm_handler.load_model()
        guidelines_text = llm_handler.generate_guidelines(transcript, key_symptom, static_followup, dynamic_followup)
    except Exception as e:
        logger.error("Guideline generation error", exc_info=True)
        return jsonify({"error": f"Guideline generation error: {str(e)}"}), 500

    logger.info(f"Generated guidelines: '{guidelines_text}'")
    try:
        tts = gTTS(text=guidelines_text, lang='en')
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        audio_base64 = base64.b64encode(audio_io.read()).decode('utf-8')
        audio_data_url = f"data:audio/mp3;base64,{audio_base64}"
    except Exception as e:
        logger.error("TTS conversion error", exc_info=True)
        audio_data_url = ""

    return jsonify({"guidelines": guidelines_text, "audio": audio_data_url})

if __name__ == "__main__":
    try:
        public_url = ngrok.connect("5000")
        print("Public URL:", public_url)
    except Exception as ngrok_error:
        print("Ngrok error:", ngrok_error)
    app.run(host="0.0.0.0", port=5000)
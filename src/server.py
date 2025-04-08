import os
import logging
import gc
import tempfile
import shutil
import re
import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import whisper
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from gtts import gTTS

# --- Configuration and Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"
CACHE_DIR = "/tmp/transformers_cache"
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
# Ensure CORS allows the specific origin and necessary headers like ngrok's bypass warning
CORS(app, resources={r"/*": {
    "origins": "http://localhost:5173",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "ngrok-skip-browser-warning"]
}})


NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "2sZL5k5FBMPppi3zC5xRRYuG5IP_6BiBZ5A9ee77WTxAfVWqa") # Replace with your token or set ENV VAR
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

request_counter = 0

executor = ThreadPoolExecutor(max_workers=2)

# --- LLM Handler Class ---
class LLMHandler:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.is_model_loaded = False

    def clear_model_cache(self):
        if os.path.exists(CACHE_DIR):
            try:
                shutil.rmtree(CACHE_DIR)
                os.makedirs(CACHE_DIR, exist_ok=True)
                logger.info(f"Cleared Transformers cache at {CACHE_DIR}")
            except OSError as e:
                logger.error(f"Error clearing cache directory {CACHE_DIR}: {e}")
        else:
            os.makedirs(CACHE_DIR, exist_ok=True)

    def load_model(self):
        if self.is_model_loaded:
            logger.info("LLM model already loaded.")
            return True
        if self.pipeline is not None:
            logger.warning("Pipeline exists but model not marked as loaded. Reloading.")

        try:
            logger.info("Clearing model cache before loading LLM...")
            self.clear_model_cache()
            logger.info("Loading LLM model...")
            # Use environment variables or fallbacks for tokens and model names
            HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN", "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") # Replace or set ENV VAR
            LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025")
            use_auth_token_value = HUGGING_FACE_TOKEN if HUGGING_FACE_TOKEN else None

            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL_NAME,
                token=use_auth_token_value,
                force_download=False,
                use_fast=False, # Consider setting to True if compatibility allows
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )
            if self.tokenizer.pad_token is None:
                # Common practice: set pad_token to eos_token if missing
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.warning("Tokenizer missing pad_token, setting it to eos_token.")

            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                token=use_auth_token_value,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                force_download=False,
                trust_remote_code=True,
                cache_dir=CACHE_DIR
                # Consider adding device_map="auto" for multi-GPU or memory optimization
            ).to(DEVICE)

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if DEVICE == "cuda" else -1 # Use device index for CUDA
            )
            self.is_model_loaded = True
            logger.info(f"LLM model '{LLM_MODEL_NAME}' loaded successfully onto {DEVICE}.")
            return True
        except Exception as e:
            logger.error(f"Error loading LLM model '{LLM_MODEL_NAME}'", exc_info=True)
            # Reset state on failure
            self.tokenizer = None
            self.model = None
            self.pipeline = None
            self.is_model_loaded = False
            return False

    def generate_text(self, prompt, max_new_tokens, num_beams, temperature, repetition_penalty, early_stopping=True) -> str:
        global request_counter
        if not self.is_model_loaded or self.pipeline is None:
            logger.warning("LLM model not loaded. Attempting to load now...")
            if not self.load_model():
                logger.error("LLM pipeline is not available because model loading failed.")
                return "Error: Model could not be loaded."

        # Ensure pad_token_id is set, fallback to eos_token_id
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        if pad_token_id is None:
            logger.error("Cannot generate text: EOS and PAD token IDs are None.")
            return "Error: Tokenizer misconfiguration."

        try:
            logger.info(f"Generating text (max_new: {max_new_tokens}, beams: {num_beams}, temp: {temperature}, penalty: {repetition_penalty})")
            # Generate text using the pipeline
            pipeline_output = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=early_stopping,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_token_id, # Explicitly set pad_token_id
                do_sample=True if temperature > 0 else False, # Enable sampling if temperature > 0
                return_full_text=False # Only return the generated part
            ) or [] # Ensure pipeline_output is iterable

            if not pipeline_output:
                logger.warning("LLM pipeline returned an empty result.")
                return ""

            # Extract generated text, handle potential missing key
            generated_text = pipeline_output[0].get('generated_text', "")

            # Optional GPU memory cleanup for CUDA devices
            request_counter += 1
            if DEVICE == "cuda" and request_counter % 10 == 0: # Clean cache every 10 requests
                logger.info(f"Cleaning GPU cache (Request {request_counter})...")
                gc.collect()
                torch.cuda.empty_cache()

            return generated_text.strip() if generated_text else ""

        except torch.cuda.OutOfMemoryError as oom_error:
            logger.error(f"CUDA OutOfMemoryError during text generation: {oom_error}", exc_info=True)
            logger.info("Clearing GPU cache after OOM...")
            gc.collect()
            torch.cuda.empty_cache()
            # Provide a user-friendly error message
            return f"Error: GPU out of memory during generation. Please try again or simplify."
        except Exception as e:
            logger.error("LLM generation error", exc_info=True)
            return f"Error during text generation: {e}"

    def extract_structured_symptoms(self, transcript: str) -> dict:
        prompt = (
            "You are a medical assistant analyzing a patient’s initial statement to identify their main health issue.\n"
            "Your top priority is to determine the 'key_symptom'—the primary health complaint the patient is experiencing (e.g., 'headache', 'chest pain', 'fever').\n"
            "Even if the description is vague, infer the most likely primary symptom based on the context.\n"
            "Extract the following details as accurately as possible:\n"
            "- key_symptom: The primary health complaint (required; do not leave as 'unknown' unless absolutely unclear).\n"
            "- severity: How intense it is (e.g., 'severe', 'mild', '8/10'). Use 'not specified' if missing.\n"
            "- onset_duration: When it began or how long it’s lasted (e.g., 'since yesterday', 'for 2 weeks'). Use 'not specified' if missing.\n"
            "- location: Body part affected (e.g., 'head', 'left arm'). Use 'not specified' if missing.\n"
            "- character: Nature of the symptom (e.g., 'sharp', 'dull', 'throbbing'). Use 'not specified' if missing.\n"
            "- associated_symptoms: Other symptoms mentioned (e.g., 'nausea, fatigue'). Use 'none mentioned' if missing.\n\n"
            f"Patient Description: \"{transcript}\"\n\n"
            "Output ONLY a valid JSON object with these fields. Be concise and ensure 'key_symptom' is always populated if possible.\n\n"
            "JSON Output:"
        )

        # Default summary structure
        summary = {"key_symptom": "unknown symptom", "severity": "not specified", "onset_duration": "not specified", "location": "not specified", "character": "not specified", "associated_symptoms": "none mentioned"}

        MAX_ATTEMPTS = 2
        raw_output = ""
        for attempt in range(MAX_ATTEMPTS):
            # Generate text with slight temperature variation for retries
            raw_output = self.generate_text(prompt, max_new_tokens=300, num_beams=1, temperature=0.2 + (attempt * 0.1), repetition_penalty=1.1)

            try:
                # Try to find a JSON object within the output using regex
                match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
                if not match:
                    logger.warning(f"Attempt {attempt + 1}: No JSON found in LLM output: '{raw_output}'")
                    continue

                # Clean and parse the found JSON string
                cleaned_output = match.group(0).strip()
                parsed_json = json.loads(cleaned_output)

                # Validate parsed structure and update summary
                if isinstance(parsed_json, dict):
                    # Handle potential variations in key names for the primary symptom
                    key_mapping = {"main_symptom": "key_symptom", "primary_symptom": "key_symptom", "chief_complaint": "key_symptom", "main_issue": "key_symptom"}
                    temp_dict = parsed_json.copy()
                    for alt_key, standard_key in key_mapping.items():
                        if alt_key in temp_dict and standard_key not in temp_dict:
                             temp_dict[standard_key] = temp_dict.pop(alt_key)

                    # Update summary with valid, non-empty values from parsed JSON
                    for key in summary.keys():
                        if key in temp_dict and isinstance(temp_dict[key], str) and temp_dict[key].strip():
                             value = temp_dict[key].strip().rstrip('.').lower()
                             # Ignore placeholder values
                             if value not in ["", "none", "n/a", "unknown"]:
                                summary[key] = value
                        elif key in temp_dict and isinstance(temp_dict[key], list) and key == "associated_symptoms":
                             # Handle list of associated symptoms
                             str_list = [str(s).strip() for s in temp_dict[key] if str(s).strip()]
                             if str_list:
                                summary[key] = ", ".join(str_list)

                    # Check if the key symptom was successfully identified
                    if not summary["key_symptom"] or summary["key_symptom"].lower() in ["not specified", "unknown", "", "none"]:
                        logger.warning(f"Attempt {attempt + 1}: Key symptom not identified: {summary}")
                        continue # Try again if key symptom is missing

                    logger.info(f"Successfully extracted symptom summary: {summary}")
                    return summary # Return the populated summary
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Attempt {attempt + 1}: LLM output parsing error: {e}. Raw output: '{raw_output}'")
                continue # Try again on parsing error

        # Fallback mechanism if LLM fails after retries
        logger.warning("LLM failed to extract key symptom after retries. Using fallback analysis.")
        # Try matching common symptoms
        common_symptoms = ["headache", "fever", "pain", "cough", "nausea", "fatigue", "dizziness", "shortness of breath", "chest pain", "stomachache"]
        transcript_lower = transcript.lower()
        for symptom in common_symptoms:
            if symptom in transcript_lower:
                summary["key_symptom"] = symptom
                logger.info(f"Fallback extracted key_symptom: {symptom}")
                break
        # Last resort: use the first likely noun/verb if no common symptom found
        if summary["key_symptom"] == "unknown symptom" and transcript.strip():
            words = transcript_lower.split()
            # Basic check to avoid common conversational words
            ignore_words = {"i", "have", "had", "for", "since", "a", "the", "and", "with", "my", "feel", "feeling", "it's", "is", "am", "very"}
            for word in words:
                 word_clean = word.strip('.,?!')
                 if word_clean not in ignore_words and len(word_clean)>2:
                    summary["key_symptom"] = word_clean
                    logger.info(f"Last resort key_symptom: {word_clean}")
                    break

        return summary

    def generate_followup_questions(self, reviewed_transcript: str, symptom_summary: dict, static_followup: list, dynamic_followup_history: list = []) -> list:
        """
        Generates a batch of 5 diverse, unique follow-up questions tailored to the conversation history,
        avoiding repetition of example questions and ensuring plain text output.
        """
        if not self.is_model_loaded or self.pipeline is None:
            logger.error("Cannot generate follow-up questions: LLM not loaded.")
            return []

        key_symptom = symptom_summary.get('key_symptom', 'the main issue')
        if key_symptom in ["unknown symptom", "not specified", "", None]:
            key_symptom = "the main health issue"

        # Compile detailed conversation history
        initial_summary_text = (
            f"- Initial Complaint: {symptom_summary.get('key_symptom', 'Not specified')}\n"
            f"- Initial Severity: {symptom_summary.get('severity', 'Not specified')}\n"
            f"- Initial Onset/Duration: {symptom_summary.get('onset_duration', 'Not specified')}\n"
            f"- Initial Location: {symptom_summary.get('location', 'Not specified')}\n"
            f"- Initial Character: {symptom_summary.get('character', 'Not specified')}\n"
            f"- Initial Associated Symptoms: {symptom_summary.get('associated_symptoms', 'Not specified')}"
        )
        static_followup_text = "\n".join(
            f"- Q: {item.get('question', '').strip()}\n  A: {item.get('answer', 'Not answered yet').strip()}"
            for item in static_followup if isinstance(item, dict) and item.get('question')
        )
        dynamic_followup_text = "\n".join(
            f"- Q: {item.get('question', '').strip()}\n  A: {item.get('answer', 'Not answered yet').strip()}"
            for item in dynamic_followup_history if isinstance(item, dict) and item.get('question')
        )
        full_context = (
            "## Patient Conversation History:\n\n"
            f"### Initial Statement & Summary:\nPatient Said: \"{reviewed_transcript}\"\nSummary:\n{initial_summary_text}\n\n"
            f"### Previous Static Questions Asked:\n{static_followup_text if static_followup_text else 'None asked植物yet.'}\n\n"
            f"### Previous Follow-up Questions Asked:\n{dynamic_followup_text if dynamic_followup_text else 'None asked yet.'}\n"
        )

        # Define example questions to avoid copying
        example_questions_to_avoid = {
            "how long does an episode typically last?",
            "is there anything that reliably makes it feel better?",
            "have you experienced any dizziness along with this?",
            "could you describe the sensation in a different way?",
            "does it affect your ability to concentrate?"
        }

        # Updated prompt to prevent copying examples and encourage unique questions
        prompt = (
            "You are a friendly and meticulous virtual health assistant conducting an interview.\n"
            f"The patient's primary concern seems to be '{key_symptom}'.\n"
            "Review the **entire conversation history** provided below very carefully.\n"
            "Your goal is to generate exactly **five diverse follow-up questions** tailored specifically to the patient’s situation, "
            "to gain a deeper and broader understanding of their condition based on the conversation history.\n\n"
            "**Instructions for Generating the Five Questions:**\n"
            "1. Base Questions on Context: Use the patient’s specific statements and summary details (e.g., severity, onset, location) to create relevant questions.\n"
            "2. Mix Question Types: Include questions that explore specific details, cover unaddressed aspects (e.g., triggers, impact, associated symptoms), and clarify vague points if present.\n"
            "3. Avoid Example Questions: Do NOT copy or closely paraphrase the example questions provided below. They are for format guidance only.\n"
            "4. Be Empathetic & Simple: Use clear, caring language. Avoid medical jargon.\n"
            "5. Ensure Uniqueness: Each question must differ substantially from all previously asked questions and the examples.\n"
            "6. Output Format: Provide ONLY the five questions as plain text, one per line, WITHOUT labels, numbering, bullets, bold text, or quotes.\n\n"
            "**Example Format (Do NOT Copy These Questions):**\n"
            "How long does an episode typically last?\n"
            "Is there anything that reliably makes it feel better?\n"
            "Have you experienced any dizziness along with this?\n"
            "Could you describe the sensation in a different way?\n"
            "Does it affect your ability to concentrate?\n\n"
            f"{full_context}\n"
            "---\n"
            "Next Five Follow-up Questions:"
        )

        # Generation and Validation Logic
        MAX_ATTEMPTS = 3
        questions = []
        generated_output = ""
        for attempt in range(MAX_ATTEMPTS):
            generated_output = self.generate_text(
                prompt,
                max_new_tokens=400,
                num_beams=3,
                temperature=0.85 + (attempt * 0.05),  # Increased base temperature for more creativity
                repetition_penalty=1.3                # Higher penalty to discourage repetition
            )

            # Robust Parsing Logic
            potential_lines = [line.strip() for line in generated_output.split('\n') if line.strip()]
            extracted_questions = []
            seen_questions_lower = {item['question'].strip().lower() for item in static_followup if item.get('question')}
            seen_questions_lower.update({item['question'].strip().lower() for item in dynamic_followup_history if item.get('question')})
            seen_questions_lower.update(example_questions_to_avoid)  # Add examples to avoid

            for line in potential_lines:
                # Remove unwanted prefixes and formatting
                q_text = re.sub(r'^\s*(\d+\.|\*|-|[A-Z]\.|\bAsk:|\bInquire:|\bClarify:)\s*', '', line).strip()
                q_text = re.sub(r'^\s*\*\*[^*]+\*\*[:\s]*', '', q_text).strip()  # Remove bolded labels
                q_text = re.sub(r'^["“](.*?)["”]?$', r'\1', q_text).strip()

                # Validate as a unique question
                if (q_text.endswith('?') and 
                    len(q_text) > 10 and 
                    q_text.lower() not in seen_questions_lower and 
                    q_text.lower() not in {q.lower() for q in extracted_questions}):
                    extracted_questions.append(q_text)
                    seen_questions_lower.add(q_text.lower())

            # Validate we got 5 questions
            if len(extracted_questions) >= 5:
                questions = extracted_questions[:5]
                logger.info(f"Successfully generated 5 unique follow-up questions: {questions}")
                break
            else:
                logger.warning(f"Attempt {attempt + 1}: Found only {len(extracted_questions)} valid/unique questions. Output: '{generated_output}'. Retrying...")

        # Final Check
        if len(questions) != 5:
            logger.error(f"Failed to generate exactly 5 unique follow-up questions after {MAX_ATTEMPTS} attempts. Last output: '{generated_output}'")
            # Fallback: Generate basic questions based on symptom_summary
            fallback_questions = [
                f"When did your {key_symptom} first start?",
                f"Does anything seem to make your {key_symptom} worse?",
                f"Have you noticed any other changes with your {key_symptom}?",
                f"Can you tell me more about how your {key_symptom} feels?",
                f"Has your {key_symptom} affected your daily routine?"
            ]
            questions = [q for q in fallback_questions if q.lower() not in seen_questions_lower][:5]
            logger.info(f"Using fallback questions: {questions}")

        return questions

    def generate_guidelines(self, reviewed_transcript: str, symptom_summary: dict, static_followup: list, dynamic_followup_history: list) -> dict:
        if not self.is_model_loaded or self.pipeline is None:
            logger.error("Cannot generate guidelines: LLM not loaded.")
            return {"guidelines": "Error: Model not available.", "summary": "Error."}

        key_symptom = symptom_summary.get('key_symptom', 'the symptom')
        if key_symptom in ["unknown symptom", "not specified", "", None]:
             key_symptom = "your main symptom"

        # Compile comprehensive context from the entire conversation
        static_followup_text = "\n".join(
            f"- Q: {item.get('question', '').strip()}\n  A: {item.get('answer', 'N/A').strip()}"
            for item in static_followup if isinstance(item, dict) and item.get('question')
        )
        dynamic_followup_text = "\n".join(
            f"- Q: {item.get('question', '').strip()}\n  A: {item.get('answer', 'N/A').strip()}"
            for item in dynamic_followup_history if isinstance(item, dict) and item.get('question')
        )
        initial_summary_text = (
             f"- Initial Complaint: {symptom_summary.get('key_symptom', 'Not specified')}\n"
             f"- Initial Severity: {symptom_summary.get('severity', 'Not specified')}\n"
             f"- Initial Onset/Duration: {symptom_summary.get('onset_duration', 'Not specified')}\n"
             f"- Initial Location: {symptom_summary.get('location', 'Not specified')}\n"
             f"- Initial Character: {symptom_summary.get('character', 'Not specified')}\n"
             f"- Initial Associated Symptoms: {symptom_summary.get('associated_symptoms', 'Not specified')}"
         )
        detailed_context = (
             "## Full Conversation Summary:\n\n"
             f"### Initial Statement & Summary:\nPatient Said: \"{reviewed_transcript}\"\nSummary:\n{initial_summary_text}\n\n"
             f"### Standard Questions & Answers:\n{static_followup_text if static_followup_text else 'None'}\n\n"
             f"### Follow-up Questions & Answers:\n{dynamic_followup_text if dynamic_followup_text else 'None'}\n"
        )

        # Prompt for generating personalized home care guidelines
        prompt = (
            "You are a caring physician assistant summarizing home care advice for a patient after a virtual consultation.\n"
            "Based **specifically and only on the complete conversation summary provided below**, create a **personalized home care plan** paragraph focused on managing the patient's primary issue: "
            f"'{key_symptom}'. Address the patient directly using simple, supportive language (like 'you should try...' or 'it might help to...').\n\n"
            "**Instructions for the Plan:**\n"
            "- Output a **single, flowing paragraph** (roughly 100-175 words).\n"
            "- **Crucially: DO NOT** use lists, bullet points, bold text, italics, or numbered steps. Plain paragraph text only.\n"
            "- **Personalize:** Tailor advice based on *all* relevant details gathered (severity, duration, location, character, associated symptoms, triggers/relief mentioned, impact discussed).\n"
            "- **Content:** Include relevant general care (like rest/hydration), specific non-drug tips related to the symptom, mention appropriate over-the-counter options (if context strongly supports it and safe), suggest activity modifications if applicable, and **always include clear 'red flag' symptoms** indicating when to seek immediate professional medical attention (e.g., 'If you notice X, Y, or Z, please see a doctor right away').\n\n"
            f"### Full Conversation Summary:\n{detailed_context}\n\n"
            "### Your Personalized Home Care Plan (Single Paragraph, Plain Text):"
        )

        try:
            # Generate the guidelines text
            result = self.generate_text(
                prompt,
                max_new_tokens=350, # Allow more tokens for detailed advice + red flags
                num_beams=3,        # Use more beams for potentially better structure/flow
                temperature=0.65,   # Balanced temperature for informative but not overly rigid text
                repetition_penalty=1.15, # Slightly increase penalty for variety
                early_stopping=True
            )
        except Exception as e:
            logger.error(f"Error during LLM guideline generation: {e}", exc_info=True)
            result = "Error: Guideline generation failed."

        # Post-process the generated guidelines
        if result.startswith("Error:") or not result.strip():
            logger.warning(f"LLM guideline generation failed or returned empty. Using fallback text.")
            # Basic fallback guideline
            final_guidelines_text = f"Based on our chat about your {key_symptom}, please make sure to rest and drink plenty of fluids like water. If your symptoms get worse, don't improve after a reasonable time, or if you develop new concerning symptoms like [mention 1-2 common red flags like high fever, severe pain, difficulty breathing depending on context], it's important to see a healthcare professional promptly."
        else:
            final_guidelines_text = result.strip()
            # Remove potential list markers/bolding missed by the prompt
            final_guidelines_text = re.sub(r"^\s*(\d+\.|\*|-)\s*", "", final_guidelines_text, flags=re.MULTILINE)
            final_guidelines_text = re.sub(r"[\*_]", "", final_guidelines_text) # Remove markdown bold/italic
            # Remove introductory phrases if present
            final_guidelines_text = re.sub(r"^(Here is|Here's) your personalized home care plan:?\s*", "", final_guidelines_text, flags=re.IGNORECASE).strip()
            # Consolidate whitespace
            final_guidelines_text = re.sub(r"\s+", " ", final_guidelines_text).strip()

        # Add a standard disclaimer
        disclaimer = "\n\n*Disclaimer: This information is based on our automated conversation and is for general informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.*"

        # Generate a concise summary of the guidelines
        summary_text = self.generate_summary(final_guidelines_text)

        return {"guidelines": final_guidelines_text + disclaimer, "summary": summary_text}

    def generate_summary(self, guidelines_text: str) -> str:
        """
        Generates a very concise (2-3 sentence) plain text summary of the provided home care guidelines,
        focusing on key self-care actions and reasons to seek help, without labels or formatting.
        """
        if not self.is_model_loaded or self.pipeline is None:
            logger.error("Cannot generate summary: LLM not loaded.")
            return "Error: Summary generation model not available."

        guidelines_to_summarize = guidelines_text.strip()
        if not guidelines_to_summarize or guidelines_to_summarize.startswith("Error:") or len(guidelines_to_summarize) < 50:
            logger.warning(f"Input guidelines text is invalid or too short: '{guidelines_to_summarize[:100]}...'")
            return "Could not generate a summary due to issues with the guidelines."

        logger.info("Attempting to generate concise summary...")

        # Refined prompt for plain text output
        prompt = (
            "You are an AI assistant skilled at summarizing medical advice concisely.\n"
            "Read the following home care advice paragraph provided to a patient.\n"
            "Create a very short summary (2-3 simple sentences) that captures the most important self-care actions "
            "and the main reasons to seek further medical help mentioned in the text.\n"
            "Address the patient directly using 'you' in plain, flowing sentences.\n"
            "Do NOT use labels (e.g., 'Key Self-Care Actions:', 'Reasons to Seek Help:'), bold text (like **), "
            "bullet points, or lists. Output only the summary as plain text.\n\n"
            "**Source Advice Paragraph:**\n"
            f"\"\"\"\n{guidelines_to_summarize}\n\"\"\"\n\n"
            "Summary (plain text, 2-3 sentences):"
        )

        try:
            raw_summary = self.generate_text(
                prompt,
                max_new_tokens=120,
                num_beams=3,         # Increased beams for better coherence
                temperature=0.6,     # Slightly higher for natural flow, but still focused
                repetition_penalty=1.2,
                early_stopping=True
            )
        except Exception as e:
            logger.error(f"LLM generation failed during summary creation: {e}", exc_info=True)
            raw_summary = "Error: Summary generation process failed."

        # Enhanced post-processing to remove unwanted formatting
        summary = raw_summary.replace("Summary (plain text, 2-3 sentences):", "").strip()
        summary = re.sub(r"^\s*Here'?s your summary:?\s*", "", summary, flags=re.IGNORECASE).strip()
        # Remove bolded labels (e.g., **Key Self-Care Actions:**)
        summary = re.sub(r'\s*\*\*[^*]+\*\*[:\s]*', ' ', summary).strip()
        # Remove list markers (e.g., - Rest, 1. Hydration)
        summary = re.sub(r'^\s*[-•*]|\d+\.\s*', '', summary, flags=re.MULTILINE).strip()
        # Consolidate whitespace and remove lingering colons or dashes
        summary = re.sub(r'\s+[-:]+\s*', ' ', summary).strip()
        summary = re.sub(r'\s+', ' ', summary).strip()

        # Final validation
        if not summary or summary.startswith("Error:") or len(summary.split()) < 5:
            logger.warning(f"Generated summary is invalid or too short: '{summary}'. Using fallback.")
            return "Please follow the detailed home care advice provided. Contact a doctor if your symptoms worsen or you notice concerning changes."
        
        # Ensure it ends with a period for readability
        if not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        logger.info(f"Successfully generated summary: {summary}")
        return summary.strip()

# --- Text-to-Speech Function ---
def text_to_speech_base64(text):
    """Converts text to speech and returns base64 encoded MP3 audio data."""
    if not text or text.startswith("Error:"):
        logger.warning(f"Skipping TTS for invalid or error text: {text[:50]}...")
        return None
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_bytes = fp.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        fp.close()
        # Return data URI format
        return f"data:audio/mp3;base64,{audio_base64}"
    except Exception as e:
        logger.error(f"Error during Text-to-Speech conversion: {e}", exc_info=True)
        return None # Return None on failure

# --- Global Handlers ---
llm_handler = LLMHandler()
whisper_model = None # Initialize Whisper model as None


# --- Flask Routes ---

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Handles audio file upload, transcribes it using Whisper."""
    global whisper_model
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    audio_file = request.files["file"]
    if audio_file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    # Load Whisper model on demand if not already loaded
    if whisper_model is None:
        try:
            logger.info("Loading Whisper model (small.en)...")
            # Load the model specifying the device
            whisper_model = whisper.load_model("small.en", device=DEVICE)
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            return jsonify({"error": "Failed to load transcription model."}), 500

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio: # Ensure suffix helps Whisper
        try:
            audio_file.save(temp_audio.name)
            logger.info(f"Audio file saved temporarily to {temp_audio.name}")
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}", exc_info=True)
            return jsonify({"error": f"Failed to save audio file: {e}"}), 500

        # Perform transcription
        try:
            logger.info("Starting transcription...")
            # Use fp16=True only if on CUDA for potential speedup
            result = whisper_model.transcribe(temp_audio.name, fp16=(DEVICE == "cuda"))
            logger.info("Transcription finished.")
        except Exception as e:
            logger.error("Transcription failed", exc_info=True)
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
        # No need to manually delete temp_audio, 'with' handles it

    # Process and return the transcript
    transcript = result.get("text", "").strip()
    # Optional: basic cleaning like consolidating whitespace
    transcript = re.sub(r'\s+', ' ', transcript)

    if not transcript:
        logger.warning("Transcription resulted in empty text.")
        # Decide if empty transcript is an error or just empty result
        return jsonify({"transcript": ""}) # Return empty transcript instead of error?
        # return jsonify({"error": "Transcription resulted in empty text."}), 500

    logger.info(f"Transcription successful: {transcript[:100]}...") # Log first 100 chars
    return jsonify({"transcript": transcript})


@app.route("/extract_symptom_summary", methods=["POST"])
def extract_symptom_summary_endpoint():
    """Extracts structured symptom summary from a given transcript using the LLM."""
    data = request.get_json()
    if not data or "transcript" not in data:
        return jsonify({"error": "Request must contain 'transcript'."}), 400

    transcript = data["transcript"].strip()
    if not transcript:
        return jsonify({"error": "Transcript cannot be empty."}), 400

    try:
        # Ensure LLM is loaded before proceeding
        if not llm_handler.is_model_loaded:
            logger.warning("LLM model not loaded for summary extraction, attempting load...")
            if not llm_handler.load_model():
                return jsonify({"error": "Failed to load analysis model."}), 500

        # Call the LLM handler method
        summary = llm_handler.extract_structured_symptoms(transcript)

        # Check if extraction produced an error message
        if isinstance(summary, str) and summary.startswith("Error:"):
             return jsonify({"error": f"Symptom extraction failed: {summary}"}), 500

        return jsonify({"symptom_summary": summary})
    except Exception as e:
        logger.error("Error during symptom summary extraction endpoint", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route("/start_interview", methods=["POST", "OPTIONS"])
def start_interview():
    """Initiates the interview process based on the initial transcript."""
    # Handle CORS preflight requests
    if request.method == "OPTIONS":
        response = app.make_response('')
        # Headers are now handled by Flask-CORS globally, but can be set explicitly if needed
        # response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        # response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        # response.headers.add('Access-Control-Allow-Headers', 'Content-Type, ngrok-skip-browser-warning')
        return response

    data = request.get_json()
    if not data or "transcript" not in data:
        return jsonify({"status": "error", "error": "Request must contain 'transcript'."}), 400

    transcript = data["transcript"].strip()
    if not transcript:
        return jsonify({"status": "error", "error": "Transcript cannot be empty."}), 400

    try:
        # Ensure LLM is loaded
        if not llm_handler.is_model_loaded:
             if not llm_handler.load_model():
                 return jsonify({"status": "error", "error": "Failed to load analysis model."}), 500

        # 1. Extract initial symptom summary
        symptom_summary = llm_handler.extract_structured_symptoms(transcript)
        if isinstance(symptom_summary, str) and symptom_summary.startswith("Error:"):
             return jsonify({"status": "error", "error": f"Failed initial analysis: {symptom_summary}"}), 500
        if symptom_summary["key_symptom"] == "unknown symptom":
            logger.warning("Could not identify a key symptom from the initial transcript.")
            # Handle inability to determine key symptom - perhaps ask a clarifying question?
            # For now, proceed with generic questions but flag the issue.

        key_symptom = symptom_summary.get("key_symptom", "unknown").lower()

        # 2. Define initial set of static questions based on key symptom (or default)
        static_questions_map = {
            # Example symptom-specific questions
            "fever": [
                {"question": "How high has your temperature been, and have you noticed if it's worse at certain times of day?", "answer": None},
                {"question": "Are you having any chills, body aches, or sweating along with the fever?", "answer": None},
                {"question": "Have you taken any medication like Tylenol or Advil for the fever, and did it help?", "answer": None}
            ],
            "headache": [
                {"question": "Can you describe the location of your headache? Is it all over, on one side, or in a specific spot like behind your eyes?", "answer": None},
                {"question": "How would you describe the pain? Is it sharp, dull, throbbing, or more like pressure?", "answer": None},
                {"question": "Does anything seem to make the headache better or worse, like lying down, light, noise, or certain foods?", "answer": None}
            ],
             "cough": [
                {"question": "Could you describe your cough? Is it dry, or are you coughing up phlegm? If so, what color is it?", "answer": None},
                {"question": "How long have you had this cough, and is it worse at certain times, like at night?", "answer": None},
                {"question": "Are you experiencing any other symptoms with the cough, such as shortness of breath, chest pain, or a sore throat?", "answer": None}
            ]
            # Add more symptom keys and relevant question sets
        }
        # Fallback to generic questions if key symptom not in map or unknown
        default_static_questions = [
            {"question": f"When did this {key_symptom if key_symptom != 'unknown' else 'issue'} start, and has it changed since then?", "answer": None},
            {"question": "Can you describe the feeling or sensation in more detail?", "answer": None},
            {"question": "Have you noticed anything that seems to trigger it or make it better?", "answer": None}
        ]
        static_followup = static_questions_map.get(key_symptom, default_static_questions)

        # 3. Initialize conversation state
        conversation_state = {
            "transcript": transcript,                # Initial user statement
            "symptom_summary": symptom_summary,      # Initial analysis
            "static_followup": static_followup,      # List of initial questions
            "dynamic_followup_history": [],          # History of LLM-generated Q&A
            "current_question": static_followup[0]["question"] if static_followup else None, # First question to ask
            "current_question_type": "static" if static_followup else None, # Track if current Q is static/dynamic/choice
            "dynamic_questions_asked_in_set": 0,     # Counter for dynamic questions in the current set of 3
            "total_dynamic_questions_asked": 0,      # Counter for total dynamic questions
            "awaiting_choice": False                 # Flag indicating if waiting for user choice (plan vs more Qs)
        }

        # 4. Generate audio for the first question
        audio_data = text_to_speech_base64(conversation_state["current_question"]) if conversation_state["current_question"] else None

        # 5. Return initial state to the client
        return jsonify({
            "status": "in_progress",                # Interview starts
            "conversation_state": conversation_state, # Send the initial state
            "audio_response": audio_data           # Send the audio for the first question
        })
    except Exception as e:
        logger.error(f"Error in start_interview: {e}", exc_info=True)
        # Return a generic error state
        return jsonify({"status": "error", "error": f"An unexpected error occurred: {str(e)}", "conversation_state": None, "audio_response": None}), 500

@app.route("/interview_step", methods=["POST"])
def interview_step():
    """Handles a single step in the interview: processes user answer, determines next question or action."""
    data = request.get_json()
    if not data or "conversation_state" not in data or "answer" not in data:
        return jsonify({"status": "error", "error": "Request must contain 'conversation_state' and 'answer'."}), 400

    current_state = data["conversation_state"]
    answer = data["answer"].strip() # Keep original case for potential LLM nuance, lower if needed for specific checks
    logger.info(f"Processing answer: '{answer[:50]}...' | Current Q: '{current_state.get('current_question', 'None')[:50]}...' | Awaiting Choice: {current_state.get('awaiting_choice')}")

    status = "in_progress" # Default status
    guidelines = None
    summary = None
    next_question = None
    audio_data = None

    try:
        # --- State Machine Logic ---

        # 1. Handle Choice Phase (Plan vs. More Questions) - MODIFIED TO GENERATE 5
        if current_state.get("awaiting_choice", False):
            logger.info(f"In 'awaiting_choice' phase with answer: '{answer}'")
            answer_lower = answer.lower()
            # Check if user wants the home care plan
            if any(keyword in answer_lower for keyword in ["plan", "home care", "guidelines", "advice", "summary", "finish", "done", "yes", "now"]):
                logger.info("Patient chose home care plan. Generating guidelines.")
                result = llm_handler.generate_guidelines(
                    current_state["transcript"], current_state["symptom_summary"],
                    current_state["static_followup"], current_state["dynamic_followup_history"]
                )
                current_state["current_question"] = None # No more questions
                current_state["current_question_type"] = None
                current_state["awaiting_choice"] = False
                status = "completed" # Interview finished
                guidelines = result.get("guidelines", "Error generating guidelines.")
                summary = result.get("summary", "Error generating summary.")
                # Provide audio for the summary (or a confirmation message)
                audio_response_text = summary if summary and not summary.startswith("Error:") else "Okay, I have prepared the home care guidelines for you."
                audio_data = text_to_speech_base64(audio_response_text)

            # Check if user wants more questions
            elif any(keyword in answer_lower for keyword in ["more", "questions", "follow-up", "continue", "ask"]):
                logger.info("Patient chose more questions. Generating next set of 5.")
                current_state["awaiting_choice"] = False # Exit choice mode
                # --- Generate the *next* set of 5 dynamic questions ---
                dynamic_questions = llm_handler.generate_followup_questions(
                    current_state["transcript"], current_state["symptom_summary"],
                    current_state["static_followup"], current_state["dynamic_followup_history"] # Pass full history
                )
                # --- Check if 5 questions were generated ---
                if not dynamic_questions: # Handles empty list return from LLM failure
                    logger.error("Failed to generate 5 additional dynamic questions. Proceeding to guidelines.")
                    result = llm_handler.generate_guidelines(
                        current_state["transcript"], current_state["symptom_summary"],
                        current_state["static_followup"], current_state["dynamic_followup_history"]
                    )
                    current_state["current_question"] = None
                    status = "completed"
                    guidelines = result.get("guidelines", "Error generating guidelines.")
                    summary = result.get("summary", "Error generating summary.")
                    audio_response_text = summary if summary and not summary.startswith("Error:") else "There was an issue generating more questions. Here are the guidelines based on what we discussed."
                    audio_data = text_to_speech_base64(audio_response_text)
                else:
                    # Add new questions to history, reset counter for the new set
                    current_state["dynamic_followup_history"].extend([{"question": q, "answer": None} for q in dynamic_questions])
                    # Find the first *newly added* question to ask
                    next_question = dynamic_questions[0]
                    current_state["current_question"] = next_question
                    current_state["current_question_type"] = "dynamic"
                    current_state["dynamic_questions_asked_in_set"] = 0 # Reset for the new set of 5
                    status = "in_progress"
                    audio_data = text_to_speech_base64(next_question)

            # Handle unclear choice
            else:
                logger.info("Unclear choice received. Asking again.")
                next_question = "Sorry, I didn't quite catch that. Would you like your personalized home care plan now, or should we continue with a few more questions?"
                current_state["current_question"] = next_question
                current_state["current_question_type"] = "choice" # Mark as choice question type
                current_state["awaiting_choice"] = True # Remain in choice mode
                status = "in_progress"
                audio_data = text_to_speech_base64(next_question)

        # 2. Handle Answering a Regular Question (Static or Dynamic) - MODIFIED CHECKS FOR 5
        elif current_state.get("current_question"):
            question_type = current_state.get("current_question_type")
            current_q_text = current_state["current_question"]

            # Find the question in the appropriate list and store the answer
            found_and_updated = False
            if question_type == "static":
                for item in current_state["static_followup"]:
                    if item["question"] == current_q_text and item["answer"] is None:
                        item["answer"] = answer
                        found_and_updated = True
                        logger.info(f"Stored answer for static question: {current_q_text[:50]}...")
                        break
            elif question_type == "dynamic":
                 # Iterate in reverse to find the most recent unanswered instance if question text repeats
                for item in reversed(current_state["dynamic_followup_history"]):
                     if item["question"] == current_q_text and item["answer"] is None:
                        item["answer"] = answer
                        current_state["dynamic_questions_asked_in_set"] += 1
                        current_state["total_dynamic_questions_asked"] +=1
                        found_and_updated = True
                        logger.info(f"Stored answer for dynamic question #{current_state['total_dynamic_questions_asked']} (Set count: {current_state['dynamic_questions_asked_in_set']}): {current_q_text[:50]}...")
                        break

            if not found_and_updated:
                 logger.warning(f"Could not find matching unanswered question for '{current_q_text[:50]}...' (Type: {question_type}). Storing answer might fail.")
                 # Decide how to handle this - maybe log and proceed, or return error? For now, log and proceed.


            # Determine the *next* question or action
            next_question = None

            # a. Check if there are remaining static questions
            for item in current_state["static_followup"]:
                if item["answer"] is None:
                    next_question = item["question"]
                    current_state["current_question_type"] = "static"
                    break

            # b. If static questions are done, move to dynamic questions
            if not next_question:
                # Check if we need to generate the *first* set of 5 dynamic questions
                if not current_state["dynamic_followup_history"]:
                    logger.info("Static questions complete. Generating first set of 5 dynamic questions.")
                    dynamic_questions = llm_handler.generate_followup_questions(
                        current_state["transcript"], current_state["symptom_summary"],
                        current_state["static_followup"], current_state["dynamic_followup_history"]
                    )
                    # --- Check if 5 questions were generated ---
                    if not dynamic_questions: # Handles empty list return
                        logger.error("Failed to generate initial 5 dynamic questions. Proceeding to guidelines.")
                        result = llm_handler.generate_guidelines(current_state["transcript"], current_state["symptom_summary"], current_state["static_followup"], current_state["dynamic_followup_history"])
                        status = "completed"; guidelines = result.get("guidelines"); summary = result.get("summary")
                        audio_response_text = summary if summary and not summary.startswith("Error:") else "There was an issue generating questions. Here are the guidelines based on our chat."
                        audio_data = text_to_speech_base64(audio_response_text)
                    else:
                        current_state["dynamic_followup_history"] = [{"question": q, "answer": None} for q in dynamic_questions]
                        next_question = dynamic_questions[0]
                        current_state["current_question_type"] = "dynamic"
                        current_state["dynamic_questions_asked_in_set"] = 0 # Start counting set of 5
                else:
                    # --- Check if we completed a set of 5 dynamic questions ---
                    # Use '>=' for safety in case counting gets off slightly
                    if current_state["dynamic_questions_asked_in_set"] >= 5:
                        logger.info("Completed a set of 5 dynamic questions. Offering choice.")
                        next_question = "We've covered quite a bit more detail. Would you like me to summarize with a personalized home care plan now, or shall we explore further with another set of questions?"
                        current_state["current_question_type"] = "choice"
                        current_state["awaiting_choice"] = True
                    else:
                        # Find the next unanswered dynamic question *within the current history*
                        found_next_dynamic = False
                        for item in current_state["dynamic_followup_history"]:
                             if item["answer"] is None:
                                next_question = item["question"]
                                current_state["current_question_type"] = "dynamic"
                                found_next_dynamic = True
                                break
                        if not found_next_dynamic:
                             # This means all dynamic questions asked so far are answered, but we didn't hit the set limit (e.g., if user chose 'plan' early after fewer than 5 Qs in a prior set)
                             logger.info("All existing dynamic questions seem answered. Proceeding to guidelines.")
                             result = llm_handler.generate_guidelines(current_state["transcript"], current_state["symptom_summary"], current_state["static_followup"], current_state["dynamic_followup_history"])
                             status = "completed"; guidelines = result.get("guidelines"); summary = result.get("summary")
                             audio_response_text = summary if summary and not summary.startswith("Error:") else "Okay, I have prepared the home care guidelines."
                             audio_data = text_to_speech_base64(audio_response_text)


            # Update current question and generate audio if still in progress
            if status == "in_progress" and next_question:
                current_state["current_question"] = next_question
                # Only generate audio if we are not already completed or errored
                if not audio_data: audio_data = text_to_speech_base64(next_question)
            elif status != "completed": # If no next question found and not completed/errored, generate guidelines
                 logger.info("No more questions to ask or state indicates completion. Generating guidelines.")
                 result = llm_handler.generate_guidelines(current_state["transcript"], current_state["symptom_summary"], current_state["static_followup"], current_state["dynamic_followup_history"])
                 status = "completed"; guidelines = result.get("guidelines"); summary = result.get("summary")
                 audio_response_text = summary if summary and not summary.startswith("Error:") else "Okay, based on our conversation, here are the home care guidelines."
                 audio_data = text_to_speech_base64(audio_response_text)
                 current_state["current_question"] = None # Clear current question as interview is over
                 current_state["current_question_type"] = None

        else:
            # Should not happen if logic is correct, but handle as error or completion
            logger.error("Interview step called but no current question is set and not awaiting choice.")
            status = "error"
            error_message = "Internal state error: No current question or choice pending."
            # Optionally try to recover or just end by generating guidelines
            result = llm_handler.generate_guidelines(current_state["transcript"], current_state["symptom_summary"], current_state["static_followup"], current_state["dynamic_followup_history"])
            status = "completed"; guidelines = result.get("guidelines", "Error generating final advice."); summary = result.get("summary", "Could not summarize.")
            audio_response_text = summary if summary and not summary.startswith("Error:") else "There seems to be an issue with the flow, but I've prepared the guidelines based on what we have."
            audio_data = text_to_speech_base64(audio_response_text)


        # --- Prepare Response ---
        response = {
            "status": status,
            "conversation_state": current_state,
            "audio_response": audio_data # Contains audio for next question or summary/confirmation
        }
        if status == "completed":
            response.update({"guidelines": guidelines, "summary": summary})
            logger.info(f"Interview completed. Responding with status '{status}'.")
        elif status == "error":
             response["error"] = error_message if 'error_message' in locals() else "An internal error occurred."
             logger.error(f"Interview step error. Responding with status '{status}'. Error: {response.get('error')}")
        else:
             logger.info(f"Interview continuing. Next question: '{current_state.get('current_question', 'None')[:50]}...'. Responding with status '{status}'.")


        return jsonify(response)

    except Exception as e:
        logger.error(f"Critical error in interview_step: {e}", exc_info=True)
        # Try to return current state with error status
        return jsonify({
            "status": "error",
            "error": f"A critical error occurred processing your response: {str(e)}",
            "conversation_state": current_state, # Return state for potential debugging on client
            "audio_response": None
        }), 500

# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Starting Server ---")
    print(f"Using device: {DEVICE}")

    # Preload models (optional but recommended for responsiveness)
    print("Attempting to preload LLM model...")
    if not llm_handler.load_model():
        # Log warning but continue; model will load on first request if needed
        print("Warning: LLM model failed to preload. Will attempt lazy loading on first relevant request.")
    else:
        print("LLM model preloaded successfully.")

    print("Attempting to preload Whisper model (small.en)...")
    try:
        # Load model globally here, so it's ready for the /transcribe endpoint
        whisper_model = whisper.load_model("small.en", device=DEVICE)
        print("Whisper model preloaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to preload Whisper model: {e}. Will attempt lazy loading on first /transcribe request.")
        whisper_model = None # Ensure it's None if preloading fails

    # Start ngrok tunnel
    public_url = None
    try:
        # Ensure bind_tls=True for HTTPS, often required by browsers for microphone access
        public_url = ngrok.connect("5000", bind_tls=True)
        print(f" * Ngrok Tunnel active at: {public_url}")
        print(f" * Allowing requests from: http://localhost:5173")
    except Exception as e:
        print(f"Warning: Failed to start Ngrok tunnel: {e}")
        print("Proceeding without Ngrok. Server will be accessible locally on port 5000.")

    # Start Flask development server
    # Use host='0.0.0.0' to make it accessible on the network (and via ngrok)
    # debug=False is recommended for production or when using ngrok reliably
    print("Starting Flask server on http://0.0.0.0:5000")
    try:
        # Use threaded=True if needed, but be mindful of GIL with CPU-bound tasks
        app.run(host="0.0.0.0", port=5000, debug=False)
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully...")
    finally:
        # Clean up resources
        print("Shutting down thread pool...")
        executor.shutdown(wait=True)
        if public_url:
            try:
                ngrok.disconnect(public_url)
                print("Ngrok tunnel disconnected.")
            except Exception as e:
                 print(f"Error disconnecting ngrok tunnel: {e}")
            try:
                ngrok.kill()
                print("Ngrok process terminated.")
            except Exception as e:
                 print(f"Error killing ngrok process: {e}")

        if DEVICE == "cuda":
            print("Attempting to clear CUDA cache...")
            # Explicitly delete models and clear cache if possible
            try:
                del llm_handler.model
                del llm_handler.tokenizer
                del llm_handler.pipeline
                if whisper_model: del whisper_model
            except Exception as del_e:
                print(f"Minor error during model deletion: {del_e}")
            gc.collect()
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")
        print("Server shutdown complete.")
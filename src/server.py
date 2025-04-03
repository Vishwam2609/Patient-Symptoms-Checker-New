# NO MONKEY PATCH BLOCK HERE

# Start with the normal imports
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
# Ensure transformers and torch are imported for the main code
import transformers # Keep this if needed below
import torch # Keep this
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from gtts import gTTS
# torch is already imported above

# ... (rest of your code from the previous correct answer) ...

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Disable flex attention and set a dedicated cache directory
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"
CACHE_DIR = "/tmp/transformers_cache"
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

NGROK_AUTH_TOKEN = "2sZL5k5FBMPppi3zC5xRRYuG5IP_6BiBZ5A9ee77WTxAfVWqa" # Consider moving this to an environment variable
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# LLMHandler class definition (Keep as is)
class LLMHandler:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.pipeline = None

    def clear_model_cache(self):
        if os.path.exists(CACHE_DIR):
            try:
                shutil.rmtree(CACHE_DIR)
                os.makedirs(CACHE_DIR, exist_ok=True)
                logger.info(f"Cleared Transformers cache at {CACHE_DIR}")
            except OSError as e:
                logger.error(f"Error clearing cache directory {CACHE_DIR}: {e}")
        else:
             os.makedirs(CACHE_DIR, exist_ok=True) # Ensure cache dir exists if not present


    def load_model(self):
        if self.pipeline is None:
            try:
                logger.info("Clearing model cache before loading LLM...")
                self.clear_model_cache()
                logger.info("Loading LLM model...")
                # Consider using environment variables more robustly or directly passing the token
                HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN", "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")
                LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025")

                # Use the token if provided
                use_auth_token_value = HUGGING_FACE_TOKEN if HUGGING_FACE_TOKEN else None

                self.tokenizer = AutoTokenizer.from_pretrained(
                    LLM_MODEL_NAME,
                    token=use_auth_token_value, # Use 'token' instead of 'use_auth_token' for newer versions
                    force_download=True,
                    use_fast=False, # Keep False if required by model/trust_remote_code
                    trust_remote_code=True,
                    cache_dir=CACHE_DIR
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL_NAME,
                    token=use_auth_token_value, # Use 'token' instead of 'use_auth_token'
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    force_download=True,
                    trust_remote_code=True,
                    cache_dir=CACHE_DIR
                    # Removed the problematic load_state_dict call source
                )
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if DEVICE == "cuda" else -1 # Use device mapping
                )
                logger.info("LLM model loaded successfully.")
            except Exception as e:
                logger.error("Error loading LLM model", exc_info=True)
                # Re-raise the exception so the calling function knows loading failed
                raise e # Re-raise

    # generate_text method (Keep as is)
    def generate_text(self, prompt, max_new_tokens, num_beams, temperature, repetition_penalty, early_stopping=True) -> str:
        if self.pipeline is None:
            try: # Add try-except around load_model here
                self.load_model()
            except Exception as e: # Catch specific loading errors
                 logger.error("LLM pipeline is not available because model loading failed.", exc_info=True)
                 # Return an empty string or raise a custom error to indicate failure upstream
                 return f"Error: Model could not be loaded. {e}" # Or raise specific exception

        # Check again after attempting to load
        if self.pipeline is None:
             logger.error("LLM pipeline is still not available after attempting to load.")
             return "Error: Model pipeline could not be initialized."

        try:
            logger.info("Generating text from LLM...")
            # Ensure prompt is a string or list of strings
            if not isinstance(prompt, (str, list)):
                 logger.error(f"Invalid prompt type: {type(prompt)}. Prompt must be str or list.")
                 return "Error: Invalid prompt format."

            pipeline_output = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=early_stopping,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                # Add pad_token_id if tokenizer doesn't have it set (common issue)
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
            ) or [] # Ensure pipeline_output is not None

            # Handle potential variations in pipeline output format
            if not pipeline_output:
                 logger.warning("LLM pipeline returned an empty result.")
                 return ""

            # Expecting a list of dictionaries
            if isinstance(pipeline_output, list) and pipeline_output and isinstance(pipeline_output[0], dict):
                 generated_text = pipeline_output[0].get('generated_text', "")
            # Handle cases where it might return just a list of strings (less common for text-generation)
            elif isinstance(pipeline_output, list) and pipeline_output and isinstance(pipeline_output[0], str):
                 generated_text = pipeline_output[0]
            else:
                 logger.warning(f"Unexpected LLM pipeline output format: {type(pipeline_output)}. Output: {pipeline_output}")
                 # Attempt to convert to string as a fallback
                 generated_text = str(pipeline_output[0]) if pipeline_output else ""

            # Clean the generated text from the prompt if needed
            if isinstance(prompt, str) and generated_text.startswith(prompt):
                 # Remove the prompt part from the beginning of the generated text
                 # This is often needed as the pipeline includes the input prompt
                 generated_text = generated_text[len(prompt):]

            return generated_text.strip() if generated_text else ""
        except Exception as e:
            logger.error("LLM generation error", exc_info=True)
            # Return error message or empty string
            return f"Error during text generation: {e}"

    # generate_followup_questions method (REVISED)
    def generate_followup_questions(self, reviewed_transcript: str, key_symptom: str, static_followup: list) -> list:
        """
        Dynamically generates exactly three distinct and clinically relevant follow-up questions
        based on the patient's transcript and key symptom. These questions MUST explore
        key aspects of the symptom NOT conceptually covered by the static questions.

        Focuses on standard clinical dimensions like:
        - Severity/Intensity (Scale, impact on function)
        - Character/Quality (Description: sharp, dull, constant, intermittent, etc.)
        - Onset/Timing/Frequency (When did it start? How often? Pattern?)
        - Location/Radiation (Where is it? Does it spread? - If applicable)
        - Triggers/Aggravating/Relieving Factors (What makes it better or worse?)
        - Impact on Daily Life (Effect on activities, work, sleep)
        - Associated Symptoms (Specific, relevant ones beyond generics if generics covered)
        - Treatments Tried/Relief Measures (What has the patient done?)

        Avoids repeating topics clearly addressed by static questions.
        Outputs exactly three unique questions, one per line, ending with a question mark.
        """
        # --- Topic Analysis Helper ---
        def get_question_topic(question_text):
            q_lower = question_text.lower()
            # Severity
            if any(kw in q_lower for kw in ["scale of 1 to 10", "how severe", "how bad", "intensity"]):
                return "Severity"
            # Character/Quality
            if any(kw in q_lower for kw in ["describe", "quality", "character", "feel like", "sharp", "dull", "throbbing", "burning", "constant", "intermittent"]):
                return "Character"
            # Onset/Timing/Frequency
            if any(kw in q_lower for kw in ["when did", "start", "began", "first time", "how long", "how often", "frequency", "pattern", "timing"]):
                return "Onset/Timing"
            # Triggers/Aggravating/Relieving
            if any(kw in q_lower for kw in ["trigger", "worsen", "aggravate", "better", "improve", "relieve", "help", "specific activities", "food", "position"]):
                return "Triggers/Relief"
            # Impact on Daily Life
            if any(kw in q_lower for kw in ["affect", "impact", "daily activities", "daily life", "work", "sleep", "function"]):
                return "Impact"
            # Location/Radiation (less common for all symptoms, but possible)
            if any(kw in q_lower for kw in ["where", "location", "point to", "spread", "radiate"]):
                return "Location"
            # Associated Symptoms (check for specifics beyond generic ones often covered initially)
            if any(kw in q_lower for kw in ["other symptoms", "anything else with", "coughing up", "accompanied by"]):
                 # Check if it's just a generic 'any other symptoms' which might overlap easily
                 if "any other symptoms" in q_lower or "anything else" in q_lower:
                     return "Assoc. Generic"
                 return "Assoc. Specific"
            # Treatments Tried
            if any(kw in q_lower for kw in ["tried", "taken anything", "treatment", "medication", "remedies"]):
                return "Treatments Tried"
            return "Other" # Default category

        # --- Identify Topics Covered by Static Questions ---
        static_questions_text = [item.get('question', '').strip() for item in static_followup if isinstance(item, dict) and item.get('question')]
        static_topics_covered = set()
        for q_text in static_questions_text:
            topic = get_question_topic(q_text)
            if topic != "Other":
                static_topics_covered.add(topic)
        logger.info(f"Static questions cover topics: {static_topics_covered}")

        # --- Prepare Context for LLM ---
        static_questions_context = "\n".join([f"- {q}" for q in static_questions_text]) if static_questions_text else "None provided."

        context_for_llm = (
            f"Patient's description: {reviewed_transcript}\n"
            f"Key symptom: {key_symptom}\n"
            "Static Follow-up Questions Already Asked (DO NOT repeat topics covered below):\n"
            f"{static_questions_context}\n"
            f"Topics already covered by static questions: {', '.join(static_topics_covered) if static_topics_covered else 'None'}\n"
        )

        # --- Define the Prompt for the LLM ---
        prompt = (
            "You are an experienced clinician performing a *focused secondary inquiry*. Your goal is to generate exactly three **clinically insightful** and **distinct** follow-up questions about the key symptom, based *only* on the context below. "
            "These questions MUST investigate aspects **NOT already covered** by the static questions listed or the topics identified as covered. "
            "Prioritize questions that explore potentially uncovered clinical dimensions like:\n"
            "- **Severity/Intensity:** (e.g., 'On a scale of 1-10, how severe is the [symptom] at its worst?' or 'How does this [symptom] interfere with your ability to [specific activity]?')\n"
            "- **Character/Quality:** (e.g., 'Can you describe the [symptom] in more detail? Is it sharp, dull, aching, burning, constant, or does it come and go?')\n"
            "- **Triggers/Aggravating/Relieving Factors:** (e.g., 'What specifically seems to make the [symptom] worse or better? Think about activities, time of day, foods, or positions.')\n"
            "- **Impact on Daily Life:** (e.g., 'How is the [symptom] specifically affecting your work, sleep, or ability to do daily tasks?')\n"
            "- **Relevant Associated Symptoms:** (e.g., If symptom is cough, 'Are you coughing anything up? If so, what color is it?' - Avoid generic 'any other symptoms?' if already asked).\n"
            "- **Treatments Tried:** (e.g., 'Have you tried any specific remedies or medications for this [symptom], and what was the effect?')\n\n"
            f"Do NOT ask about topics already covered ({', '.join(static_topics_covered) if static_topics_covered else 'None'}). "
            "Ensure each question is unique in the information it seeks, uses clear patient-friendly language, ends with a question mark, and probes for useful details. "
            "Output ONLY the three questions, one per line.\n\n"
            f"Context:\n{context_for_llm}\n"
            "### OUTPUT:"
        )

        # --- Generate Text using LLM ---
        result = self.generate_text(prompt, max_new_tokens=180, num_beams=5, temperature=0.65, repetition_penalty=1.2)

        # --- Process and Validate Generated Questions ---
        generated_questions = []
        if result and not result.startswith("Error:"):
            lines = [line.strip() for line in result.split("\n") if line.strip()]
            for q in lines:
                q = re.sub(r'^[A-Z]:\s*', '', q)
                q = re.sub(r'^\s*Dynamic Follow-Up\s*\(\s*\d+\s+of\s+\d+\s*\)\s*:?\s*', '', q, flags=re.IGNORECASE)
                q = re.sub(r'^[\d\.\-\)\s]+', '', q) # Remove leading list markers
                q = q.strip()
                if q:
                    if not q.endswith('?'):
                        q += '?'
                    # Basic check for relevance (contains symptom or generic pronoun)
                    if key_symptom.lower() in q.lower() or any(pronoun in q.lower() for pronoun in [' it', ' this', ' symptom']):
                        generated_questions.append(q)
                    else:
                        logger.warning(f"Filtering out potentially irrelevant generated question: '{q}' for symptom '{key_symptom}'")

        # --- Filter and Select Distinct Questions ---
        distinct_questions = []
        seen_topics = set(static_topics_covered) # Start with topics covered by static questions
        seen_question_texts_lower = {q.lower() for q in static_questions_text}

        for q in generated_questions:
            q_lower = q.lower()
            q_topic = get_question_topic(q)

            # Check 1: Is it a duplicate of a static question (textual)?
            is_duplicate_static = q_lower in seen_question_texts_lower

            # Check 2: Does it cover a topic already covered (by static or previous dynamic)?
            is_topic_covered = q_topic in seen_topics and q_topic != "Other" # Allow multiple 'Other' questions if needed

            if not is_duplicate_static and not is_topic_covered:
                distinct_questions.append(q)
                seen_topics.add(q_topic)
                seen_question_texts_lower.add(q_lower) # Add to check against future generated ones too
                if len(distinct_questions) == 3:
                    break # Stop once we have 3 distinct questions

        # --- Generate Fallback Questions if Needed ---
        if len(distinct_questions) < 3:
            logger.warning(f"LLM generated only {len(distinct_questions)} distinct, non-redundant questions covering new topics. Adding fallbacks.")

            # Define potential fallback topics and standard questions
            fallback_options = {
                "Severity": f"On a scale of 1 to 10, with 10 being the worst imaginable, how severe is your {key_symptom} right now?",
                "Character": f"Can you describe the {key_symptom}? For example, is it sharp, dull, aching, burning, constant, or intermittent?",
                "Triggers/Relief": f"Is there anything specific you've noticed that makes the {key_symptom} better or worse (like activity, rest, food, position)?",
                "Impact": f"How is this {key_symptom} impacting your daily activities, such as work, sleep, or hobbies?",
                "Treatments Tried": f"Have you tried taking or doing anything to relieve the {key_symptom}? If so, did it help?",
                 # Add Onset/Timing only if truly fundamental and missed
                # "Onset/Timing": f"Just to confirm, when exactly did this {key_symptom} start, and has it changed since then?"
            }

            # Prioritize topics not covered yet
            potential_fallback_topics = ["Severity", "Character", "Triggers/Relief", "Impact", "Treatments Tried"] # Order matters slightly

            for topic in potential_fallback_topics:
                if topic not in seen_topics:
                    fallback_q = fallback_options[topic]
                    fallback_q_lower = fallback_q.lower()
                    # Final check to prevent adding a fallback that's somehow textually identical to a static one
                    if fallback_q_lower not in seen_question_texts_lower:
                         distinct_questions.append(fallback_q)
                         seen_topics.add(topic) # Mark topic as covered by fallback
                         seen_question_texts_lower.add(fallback_q_lower)
                         if len(distinct_questions) == 3:
                              break

        # Fallback if STILL less than 3 (highly unlikely now, but safe)
        if len(distinct_questions) < 3:
             logger.error("Could not generate 3 distinct follow-up questions even with fallbacks. Returning generic defaults.")
             generic_defaults = [
                  f"On a scale of 1 to 10, how severe is the {key_symptom}?",
                  f"Can you describe the {key_symptom} in more detail (e.g., sharp, dull, constant)?",
                  f"Does anything seem to make the {key_symptom} better or worse?",
             ]
             # Add only enough to reach 3, avoiding duplicates if possible
             existing_lower = {q.lower() for q in distinct_questions}
             for qd in generic_defaults:
                 if len(distinct_questions) < 3 and qd.lower() not in existing_lower:
                     distinct_questions.append(qd)


        logger.info(f"Final distinct follow-up questions: {distinct_questions[:3]}")
        return distinct_questions[:3] # Return exactly the first 3

    # generate_guidelines method (Keep as is, but add error check for generate_text)
    def generate_guidelines(self, reviewed_transcript: str, key_symptom: str, static_followup: list, dynamic_followup: list) -> str:
        """
        Generates a concise, patient-friendly home care plan in one paragraph of approximately 150 words.
        The guidelines consider the patient's reviewed transcript, extracted key symptom, static follow-up Q&A,
        and dynamic follow-up Q&A. The plan focuses on specific home care strategies the patient can follow
        to get relief from the reported symptom.
        """
        # Ensure follow-up items are dicts with 'question' and 'answer' keys
        static_followup_text = "\n".join(
            [f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}" for item in static_followup if isinstance(item, dict)]
        )
        dynamic_followup_text = "\n".join(
            [f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}" for item in dynamic_followup if isinstance(item, dict)]
        )

        detailed_context = (
            "Conversation so far:\n"
            f"Patient's description: {reviewed_transcript}\n"
            f"Key symptom: {key_symptom}\n"
            "Static Follow-up Q&A:\n" + (static_followup_text or "None") + "\n"
            "Dynamic Follow-up Q&A:\n" + (dynamic_followup_text or "None") + "\n"
        )

        prompt = (
            "You are a compassionate doctor speaking directly to a patient with limited medical knowledge. "
            "Based solely on the conversation below, create a concise, personalized home care plan in one paragraph of approximately 150 words. "
            "Focus on practical, specific home care strategies the patient can follow to find relief from the key symptom, "
            "integrating insights from both the static and dynamic follow-up Q&A without repeating the questions or answers verbatim. "
            "Address the symptom’s severity, triggers, daily impact, or associated symptoms as relevant, ensuring all advice is tailored to the patient’s responses. "
            "Use clear, simple language, avoid medical jargon, and make the tone encouraging and supportive. "
            "Do not include section headings, bolded text, or numbered lists—just a single, flowing paragraph.\n\n"
            f"{detailed_context}\n\n"
            "Your Personalized Home Care Plan:" # Removed extra newline
        )

        result = self.generate_text(
            prompt,
            max_new_tokens=250, # Increased slightly for potentially longer advice
            num_beams=5,
            temperature=0.6, # Adjusted temp slightly
            repetition_penalty=1.1, # Adjusted penalty slightly
            early_stopping=True
        )

        # Check if generation failed
        if result.startswith("Error:"):
            logger.error(f"Failed to generate guidelines: {result}")
            return f"I apologize, but I encountered an issue generating personalized guidelines based on our conversation. Please consult with a healthcare professional for advice regarding your {key_symptom}."


        # --- Post-processing the guidelines ---
        # Remove potential leading labels or markers
        paragraph = " ".join(result.split()) # Consolidate whitespace

        # Remove common model artifacts or formatting intrusions
        paragraph = re.sub(r"^\d+\.\s+", "", paragraph) # Leading numbers
        paragraph = re.sub(r"(\s)\d+\.\s+", r"\1", paragraph) # Mid-text numbers
        paragraph = re.sub(r"\*\*.*?\*\*\s*[:\-]?\s*", "", paragraph) # Bolded labels like **Advice:**
        paragraph = re.sub(r"\[.*?\]\s*", "", paragraph) # Content in square brackets
        paragraph = paragraph.replace("Home Care Plan:", "").strip() # Remove explicit label if present

        # Target word count (approximate) - This logic is flawed, just return the cleaned paragraph
        # Trying to force a word count often makes the text unnatural. Let the model control the length primarily.
        # words = paragraph.split()
        # target_words = 150
        # if len(words) < target_words * 0.8: # If significantly shorter
        #     logger.warning(f"Generated guidelines shorter than expected ({len(words)} words).")
        # elif len(words) > target_words * 1.2: # If significantly longer
        #     logger.warning(f"Generated guidelines longer than expected ({len(words)} words). Truncating.")
        #     words = words[:int(target_words*1.1)] # Truncate with some buffer

        # final_paragraph = " ".join(words)

        final_paragraph = paragraph # Use the cleaned paragraph directly

        # Ensure proper sentence ending
        if final_paragraph and final_paragraph[-1].isalnum():
            final_paragraph += "."

        return final_paragraph

llm_handler = LLMHandler()

# --- Flask Routes (Keep as is, but add error handling for LLM calls) ---

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please upload an audio file with key 'file'."}), 400

    audio_file = request.files["file"]
    # Use a temporary directory to handle potential file naming issues
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "audio.wav") # Give it a specific name/extension
        try:
             audio_file.save(temp_path)
        except Exception as save_err:
             logger.error(f"Error saving uploaded file: {save_err}", exc_info=True)
             return jsonify({"error": f"Failed to save uploaded file: {save_err}"}), 500

        whisper_model = None # Initialize
        try:
             # Load model within the request if memory is a concern, or preload if frequently used
             whisper_model = whisper.load_model("small.en", device=DEVICE)
             logger.info("Whisper model loaded.")
             result = whisper_model.transcribe(temp_path, fp16=False) # fp16=False is safer for CPU
             logger.info(f"Raw transcription result: {result}")
        except Exception as e:
             logger.error("Transcription failed", exc_info=True)
             # Check for common errors like ffmpeg not found
             if "ffmpeg" in str(e).lower():
                  return jsonify({"error": "Transcription failed: ffmpeg not found or not configured correctly."}), 500
             return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
        finally:
             # Ensure model is unloaded to free memory
             if whisper_model is not None:
                 del whisper_model
                 if DEVICE == 'cuda':
                      torch.cuda.empty_cache() # Clear GPU cache if CUDA was used
                 gc.collect()
                 logger.info("Whisper model unloaded and garbage collected.")


    transcript_raw = result.get("text", "")
    # More robust cleaning
    transcript = transcript_raw if isinstance(transcript_raw, str) else " ".join(map(str, transcript_raw))
    transcript = re.sub(r'\s+', ' ', transcript).strip() # Consolidate whitespace
    # Keep basic punctuation useful for context
    transcript = re.sub(r'[^\w\s.,!?-]', '', transcript) # Allow basic punctuation

    logger.info(f"Processed transcript: '{transcript}'")
    if not transcript:
        # Provide more context if possible (e.g., empty result vs. error)
        if "text" not in result or not result["text"]:
            return jsonify({"error": "Transcription resulted in empty text. Please ensure the audio is clear and contains speech."}), 500
        else:
             return jsonify({"error": "Transcript processing failed. Check logs."}), 500

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
        # Define the prompt clearly
        prompt = (
             "You are a medical assistant analyzing a patient's statement. "
             "Based on the following patient description, identify and extract only the single most prominent key symptom or complaint. "
             "Express this symptom concisely in one or a few words (e.g., 'headache', 'stomach pain', 'difficulty breathing'). "
             "Do not add any explanation or introductory phrases.\n\n"
             f"Patient Description: \"{transcript}\"\n\n"
             "Key Symptom:" # Changed from "Answer:" for clarity
        )
        key_symptom_raw = llm_handler.generate_text(
            prompt,
            max_new_tokens=15, # Reduced max tokens
            num_beams=3,
            temperature=0.5, # Lower temp for focused extraction
            repetition_penalty=1.1
        )

        # Check if generation failed
        if key_symptom_raw.startswith("Error:"):
            logger.error(f"Key symptom extraction failed: {key_symptom_raw}")
            return jsonify({"error": f"Key symptom extraction failed due to LLM error: {key_symptom_raw}"}), 500

        # Post-process the result more carefully
        # Remove potential prefixes or instructions the model might have repeated
        key_symptom = key_symptom_raw.replace("Key Symptom:", "").strip()
        # Remove punctuation that might trail
        key_symptom = re.sub(r'[.,!?]$', '', key_symptom).strip()
        # Optional: Convert to lower case for consistency? Depends on downstream use.
        # key_symptom = key_symptom.lower()

        if not key_symptom:
            logger.warning(f"LLM returned empty result for key symptom extraction from transcript: {transcript}")
            # Fallback or specific error
            return jsonify({"error": "Could not extract a key symptom from the provided transcript."}), 500

    except Exception as e:
        # Catch potential errors during the generate_text call itself if not handled inside
        logger.error("Key symptom extraction error (outer)", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during key symptom extraction: {str(e)}"}), 500

    logger.info(f"Extracted key symptom: '{key_symptom}'")
    return jsonify({"key_symptom": key_symptom})


@app.route("/generate_followup_questions", methods=["POST"])
def generate_followup_questions_endpoint():
    data = request.get_json()
    # Validate input structure more carefully
    if not data or not isinstance(data, dict):
         return jsonify({"error": "Invalid request body. JSON object expected."}), 400

    reviewed_transcript = data.get("reviewed_transcript", "").strip()
    key_symptom = data.get("key_symptom", "").strip()
    static_followup = data.get("static_followup", []) # Assume it's a list

    # Check required fields
    if not reviewed_transcript or not key_symptom:
        return jsonify({"error": "Missing required fields. Please provide non-empty 'reviewed_transcript' and 'key_symptom'."}), 400

    # Validate static_followup format (optional but good practice)
    if not isinstance(static_followup, list):
        return jsonify({"error": "'static_followup' must be a list of objects (can be empty)."}), 400
    # Further validation can check if list items are dicts with 'question'/'answer'

    try:
        questions = llm_handler.generate_followup_questions(reviewed_transcript, key_symptom, static_followup)
        # The function now handles LLM errors internally and returns defaults/empty list
        if not questions:
            # This case might occur if even default questions were deemed redundant
            logger.warning("No follow-up questions generated, possibly due to redundancy or errors.")
            # Return empty list or a message
            return jsonify({"follow_up_questions": [], "message": "No suitable follow-up questions could be generated."})

    except Exception as e:
        # Catch unexpected errors in the endpoint logic itself
        logger.error("Error in /generate_followup_questions endpoint", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

    return jsonify({"follow_up_questions": questions})


@app.route("/generate_guidelines", methods=["POST"])
def generate_guidelines_endpoint(): # Renamed function for clarity
    data = request.get_json()
    # Validate input
    if not data or not isinstance(data, dict):
         return jsonify({"error": "Invalid request body. JSON object expected."}), 400

    transcript = data.get("transcript", "").strip()
    key_symptom = data.get("key_symptom", "").strip()
    static_followup = data.get("static_followup", []) # Renamed for clarity internally
    dynamic_followup = data.get("dynamic_followup", [])

    if not transcript or not key_symptom:
        return jsonify({"error": "Transcript and key_symptom cannot be empty."}), 400

    # Validate that follow-ups are lists (basic check)
    if not isinstance(static_followup, list) or not isinstance(dynamic_followup, list):
         return jsonify({"error": "Follow-up data must be provided as lists."}), 400


    # Check if answers are provided (crucial for guideline generation)
    # Allow empty answers but log a warning, as the LLM might struggle
    missing_static_answers = any(not item.get("answer", "").strip() for item in static_followup if isinstance(item, dict))
    missing_dynamic_answers = any(not item.get("answer", "").strip() for item in dynamic_followup if isinstance(item, dict))

    if missing_static_answers:
        logger.warning("Some static follow-up questions might be unanswered. Guidelines might be less specific.")
        # Decide if this should be a hard error or just a warning
        # return jsonify({"error": "All static follow-up questions must be answered for guideline generation."}), 400
    if missing_dynamic_answers:
        logger.warning("Some dynamic follow-up questions might be unanswered. Guidelines might be less specific.")
        # return jsonify({"error": "All dynamic follow-up questions must be answered for guideline generation."}), 400

    # Load model if needed (handled within generate_guidelines now)
    # llm_handler.load_model() # Removed, called inside generate_guidelines

    guidelines_text = "" # Initialize
    try:
        guidelines_text = llm_handler.generate_guidelines(transcript, key_symptom, static_followup, dynamic_followup)
        # Check if generation failed (function returns error string now)
        if guidelines_text.startswith("Error:") or guidelines_text.startswith("I apologize, but"):
             logger.error(f"Guideline generation failed: {guidelines_text}")
             # Return the error message from the LLM or a generic one
             return jsonify({"error": guidelines_text}), 500

        logger.info(f"Generated guidelines: '{guidelines_text}'")

    except Exception as e:
         # Catch unexpected errors in the endpoint logic
         logger.error("Error in /generate_guidelines endpoint during LLM call", exc_info=True)
         return jsonify({"error": f"An unexpected error occurred during guideline generation: {str(e)}"}), 500


    # --- TTS Generation ---
    audio_data = "" # Initialize
    if guidelines_text: # Only attempt TTS if guidelines were successfully generated
        try:
            tts = gTTS(text=guidelines_text, lang='en', slow=False) # Consider slow=False for natural speed
            audio_io = io.BytesIO()
            tts.write_to_fp(audio_io)
            audio_io.seek(0)
            audio_base64 = base64.b64encode(audio_io.read()).decode('utf-8')
            audio_data = f"data:audio/mp3;base64,{audio_base64}"
            logger.info("TTS audio generated successfully.")
        except Exception as e:
            logger.error("TTS conversion error", exc_info=True)
            # Don't fail the whole request, just return guidelines without audio
            # audio_data will remain ""

    return jsonify({"guidelines": guidelines_text, "audio_data": audio_data})


if __name__ == "__main__":
    # Set up Ngrok tunnel
    public_url = None
    try:
        # Ensure Ngrok is shutdown if script restarts uncleanly
        ngrok.kill()
        public_url = ngrok.connect("5000") # Use port 5000
        print(f"Ngrok Tunnel active at: {public_url}")
    except Exception as ngrok_error:
        print(f"Failed to start Ngrok tunnel: {ngrok_error}")
        # Decide if you want to exit or run locally only
        # exit() # Or just print warning and continue locally

    # Run Flask app
    # Use waitress or gunicorn for production instead of Flask's development server
    print("Starting Flask development server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False) # debug=False is important for production/stability

    # Optional: Disconnect ngrok on clean exit (might not run if CTRL+C is too abrupt)
    # finally:
    #     if public_url:
    #          print("Disconnecting Ngrok tunnel...")
    #          ngrok.disconnect(public_url)
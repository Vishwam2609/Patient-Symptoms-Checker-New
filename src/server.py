import os
import logging
import gc
import tempfile
import shutil
import re
import base64
import io
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import whisper
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from gtts import gTTS
import json

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

NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "2sZL5k5FBMPppi3zC5xRRYuG5IP_6BiBZ5A9ee77WTxAfVWqa")
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Request counter for periodic GPU cleanup
request_counter = 0

# Thread pool executor for offloading blocking tasks
executor = ThreadPoolExecutor(max_workers=2)

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
            os.makedirs(CACHE_DIR, exist_ok=True)

    def load_model(self):
        if self.pipeline is None:
            try:
                logger.info("Clearing model cache before loading LLM...")
                self.clear_model_cache()
                logger.info("Loading LLM model...")
                HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN", "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")
                LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025")
                use_auth_token_value = HUGGING_FACE_TOKEN if HUGGING_FACE_TOKEN else None

                self.tokenizer = AutoTokenizer.from_pretrained(
                    LLM_MODEL_NAME,
                    token=use_auth_token_value,
                    force_download=True,
                    use_fast=False,
                    trust_remote_code=True,
                    cache_dir=CACHE_DIR
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL_NAME,
                    token=use_auth_token_value,
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
        global request_counter
        if self.pipeline is None:
            try:
                self.load_model()
            except Exception as e:
                logger.error("LLM pipeline is not available because model loading failed.", exc_info=True)
                return f"Error: Model could not be loaded. {e}"

        try:
            logger.info("Generating text from LLM...")
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
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
            ) or []

            if not pipeline_output:
                logger.warning("LLM pipeline returned an empty result.")
                return ""

            generated_text = pipeline_output[0].get('generated_text', "")
            if isinstance(prompt, str) and generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]

            # Periodic GPU cleanup
            request_counter += 1
            if DEVICE == "cuda" and request_counter % 10 == 0:
                torch.cuda.empty_cache()
                logger.info("Periodic GPU memory cleanup performed.")

            return generated_text.strip() if generated_text else ""
        except Exception as e:
            logger.error("LLM generation error", exc_info=True)
            return f"Error during text generation: {e}"

    def extract_structured_symptoms(self, transcript: str) -> dict:
        """
        Uses the LLM to extract a structured summary of symptoms from a patient transcript.
        Extracts key symptom, severity, onset/duration, location, character, and associated symptoms in a generalized manner.
        """
        # Generalized prompt without specific symptom examples
        prompt = (
            "You are a medical assistant tasked with analyzing a patient's initial statement.\n"
            "Extract the following details from the description provided:\n"
            "- key_symptom: The primary health complaint or most prominent symptom the patient mentions (e.g., a single noun or short phrase like 'fever', 'pain', 'coughing'). Identify the main focus of their concern.\n"
            "- severity: How intense or bad the primary symptom is (e.g., 'severe', 'mild', 'getting worse'). Use 'not specified' if not mentioned.\n"
            "- onset_duration: When the symptom began or how long it has persisted (e.g., 'since yesterday', 'for 2 weeks'). Use 'not specified' if not mentioned.\n"
            "- location: The part of the body affected by the primary symptom (e.g., 'chest', 'whole body'). Use 'not specified' if not mentioned.\n"
            "- character: The quality or nature of the primary symptom (e.g., 'sharp', 'dull', 'throbbing'). Use 'not specified' if not mentioned.\n"
            "- associated_symptoms: Any additional symptoms mentioned alongside the primary one (e.g., 'nausea', 'fatigue'). Use 'none mentioned' if none are noted.\n\n"
            f"Patient Description: \"{transcript}\"\n\n"
            "Carefully analyze the description to determine the most prominent symptom as the key_symptom. "
            "Output ONLY a valid JSON object containing these fields, with values as strings. "
            "Ensure the output is concise and directly reflects the patient’s statement.\n\n"
            "JSON Output:"
        )

        # Optimized parameters for generalized, reliable JSON output
        raw_output = self.generate_text(
            prompt,
            max_new_tokens=300,  # Increased to handle complex inputs
            num_beams=1,         # Single beam for simplicity and speed
            temperature=0.3,     # Balanced for consistency without over-rigidity
            repetition_penalty=1.2  # Reduce repetition for clearer output
        )

        # Default summary structure
        summary = {
            "key_symptom": "unknown symptom",
            "severity": "not specified",
            "onset_duration": "not specified",
            "location": "not specified",
            "character": "not specified",
            "associated_symptoms": "none mentioned"
        }

        try:
            # Clean markdown or extra formatting
            cleaned_output = re.sub(r"```json\s*([\s\S]*?)\s*```", r"\1", raw_output).strip()
            parsed_json = json.loads(cleaned_output)

            if isinstance(parsed_json, dict):
                # Handle potential key variations
                key_mapping = {
                    "main_symptom": "key_symptom",
                    "primary_symptom": "key_symptom",
                    "chief_complaint": "key_symptom",
                    "main_issue": "key_symptom"
                }
                for alt_key, standard_key in key_mapping.items():
                    if alt_key in parsed_json and standard_key not in parsed_json:
                        parsed_json[standard_key] = parsed_json.pop(alt_key)

                # Update summary with parsed values
                for key in summary.keys():
                    if key in parsed_json and isinstance(parsed_json[key], str) and parsed_json[key].strip():
                        summary[key] = parsed_json[key].strip()

                # Validate key_symptom
                if not summary["key_symptom"] or summary["key_symptom"].lower() in ["not specified", "unknown", ""]:
                    raise ValueError("Key symptom not adequately identified by LLM")

            logger.info(f"Successfully extracted symptom summary: {summary}")
            return summary

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"LLM output parsing/validation error: {e}. Raw output: '{raw_output}'")

            # Fallback: Use regex-based extraction for each field.
            transcript_lower = transcript.lower()

            # Broader symptom detection with a regex pattern.
            # Using re.findall to capture all occurrences so that we can select the most relevant one.
            symptom_pattern = r"\b(fever|headache|cough|coughing|pain|ache|sore|nausea|dizziness|fatigue|shortness of breath|vomiting|diarrhea|rash|chills|congestion|swelling|itching|bleeding|weakness)\b"
            symptom_matches = re.findall(symptom_pattern, transcript_lower)
            if symptom_matches:
                # Choose the last match as the primary symptom (helps when adjectives precede the symptom)
                summary["key_symptom"] = symptom_matches[-1]

            # Extract severity
            severity_pattern = r"\b(severe|mild|bad|terrible|awful|slight|moderate|extreme|worse|better|intense|[0-9]+/[0-9]+)\b"
            severity_match = re.search(severity_pattern, transcript_lower)
            if severity_match:
                summary["severity"] = severity_match.group(0)

            # Extract onset/duration
            onset_pattern = r"\b(since|yesterday|today|for\s+\d+\s+(days?|weeks?|months?)|last\s+\w+|over\s+the\s+weekend|recently|suddenly|started|ongoing)\b"
            onset_match = re.search(onset_pattern, transcript_lower)
            if onset_match:
                summary["onset_duration"] = onset_match.group(0)

            # Extract location
            location_pattern = r"\b(head|neck|throat|chest|back|stomach|abdomen|arm|leg|hand|foot|all\s+over|whole\s+body|side|left|right|upper|lower)\b"
            location_match = re.search(location_pattern, transcript_lower)
            if location_match:
                summary["location"] = location_match.group(0)

            # Extract character
            character_pattern = r"\b(sharp|dull|throbbing|pounding|burning|stabbing|aching|tight|sore|constant|off\s+and\s+on|intermittent)\b"
            character_match = re.search(character_pattern, transcript_lower)
            if character_match:
                summary["character"] = character_match.group(0)

            # Extract associated symptoms (secondary occurrences excluding the primary symptom)
            assoc_matches = [m for m in re.findall(symptom_pattern, transcript_lower) if m != summary["key_symptom"]]
            if assoc_matches:
                summary["associated_symptoms"] = ", ".join(assoc_matches)

            # Final LLM-based fallback if key symptom is still missing
            if summary["key_symptom"] == "unknown symptom":
                fallback_prompt = (
                    "You are a medical assistant. Identify the most prominent symptom or health issue from this patient description in a few words.\n\n"
                    f"Patient Description: \"{transcript}\"\nPrimary Symptom:"
                )
                fallback_symptom = self.generate_text(
                    fallback_prompt,
                    max_new_tokens=15,
                    num_beams=2,
                    temperature=0.5,
                    repetition_penalty=1.1
                )
                if fallback_symptom and not fallback_symptom.startswith("Error:"):
                    summary["key_symptom"] = fallback_symptom.strip()

            logger.warning(f"Using regex fallback for symptom extraction: {summary}")
            return summary

        except Exception as e:
            logger.error(f"Unexpected error during symptom extraction: {e}", exc_info=True)
            logger.warning(f"Returning default summary: {summary}")
            return summary

    # --- Final generate_followup_questions function --
    def generate_followup_questions(self, reviewed_transcript: str, symptom_summary: dict, static_followup: list) -> list:
        """
        Generates three distinct, clinically relevant follow-up questions based on
        an initial transcript, a structured symptom summary, and previous static questions.
        Output questions do not have leading numbers or bullets.
        """
        # --- Get Key Symptom for fallbacks/logging ---
        key_symptom = symptom_summary.get('key_symptom', 'the symptom') # Use from summary

        # --- Define get_question_topic (no change needed here) ---
        def get_question_topic(question_text):
            # (Keep your existing get_question_topic function)
            q_lower = question_text.lower()
            if any(kw in q_lower for kw in ["scale of 1 to 10", "how severe", "how bad", "intensity"]): return "Severity"
            if any(kw in q_lower for kw in ["describe", "quality", "feel like", "sharp", "dull", "constant", "intermittent"]): return "Character"
            if any(kw in q_lower for kw in ["when did", "start", "how long", "how often", "frequency", "pattern"]): return "Onset/Timing"
            if any(kw in q_lower for kw in ["where", "location", "point to", "spread", "radiate"]): return "Location"
            if any(kw in q_lower for kw in ["trigger", "worsen", "better", "relieve", "help", "specific activities"]): return "Triggers/Relief"
            if any(kw in q_lower for kw in ["affect", "impact", "daily", "work", "sleep", "function"]): return "Impact"
            if any(kw in q_lower for kw in ["other symptoms", "anything else", "accompanied by"]): return "Associated Symptoms"
            if any(kw in q_lower for kw in ["tried", "taken anything", "treatment", "medication", "remedies"]): return "Treatments Tried"
            return "Other"

        # --- Static Question Processing (no change needed here) ---
        static_questions_text = [item.get('question', '').strip() for item in static_followup if isinstance(item, dict) and item.get('question')]
        static_topics_covered = set(get_question_topic(q) for q in static_questions_text if get_question_topic(q) != "Other")
        static_questions_context = "\n".join([f"- {q}" for q in static_questions_text]) if static_questions_text else "None provided."

        # --- Topic Selection Logic (no change needed here) ---
        all_topics_ordered = ["Severity", "Character", "Onset/Timing", "Location", "Triggers/Relief", "Associated Symptoms", "Impact", "Treatments Tried"]
        uncovered_topics = [topic for topic in all_topics_ordered if topic not in static_topics_covered]
        # (Keep fallback logic for uncovered_topics if len < 3)
        if len(uncovered_topics) < 3:
             fallback_suggestions = ["Impact", "Treatments Tried", "Associated Symptoms"]
             needed = 3 - len(uncovered_topics)
             added_count = 0
             for fb_topic in fallback_suggestions:
                 if added_count < needed and fb_topic not in uncovered_topics and fb_topic not in static_topics_covered:
                     uncovered_topics.append(fb_topic)
                     added_count += 1
             idx = 0
             while len(uncovered_topics) < 3:
                 topic_candidate = all_topics_ordered[idx % len(all_topics_ordered)]
                 if topic_candidate not in uncovered_topics:
                      uncovered_topics.append(topic_candidate)
                 idx += 1


        # --- **MODIFIED**: Build Richer Context for the Prompt ---
        initial_summary_text = (
            f"- Main Complaint: {symptom_summary.get('key_symptom', 'Not specified')}\n"
            f"- Severity Mentioned: {symptom_summary.get('severity', 'Not specified')}\n"
            f"- Onset/Duration Mentioned: {symptom_summary.get('onset_duration', 'Not specified')}\n"
            f"- Location Mentioned: {symptom_summary.get('location', 'Not specified')}\n"
            f"- Character Mentioned: {symptom_summary.get('character', 'Not specified')}\n"
            f"- Associated Symptoms Mentioned: {symptom_summary.get('associated_symptoms', 'Not specified')}"
        )

        context_for_llm = (
            f"Patient's Initial Statement Analysis:\n{initial_summary_text}\n"
            f"(Full Transcript: \"{reviewed_transcript}\")\n\n" # Keep full transcript for nuance
            "Static Follow-up Questions Already Asked (DO NOT repeat questions on these topics):\n"
            f"{static_questions_context}\n"
            f"Topics covered by static questions: {', '.join(static_topics_covered) if static_topics_covered else 'None'}\n"
        )

        # --- **MODIFIED**: Update Prompt with Richer Context ---
        prompt = (
            "You are a clinical assistant reviewing initial patient information and planning follow-up questions.\n"
            "Based on the analyzed initial statement and the static questions already asked (context below), generate exactly THREE distinct, open-ended follow-up questions "
            f"to further investigate the patient's condition, focusing on the main complaint: '{key_symptom}'.\n"
            "Ensure the questions are patient-friendly, end with a question mark, and explore different clinical aspects.\n"
            "Crucially, AVOID asking about topics already well-covered by the 'Static Follow-up Questions' or clearly detailed in the 'Initial Statement Analysis'.\n"
            "Focus on gathering more details about these potentially uncovered areas:\n"
            f"1. {uncovered_topics[0]} (e.g., Ask about specific triggers, character, or impact if not detailed)\n"
            f"2. {uncovered_topics[1]} (e.g., Ask about treatments tried, associated symptoms, or timing patterns if unclear)\n"
            f"3. {uncovered_topics[2]} (e.g., Ask clarifying questions based on initial details or explore less covered areas)\n"
            "Phrase your questions naturally. Output ONLY the three questions, each on a new line. Do NOT add numbers like '1.' or bullets like '-' to the beginning of the questions.\n\n"
            f"### Context:\n{context_for_llm}\n"
            "### OUTPUT (3 Questions, plain text, one per line):"
        )

        # --- LLM Call and Post-processing (Keep the existing logic) ---
        result = self.generate_text(prompt, max_new_tokens=150, num_beams=3, temperature=0.6, repetition_penalty=1.2) # Or use FOLLOWUP_PARAMS
        generated_questions_raw = [line.strip() for line in result.split("\n") if line.strip() and line.strip().endswith("?")]

        distinct_questions = []
        seen_topics = set(static_topics_covered)
        # Also consider topics mentioned in initial summary as partially covered
        for key, value in symptom_summary.items():
            if value.lower() != 'not specified' and value.lower() != 'none mentioned':
                topic = get_question_topic(f"{key}: {value}") # Rough topic mapping
                if topic != "Other":
                    seen_topics.add(topic)

        seen_questions_text = set(q.lower() for q in static_questions_text)

        # (Keep the filtering logic for generated_questions_raw, using seen_topics and seen_questions_text)
        for q in generated_questions_raw:
             q_lower = q.lower()
             is_similar_to_static = any(q_lower in static_q.lower() or static_q.lower() in q_lower for static_q in static_questions_text)
             if is_similar_to_static: continue

             topic = get_question_topic(q)
             cleaned_q = re.sub(r"^\s*(\d+\.|\*|-)\s*", "", q).strip()
             if not cleaned_q: continue

             cleaned_q_lower = cleaned_q.lower()
             # Check if topic is truly new *and* question text is new
             if topic not in seen_topics and cleaned_q not in distinct_questions and cleaned_q_lower not in seen_questions_text:
                 distinct_questions.append(cleaned_q)
                 seen_questions_text.add(cleaned_q_lower)
                 if topic != "Other": seen_topics.add(topic)
             # Allow adding if topic was seen but question is different (less preferred)
             elif topic in seen_topics and cleaned_q not in distinct_questions and cleaned_q_lower not in seen_questions_text:
                  distinct_questions.append(cleaned_q)
                  seen_questions_text.add(cleaned_q_lower)

             if len(distinct_questions) == 3: break

        # --- Fallback Question Generation (Use key_symptom extracted earlier) ---
        fallback_options = {
            # (Keep your fallback_options dictionary, it uses key_symptom correctly)
             "Severity": f"On a scale of 1 to 10, how would you rate the severity of your {key_symptom} right now?",
             "Character": f"Can you describe what the {key_symptom} feels like? For example, is it sharp, dull, burning, aching?",
             # ... rest of fallbacks
             "Treatments Tried": f"Have you tried any medications or home remedies for this {key_symptom}, and did they help?"
        }

        final_uncovered_topics = [topic for topic in all_topics_ordered if topic not in seen_topics]

        # (Keep the fallback filling logic)
        idx = 0
        while len(distinct_questions) < 3 and idx < len(final_uncovered_topics):
            # ... (fill with fallbacks) ...
             topic_to_add = final_uncovered_topics[idx]
             fallback_question = fallback_options.get(topic_to_add)
             if fallback_question and fallback_question not in distinct_questions and fallback_question.lower() not in seen_questions_text:
                 distinct_questions.append(fallback_question)
                 seen_topics.add(topic_to_add)
                 seen_questions_text.add(fallback_question.lower())
             idx += 1
        # ... (fill with generic fallbacks if needed) ...
        generic_fallbacks = [
             f"Can you tell me a bit more about the {key_symptom}?",
             f"Is there anything else important about the {key_symptom} I should know?",
             f"How concerned are you about this {key_symptom}?"
        ]
        idx = 0
        while len(distinct_questions) < 3 and idx < len(generic_fallbacks):
             if generic_fallbacks[idx] not in distinct_questions and generic_fallbacks[idx].lower() not in seen_questions_text:
                 distinct_questions.append(generic_fallbacks[idx])
                 seen_questions_text.add(generic_fallbacks[idx].lower())
             idx += 1


        # --- Final Cleaning (Keep the existing logic) ---
        final_cleaned_questions = []
        for q in distinct_questions[:3]:
            # (clean leading markers)
             cleaned_q = re.sub(r"^\s*(\d+\.|\*|-)\s*", "", q).strip()
             if cleaned_q:
                 final_cleaned_questions.append(cleaned_q)


        logger.info(f"Generated dynamic questions using richer context (final cleaned): {final_cleaned_questions}")
        return final_cleaned_questions

    # --- Final generate_guidelines function ---
    # Replace the old generate_guidelines method
    def generate_guidelines(self, reviewed_transcript: str, symptom_summary: dict, static_followup: list, dynamic_followup: list) -> dict:
        """
        Generates a personalized home care plan and a concise summary based on the patient's conversation and symptom analysis.

        This function builds a detailed prompt from:
          - The structured symptom summary.
          - The full reviewed transcript.
          - Static and dynamic follow-up Q&A.

        The LLM is then used to generate a single, flowing paragraph (approximately 100-150 words) with plain text advice.
        The advice includes general care recommendations, specific non-pharmacological suggestions, possible OTC options,
        modifications to activities, and red flag conditions that would require further medical evaluation.

        A standard disclaimer is appended to the result. If the LLM fails or returns an empty result, a fallback guideline
        is used based on the primary symptom.

        Args:
            reviewed_transcript (str): The full transcript of the patient consultation.
            symptom_summary (dict): A dictionary with keys such as 'key_symptom', 'severity', 'onset_duration', 'location',
                                    'character', and 'associated_symptoms'.
            static_followup (list): A list of dictionaries for static follow-up Q&A.
            dynamic_followup (list): A list of dictionaries for dynamic follow-up Q&A.

        Returns:
            dict: A single paragraph of personalized home care advice with an appended disclaimer and 'summary' (concise 2-3 line overview).
        """
        # Retrieve key symptom from summary, with a fallback label.
        key_symptom = symptom_summary.get('key_symptom', 'the symptom')

        # Format static follow-up questions and answers.
        static_followup_text = "\n".join(
            f"- Q: {item.get('question', '').strip()}\n  A: {item.get('answer', '').strip()}"
            for item in static_followup if isinstance(item, dict) and item.get('question') and item.get('answer')
        )

        # Format dynamic follow-up questions and answers.
        dynamic_followup_text = "\n".join(
            f"- Q: {item.get('question', '').strip()}\n  A: {item.get('answer', '').strip()}"
            for item in dynamic_followup if isinstance(item, dict) and item.get('question') and item.get('answer')
        )

        # Create an initial summary of symptom details.
        initial_summary_text = (
            f"- Main Complaint: {symptom_summary.get('key_symptom', 'Not specified')}\n"
            f"- Initial Severity: {symptom_summary.get('severity', 'Not specified')}\n"
            f"- Initial Onset/Duration: {symptom_summary.get('onset_duration', 'Not specified')}\n"
            f"- Initial Location: {symptom_summary.get('location', 'Not specified')}\n"
            f"- Initial Character: {symptom_summary.get('character', 'Not specified')}\n"
            f"- Initial Associated Symptoms: {symptom_summary.get('associated_symptoms', 'Not specified')}"
        )

        # Build detailed context including transcript and follow-up Q&A.
        detailed_context = (
            "Conversation Summary:\n"
            "Patient's Initial Statement Analysis:\n" + initial_summary_text + "\n"
            f"(Full Transcript: \"{reviewed_transcript}\")\n\n"
            "Static Follow-up Q&A:\n" + (static_followup_text if static_followup_text else "None provided.\n") + "\n"
            "Dynamic Follow-up Q&A:\n" + (dynamic_followup_text if dynamic_followup_text else "None provided.\n")
        )

        # Build the full prompt for the LLM.
        prompt = (
            "You are a caring physician summarizing home care advice for a patient after a consultation.\n"
            "Based specifically on the conversation summary below (including initial analysis and follow-up answers), "
            "create a concise, personalized home care plan. The plan should focus on practical strategies for managing the "
            f"patient's condition, primarily related to '{key_symptom}'. Use simple, supportive, and empathetic language "
            "directly addressing the patient.\n\n"
            "Present the advice as a single, flowing paragraph of text (around 100-150 words). "
            "DO NOT use any lists, bullet points, numbering, or bold text in your final output; just provide plain paragraph text.\n\n"
            "Tailor the advice by considering all provided details, including:\n"
            "- General advice (e.g., rest, hydration).\n"
            "- Specific non-pharmacological suggestions relevant to the symptoms and follow-up details.\n"
            "- Mention of over-the-counter (OTC) options if appropriate based on the full context.\n"
            "- Activities to potentially avoid or modify based on triggers or patient responses.\n"
            "- Clear red flag conditions under which the patient should seek further medical attention.\n\n"
            "### Conversation Summary:\n" + detailed_context + "\n"
            "### Your Personalized Home Care Plan (Single Paragraph, Plain Text):"
        )

        try:
            # Generate the home care plan using the LLM.
            result = self.generate_text(
                prompt,
                max_new_tokens=250,
                num_beams=3,
                temperature=0.6,
                repetition_penalty=1.15
            )
        except Exception as e:
            logger.error(f"Error during LLM guideline generation: {e}", exc_info=True)
            result = ""

        # Fallback guideline in case of error or empty result.
        fallback_guideline = (
            f"Based on our discussion regarding your {key_symptom}, please ensure you get plenty of rest and maintain proper hydration. "
            "If you experience any worsening of symptoms or notice new concerning signs, consider over-the-counter remedies as appropriate and avoid activities that trigger discomfort. "
            "Should your condition not improve or if you experience significant changes, please seek prompt medical advice."
        )

        disclaimer = "\n\n*Disclaimer: This is general automated advice based on the conversation. Consult a healthcare professional for diagnosis and treatment.*"

        if result.startswith("Error:") or not result.strip():
            logger.warning(f"LLM guideline generation failed or returned empty. Using fallback for key_symptom: {key_symptom}")
            final_guidelines = fallback_guideline
        else:
            final_guidelines = result.strip()

        # Clean up the generated output: remove any unintended formatting.
        final_guidelines = re.sub(r"^\s*(\d+\.|\*|-)\s*", "", final_guidelines, flags=re.MULTILINE)
        final_guidelines = re.sub(r"\*\*(.*?)\*\*", r"\1", final_guidelines)
        final_guidelines = re.sub(r"\s+", " ", final_guidelines).strip()

        # Ensure the result appears like home care advice.
        if not any(phrase in final_guidelines.lower() for phrase in ["based on", "for your", "try", "you should", "it's important", "please focus"]):
            final_guidelines = f"For your {key_symptom}, here are some home care suggestions: " + final_guidelines

        # Generate concise summary from the guidelines (before adding disclaimer)
        summary_text = self.generate_summary(final_guidelines)

        # Return both guidelines (with disclaimer) and summary
        return {"guidelines": final_guidelines + disclaimer, "summary": summary_text}

    def generate_summary(self, guidelines_text: str) -> str:
        """
        Generates a concise 2-3 line summary of the provided guidelines, focusing on key points and overall advice.
        """
        prompt = (
            "Based on the following home care guidelines, provide a very concise summary (2-3 lines) that highlights the key points and overall advice. "
            "Ensure the summary is clear, patient-friendly, and directly addresses the patient.\n\n"
            f"Guidelines:\n{guidelines_text}\n\n"
            "Concise Summary:"
        )
        summary = self.generate_text(
            prompt,
            max_new_tokens=100,  # Limits length to ensure conciseness (approx. 60-80 words for 2-3 lines)
            num_beams=3,
            temperature=0.5,     # Lower temperature for focused, clear output
            repetition_penalty=1.2
        )
        return summary.strip()

llm_handler = LLMHandler()

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    audio_file = request.files["file"]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "audio.wav")
        try:
            audio_file.save(temp_path)
        except Exception as e:
            logger.error(f"Error saving file: {e}", exc_info=True)
            return jsonify({"error": f"Failed to save file: {e}"}), 500

        whisper_model = None
        try:
            # Offload Whisper tasks to a thread pool
            with executor as pool:
                whisper_model = pool.submit(whisper.load_model, "small.en", device=DEVICE).result()
                result = pool.submit(whisper_model.transcribe, temp_path, fp16=False).result()
        except Exception as e:
            logger.error("Transcription failed", exc_info=True)
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
        finally:
            if whisper_model is not None:
                del whisper_model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    transcript = re.sub(r'\s+', ' ', result.get("text", "").strip())
    if not transcript:
        return jsonify({"error": "Transcription resulted in empty text."}), 500
    return jsonify({"transcript": transcript})

# --- In your Flask app (server.py) ---

# Remove the old '/extract_symptoms' and replace with this:
@app.route("/extract_symptom_summary", methods=["POST"])
def extract_symptom_summary_endpoint():
    data = request.get_json()
    if not data or "transcript" not in data:
        return jsonify({"error": "No transcript provided."}), 400

    transcript = data["transcript"].strip()
    if not transcript:
         return jsonify({"error": "Empty transcript provided."}), 400

    try:
        # Use the new method in LLMHandler
        # Consider running this in the executor if it becomes slow
        summary = llm_handler.extract_structured_symptoms(transcript)
        return jsonify({"symptom_summary": summary})
    except Exception as e:
        logger.error("Error during symptom summary extraction", exc_info=True)
        return jsonify({"error": f"Failed to extract symptom summary: {str(e)}"}), 500

# --- In your Flask app (server.py) ---

# Modify the '/generate_followup_questions' endpoint
@app.route("/generate_followup_questions", methods=["POST"])
def generate_followup_questions_endpoint():
    data = request.get_json()
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid request body."}), 400

    reviewed_transcript = data.get("reviewed_transcript", "").strip()
    # --- **MODIFIED**: Expect symptom_summary dictionary ---
    symptom_summary = data.get("symptom_summary") # Expect the dict now
    static_followup = data.get("static_followup", [])

    # --- **MODIFIED**: Validate symptom_summary ---
    if not reviewed_transcript or not symptom_summary or not isinstance(symptom_summary, dict):
        # Ensure key_symptom exists within the summary for basic operation
        if not symptom_summary or not symptom_summary.get("key_symptom"):
             return jsonify({"error": "Missing required fields: reviewed_transcript or valid symptom_summary."}), 400

    try:
        # --- **MODIFIED**: Pass symptom_summary to the handler ---
        questions = llm_handler.generate_followup_questions(reviewed_transcript, symptom_summary, static_followup)
        return jsonify({"follow_up_questions": questions})
    except Exception as e:
        logger.error("Error in follow-up questions", exc_info=True)
        return jsonify({"error": f"Error generating follow-up questions: {str(e)}"}), 500

# --- In your Flask app (server.py) ---

# Modify the '/generate_guidelines' endpoint
@app.route("/generate_guidelines", methods=["POST"])
def generate_guidelines_endpoint():
    data = request.get_json()
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid request body."}), 400

    # --- **MODIFIED**: Expect transcript AND symptom_summary ---
    # Keep transcript for the full context if needed by handler
    transcript = data.get("transcript", "").strip() # Keep original transcript
    symptom_summary = data.get("symptom_summary") # Expect the dict
    static_followup = data.get("static_followup", [])
    dynamic_followup = data.get("dynamic_followup", [])

    # --- **MODIFIED**: Validate symptom_summary ---
    if not transcript or not symptom_summary or not isinstance(symptom_summary, dict):
         # Ensure key_symptom exists within the summary for basic operation
         if not symptom_summary or not symptom_summary.get("key_symptom"):
             return jsonify({"error": "Transcript and valid symptom_summary required."}), 400

    # --- **MODIFIED**: Pass symptom_summary to the handler ---
    # Offload LLM generation to a thread pool (keep this)
    # Generate guidelines and summary
    with executor as pool:
        future = pool.submit(
            llm_handler.generate_guidelines, transcript, symptom_summary, static_followup, dynamic_followup
        )
        result = future.result()
        guidelines_text = result["guidelines"]
        summary_text = result["summary"]

    if not guidelines_text or guidelines_text.startswith("Error:"): # Check for errors here too
        # Attempt to provide a basic fallback even if generation failed
        key_symptom = symptom_summary.get("key_symptom", "your symptom")
        guidelines_text = (
             f"For your {key_symptom}, focus on getting adequate rest and staying hydrated. "
             "Monitor your symptoms closely. If they worsen or don't improve, please seek medical attention."
             "\n\n*Disclaimer: This is general automated advice. Consult a healthcare professional for diagnosis and treatment.*"
        )
        # Don't return 500, provide the fallback text
        # return jsonify({"error": "Failed to generate guidelines."}), 500

    # --- Text-to-Speech (Keep this logic) ---
    # Use summary for TTS
    audio_data = ""
    try:
        if summary_text:
            tts = gTTS(text=summary_text, lang='en', slow=False)
            audio_io = io.BytesIO()
            tts.write_to_fp(audio_io)
            audio_io.seek(0)
            audio_base64 = base64.b64encode(audio_io.read()).decode('utf-8')
            audio_data = f"data:audio/mp3;base64,{audio_base64}"
    except Exception as e:
        logger.error("TTS error", exc_info=True)

    # Return summary, guidelines, and audio data
    return jsonify({"summary": summary_text, "guidelines": guidelines_text, "audio_data": audio_data})


if __name__ == "__main__":
    print("Preloading LLM model...")
    try:
        llm_handler.load_model()
    except Exception as e:
        print(f"Failed to preload model: {e}. Falling back to on-demand loading.")

    public_url = None
    try:
        ngrok.kill()
        public_url = ngrok.connect("5000")
        print(f"Ngrok Tunnel active at: {public_url}")
    except Exception as e:
        print(f"Failed to start Ngrok tunnel: {e}")

    print("Starting Flask server on http://0.0.0.0:5000")
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        executor.shutdown(wait=True)  # Clean up thread pool on exit
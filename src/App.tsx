import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Button } from './components/ui/button';
import { Mic, Loader2, User } from 'lucide-react';
import './styles.css';

// Adjust the API_BASE_URL to match your server endpoint
const API_BASE_URL = 'https://5f54-35-198-212-87.ngrok-free.app/';

// Define application steps
type AppStep =
  | 'initial'
  | 'recording'
  | 'review'
  | 'followupLoading'
  | 'followup'
  | 'dynamicFollowupLoading'
  | 'dynamicFollowup'
  | 'final';

// Define message structure for conversation
interface Message {
  sender: 'doctor' | 'patient';
  text: string;
  timestamp: Date;
}

// Define symptom summary structure
interface SymptomSummary {
  key_symptom: string;
  severity: string;
  onset_duration: string;
  location: string;
  character: string;
  associated_symptoms: string;
}

/** Utility to speak text, canceling any ongoing speech */
function speak(text: string) {
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.9;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
}

/** Starts browser-based speech recognition with silence detection */
const startSpeechRecognition = (
  onResult: (transcript: string) => void,
  onError: (err: any) => void,
  toggleMic?: (active: boolean) => void
) => {
  const SpeechRecognition =
    (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  if (!SpeechRecognition) {
    onError("Speech recognition not supported. Please type your response below.");
    return null;
  }
  const recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;
  recognition.continuous = true;

  let finalTranscript = '';
  let silenceTimeout: NodeJS.Timeout | null = null;
  const SILENCE_THRESHOLD = 1000; // 1 second of silence

  recognition.onstart = () => toggleMic && toggleMic(true);

  recognition.onresult = (event: any) => {
    let interimTranscript = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      if (result.isFinal) {
        finalTranscript += result[0].transcript + ' ';
      } else {
        interimTranscript += result[0].transcript;
      }
    }

    if (interimTranscript || finalTranscript) {
      if (silenceTimeout) clearTimeout(silenceTimeout);
      silenceTimeout = setTimeout(() => {
        recognition.stop();
      }, SILENCE_THRESHOLD);
    }
  };

  recognition.onend = () => {
    toggleMic && toggleMic(false);
    if (silenceTimeout) clearTimeout(silenceTimeout);
    onResult(finalTranscript.trim() || '');
  };

  recognition.onerror = (event: any) => {
    toggleMic && toggleMic(false);
    onError(event.error);
    if (silenceTimeout) clearTimeout(silenceTimeout);
  };

  recognition.start();
  return recognition;
};

/** Speaks a message and listens for a response */
const speakAndListen = (
  message: string,
  onResult: (result: string) => void,
  addMessage: (msg: Message) => void,
  toggleMic: (active: boolean) => void,
  setCurrentPrompt: (prompt: string) => void
) => {
  window.speechSynthesis.cancel();
  setCurrentPrompt(message);
  const utterance = new SpeechSynthesisUtterance(message);
  utterance.rate = 0.9;
  utterance.pitch = 1.0;
  utterance.onend = () => {
    setCurrentPrompt('');
    startSpeechRecognition(
      (result) => {
        addMessage({ sender: 'patient', text: result, timestamp: new Date() });
        onResult(result);
      },
      (err) => onResult(""), // Pass empty string on error to trigger retry
      toggleMic
    );
  };
  window.speechSynthesis.speak(utterance);
  addMessage({ sender: 'doctor', text: message, timestamp: new Date() });
};

/** Custom hook for voice input with retry logic */
const useVoiceInput = (
  addMessage: (msg: Message) => void,
  toggleMic: (active: boolean) => void,
  setCurrentPrompt: (prompt: string) => void
) => {
  const voiceInput = useCallback(
    async (promptMsg: string): Promise<string> => {
      let attempts = 0;
      const MAX_ATTEMPTS = 3;

      while (true) {
        try {
          const result: string = await new Promise((resolve) => {
            speakAndListen(promptMsg, resolve, addMessage, toggleMic, setCurrentPrompt);
          });
          if (result) return result; // Return if valid response received
          throw new Error("No input detected");
        } catch (err) {
          attempts++;
          if (attempts < MAX_ATTEMPTS) {
            const retryMsg = `I’m sorry, I didn’t catch that. Could you say it again? This is try ${attempts + 1} of ${MAX_ATTEMPTS}.`;
            speak(retryMsg);
            addMessage({ sender: 'doctor', text: retryMsg, timestamp: new Date() });
          } else {
            const continueMsg = "I’m still having trouble hearing you. Let’s try one more time, or you can type if you prefer.";
            speak(continueMsg);
            addMessage({ sender: 'doctor', text: continueMsg, timestamp: new Date() });
            attempts = 0; // Reset attempts
          }
        }
      }
    },
    [addMessage, toggleMic, setCurrentPrompt]
  );

  return { voiceInput };
};

/** API function to extract full symptom summary with retries */
async function extractSymptoms(transcript: string, retries = 3): Promise<SymptomSummary> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await fetch(`${API_BASE_URL}/extract_symptom_summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript }),
      });
      if (!response.ok) throw new Error(`Attempt ${attempt} failed`);
      const data = await response.json();
      return data.symptom_summary;
    } catch (error) {
      if (attempt === retries) throw new Error("Symptom extraction failed after retries");
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  throw new Error("Unexpected error in extractSymptoms");
}

/** API function to generate follow-up questions based on full symptom summary */
async function generateFollowupQuestions(
  reviewed_transcript: string,
  symptom_summary: SymptomSummary,
  static_followup: { question: string; answer: string }[],
  retries = 3
): Promise<string[]> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await fetch(`${API_BASE_URL}/generate_followup_questions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reviewed_transcript, symptom_summary, static_followup }),
      });
      if (!response.ok) throw new Error(`Attempt ${attempt} failed`);
      const data = await response.json();
      return data.follow_up_questions;
    } catch (error) {
      if (attempt === retries) throw new Error("Dynamic follow-up question generation failed after retries");
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  throw new Error("Unexpected error in generateFollowupQuestions");
}

/** Main App Component */
const App: React.FC = () => {
  const [step, setStep] = useState<AppStep>('initial');
  const [transcript, setTranscript] = useState<string>('');
  const [symptomSummary, setSymptomSummary] = useState<SymptomSummary | null>(null);
  const [staticFollowUpQuestions, setStaticFollowUpQuestions] = useState<string[]>([]);
  const [staticFollowUpAnswers, setStaticFollowUpAnswers] = useState<string[]>([]);
  const [dynamicFollowUpQuestions, setDynamicFollowUpQuestions] = useState<string[]>([]);
  const [dynamicFollowUpAnswers, setDynamicFollowUpAnswers] = useState<string[]>([]);
  const [currentStaticIndex, setCurrentStaticIndex] = useState<number>(0);
  const [currentDynamicIndex, setCurrentDynamicIndex] = useState<number>(0);
  const [guidelines, setGuidelines] = useState<string>('');
  const [audioSrc, setAudioSrc] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [overlayVisible, setOverlayVisible] = useState<boolean>(true);
  const [welcomeSpoken, setWelcomeSpoken] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string>('');
  const [conversation, setConversation] = useState<Message[]>([]);
  const [micActive, setMicActive] = useState<boolean>(false);
  const [currentPrompt, setCurrentPrompt] = useState<string>('');
  const [isManuallyPaused, setIsManuallyPaused] = useState<boolean>(false);
  const isConfirmingRef = useRef<boolean>(false);
  const isAskingStaticRef = useRef<boolean>(false);
  const isAskingDynamicRef = useRef<boolean>(false);
  const isLoadingDynamicRef = useRef<boolean>(false);
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const waitingIntervalRef = useRef<number | null>(null);

  const { voiceInput } = useVoiceInput(
    (msg: Message) => addMessage(msg),
    setMicActive,
    setCurrentPrompt
  );

  // Auto-scroll to the latest conversation message
  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation]);

  const addMessage = (msg: Message) => setConversation((prev) => [...prev, msg]);

  // Memoized step indicator
  const getStepIndicator = useMemo(
    () => () => {
      switch (step) {
        case 'initial': return 'Welcome to Your Health Journey';
        case 'recording': return 'Step 1: Sharing How You Feel';
        case 'review': return 'Step 2: Let’s Double-Check';
        case 'followupLoading': return 'Step 3: Preparing Questions for You';
        case 'followup': return 'Step 3: Learning More About You';
        case 'dynamicFollowupLoading': return 'Step 4: Asking a Few More Things';
        case 'dynamicFollowup': return 'Step 4: Getting to Know You Better';
        case 'final': return 'Step 5: Your Personal Care Plan';
        default: return '';
      }
    },
    [step]
  );

  // Memoized progress percentage
  const getProgressPercentage = useMemo(
    () => () => {
      switch (step) {
        case 'initial': return '0%';
        case 'recording': return '20%';
        case 'review': return '40%';
        case 'followupLoading': return '50%';
        case 'followup': return '60%';
        case 'dynamicFollowupLoading': return '70%';
        case 'dynamicFollowup': return '80%';
        case 'final': return '100%';
        default: return '0%';
      }
    },
    [step]
  );

  // Loading message based on step
  const getLoadingMessage = (): string => {
    switch (step) {
      case 'followupLoading': return "We’re preparing some thoughtful questions to better understand how you’re doing.";
      case 'dynamicFollowupLoading': return "Just a moment—we’re tailoring a few more questions specifically for you.";
      case 'final': return "We’re putting together a care plan just for you to help you feel better soon.";
      default: return "We’re working on something special to support you...";
    }
  };

  /** Handle waiting message during long loading periods */
  useEffect(() => {
    if (loading && (step === 'followupLoading' || step === 'dynamicFollowupLoading' || step === 'final')) {
      waitingIntervalRef.current = window.setInterval(() => {
        const waitingMsg = "We’re still working on this for you. Hang in there, we’ll be with you shortly!";
        speak(waitingMsg);
        addMessage({ sender: 'doctor', text: waitingMsg, timestamp: new Date() });
      }, 20000);
    } else if (waitingIntervalRef.current) {
      clearInterval(waitingIntervalRef.current);
      waitingIntervalRef.current = null;
    }

    return () => {
      if (waitingIntervalRef.current) {
        clearInterval(waitingIntervalRef.current);
        waitingIntervalRef.current = null;
      }
    };
  }, [loading, step]);

  /** Handle overlay and welcome message */
  const handleOverlayClick = () => {
    if (!welcomeSpoken) {
      speak("Hello there! We’re so glad you’re here. Whenever you’re ready, just tell us how you’re feeling, and we’ll create a care plan tailored just for you. Tap 'Let’s Begin' when you’re ready to start.");
      addMessage({ sender: 'doctor', text: "Hello there! We’re so glad you’re here...", timestamp: new Date() });
      setWelcomeSpoken(true);
    }
    setOverlayVisible(false);
  };

  const handleStartRecording = async () => {
    setStep('recording');
    const result = await voiceInput("Whenever you’re ready, please tell me how you’re feeling today. What’s been going on?");
    setTranscript(result);
    addMessage({ sender: 'doctor', text: "Thank you for sharing that with me. I’ve noted what you said. You’re doing great!", timestamp: new Date() });
    setStep('review');
  };

  /** Review Step: Confirm transcript */
  useEffect(() => {
    if (step === 'review' && !isConfirmingRef.current && !isManuallyPaused) {
      isConfirmingRef.current = true;
      (async () => {
        try {
          const reviewMsg = `Here’s what I understood you said: "${transcript}". Does that sound right to you? Just say "yes" or "no."`;
          const result = await voiceInput(reviewMsg);
          if (result.toLowerCase().includes("yes")) {
            speak("Great, I’m glad we’re on the same page. Let’s move forward.");
            addMessage({ sender: 'doctor', text: "Great, I’m glad we’re on the same page. Let’s move forward.", timestamp: new Date() });
            setStep('followupLoading');
          } else {
            speak("No problem at all. Let’s try again. Can you tell me again how you’re feeling?");
            addMessage({ sender: 'doctor', text: "No problem at all. Let’s try again. Can you tell me again how you’re feeling?", timestamp: new Date() });
            setStep('recording');
            const retryResult = await voiceInput("Take your time and tell me again how you’re feeling.");
            setTranscript(retryResult);
            setStep('review');
          }
        } finally {
          isConfirmingRef.current = false;
        }
      })();
    }
  }, [step, transcript, voiceInput, isManuallyPaused]);

  /** Follow-Up Loading: Extract symptoms and prepare static questions */
  useEffect(() => {
    if (step === 'followupLoading' && !isManuallyPaused) {
      window.speechSynthesis.cancel();
      speak("I’m just getting some questions ready to learn more about how you’re feeling. Bear with me for a moment.");
      addMessage({ sender: 'doctor', text: "I’m just getting some questions ready to learn more about how you’re feeling. Bear with me for a moment.", timestamp: new Date() });
      setLoading(true);

      (async () => {
        try {
          const summary = await extractSymptoms(transcript);
          setSymptomSummary(summary);
          const cleanedSymptom = summary.key_symptom.toLowerCase().trim().replace(/[^a-z]/g, "");

          const staticMapping: { [key: string]: string[] } = {
            fever: [
              "Can you tell me when you first noticed the fever, and has it been constant or has it changed?",
              "Are you experiencing any chills, sweating, or body aches along with the fever?",
              "Have you recently been in a new place or around someone who was sick?"
            ],
            coughing: [
              "When did your cough start, and is it ongoing or does it come and go?",
              "Is your cough dry, or are you coughing up mucus or anything else?",
              "Are you having any trouble breathing, like feeling short of breath or tightness in your chest?"
            ],
            headache: [
              "When did your headache begin, and how severe or long-lasting is it?",
              "Is the headache in one specific area, or does it cover your whole head?",
              "Are you also feeling nauseous, sensitive to light or noise, or seeing anything unusual?"
            ]
          };

          const possibleKeys = Object.keys(staticMapping);
          const foundKey = possibleKeys.find(
            (key) => cleanedSymptom === key || cleanedSymptom.includes(key) || key.includes(cleanedSymptom)
          );

          if (foundKey) {
            const questions = staticMapping[foundKey];
            setStaticFollowUpQuestions(questions);
            setStaticFollowUpAnswers(Array(questions.length).fill(""));
            setCurrentStaticIndex(0);
            setStep('followup');
          } else {
            setStep('dynamicFollowupLoading');
          }
        } catch (error) {
          setErrorMsg("I’m sorry, we had a little trouble understanding that. Could we start over?");
          speak("I’m sorry, we had a little trouble understanding that. Could we start over?");
          setStep('initial');
        }
        setLoading(false);
      })();
    }
  }, [step, transcript, isManuallyPaused]);

  /** Static Follow-Up Questions */
  useEffect(() => {
    if (step === 'followup' && staticFollowUpQuestions.length > 0 && !isAskingStaticRef.current && !isManuallyPaused) {
      isAskingStaticRef.current = true;
      (async () => {
        try {
          const question = staticFollowUpQuestions[currentStaticIndex];
          const result = await voiceInput(question);
          const newAnswers = [...staticFollowUpAnswers];
          newAnswers[currentStaticIndex] = result;
          setStaticFollowUpAnswers(newAnswers);

          if (currentStaticIndex + 1 < staticFollowUpQuestions.length) {
            setCurrentStaticIndex(currentStaticIndex + 1);
          } else {
            speak("Thank you for those details. I’m going to ask a few more questions to get a clearer picture.");
            addMessage({ sender: 'doctor', text: "Thank you for those details. I’m going to ask a few more questions to get a clearer picture.", timestamp: new Date() });
            setStep('dynamicFollowupLoading');
          }
        } finally {
          isAskingStaticRef.current = false;
        }
      })();
    }
  }, [step, currentStaticIndex, staticFollowUpQuestions, staticFollowUpAnswers, voiceInput, isManuallyPaused]);

  /** Dynamic Follow-Up Questions */
  useEffect(() => {
    if (step === 'dynamicFollowupLoading' && !isLoadingDynamicRef.current && !isManuallyPaused) {
      isLoadingDynamicRef.current = true;
      window.speechSynthesis.cancel();
      speak("I’m putting together a few more questions to help me understand your situation even better. Just a moment.");
      addMessage({ sender: 'doctor', text: "I’m putting together a few more questions to help me understand your situation even better. Just a moment.", timestamp: new Date() });
      setLoading(true);

      (async () => {
        try {
          const staticQA = staticFollowUpQuestions.map((q, i) => ({ question: q, answer: staticFollowUpAnswers[i] }));
          const questions = await generateFollowupQuestions(transcript, symptomSummary!, staticQA);
          setDynamicFollowUpQuestions(questions);
          setDynamicFollowUpAnswers(Array(questions.length).fill(""));
          setCurrentDynamicIndex(0);
          setStep('dynamicFollowup');
        } catch (error) {
          speak("I’m sorry, we couldn’t generate more questions, but I’ll do my best with what we have. Let’s move forward.");
          addMessage({ sender: 'doctor', text: "I’m sorry, we couldn’t generate more questions, but I’ll do my best with what we have. Let’s move forward.", timestamp: new Date() });
          setStep('final');
        } finally {
          setLoading(false);
          isLoadingDynamicRef.current = false;
        }
      })();
    } else if (step === 'dynamicFollowup' && dynamicFollowUpQuestions.length > 0 && !isAskingDynamicRef.current && !isManuallyPaused) {
      isAskingDynamicRef.current = true;
      (async () => {
        try {
          const question = dynamicFollowUpQuestions[currentDynamicIndex];
          const result = await voiceInput(question);
          const newDynAnswers = [...dynamicFollowUpAnswers];
          newDynAnswers[currentDynamicIndex] = result;
          setDynamicFollowUpAnswers(newDynAnswers);

          if (currentDynamicIndex + 1 < dynamicFollowUpQuestions.length) {
            setCurrentDynamicIndex(currentDynamicIndex + 1);
          } else {
            speak("Thank you so much for all your answers. I’m now putting together your care plan.");
            addMessage({ sender: 'doctor', text: "Thank you so much for all your answers. I’m now putting together your care plan.", timestamp: new Date() });
            setStep('final');
          }
        } finally {
          isAskingDynamicRef.current = false;
        }
      })();
    }
  }, [step, dynamicFollowUpQuestions, currentDynamicIndex, transcript, symptomSummary, staticFollowUpQuestions, staticFollowUpAnswers, voiceInput, isManuallyPaused]);

  /** Generate Final Guidelines */
  useEffect(() => {
    if (step === 'final' && !loading && !isManuallyPaused) {
      setLoading(true);
      speak("I’m working on creating a special care plan just for you to help you feel better soon.");
      addMessage({ sender: 'doctor', text: "I’m working on creating a special care plan just for you to help you feel better soon.", timestamp: new Date() });

      const staticQA = staticFollowUpQuestions.map((q, i) => ({ question: q, answer: staticFollowUpAnswers[i] }));
      const dynamicQA = dynamicFollowUpQuestions.map((q, i) => ({ question: q, answer: dynamicFollowUpAnswers[i] }));

      fetch(`${API_BASE_URL}/generate_guidelines`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript,
          symptom_summary: symptomSummary,
          follow_up: staticQA,
          dynamic_followup: dynamicQA
        })
      })
        .then(res => {
          if (!res.ok) throw new Error("Guideline generation failed");
          return res.json();
        })
        .then(data => {
          if (data.guidelines) {
            setGuidelines(data.guidelines);
            if (data.audio_data) setAudioSrc(data.audio_data);
            speak(`Here’s your personalized care plan to help you feel better: ${data.guidelines}. If you need anything else, don’t hesitate to reach out.`);
            addMessage({ sender: 'doctor', text: `Here’s your personalized care plan to help you feel better: ${data.guidelines}. If you need anything else, don’t hesitate to reach out.`, timestamp: new Date() });
          }
        })
        .catch(() => {
          speak("Oh, I’m sorry, something went wrong while making your care plan. Could you try again later? We’re here to help.");
          addMessage({ sender: 'doctor', text: "Oh, I’m sorry, something went wrong while making your care plan. Could you try again later? We’re here to help.", timestamp: new Date() });
        })
        .finally(() => {
          setLoading(false);
        });
    }
  }, [step, transcript, symptomSummary, staticFollowUpQuestions, staticFollowUpAnswers, dynamicFollowUpQuestions, dynamicFollowUpAnswers, isManuallyPaused]);

  /** Restart the application */
  const handleRestart = () => {
    setStep('initial');
    setTranscript('');
    setSymptomSummary(null);
    setStaticFollowUpQuestions([]);
    setStaticFollowUpAnswers([]);
    setDynamicFollowUpQuestions([]);
    setDynamicFollowUpAnswers([]);
    setCurrentStaticIndex(0);
    setCurrentDynamicIndex(0);
    setGuidelines('');
    setAudioSrc('');
    setConversation([]);
    setOverlayVisible(true);
    setWelcomeSpoken(false);
    setErrorMsg('');
    setIsManuallyPaused(false);
    speak("Let’s start fresh. Whenever you’re ready, tap 'Let’s Begin'.");
  };

  return (
    <div className="flex flex-col w-full min-h-screen font-sans bg-gradient-to-br from-teal-50 via-blue-100 to-indigo-50">
      <header className="w-full px-6 py-6 shadow-lg bg-gradient-to-r from-teal-600 to-indigo-700">
        <h1 className="text-4xl font-bold tracking-tight text-center text-white">Your Health, Our Care</h1>
        <div className="relative w-full h-4 mt-4 overflow-hidden bg-teal-200 rounded-full">
          <div
            className="absolute top-0 left-0 h-full transition-all duration-500 ease-in-out bg-gradient-to-r from-teal-400 to-indigo-500"
            style={{ width: getProgressPercentage() }}
          />
          {["1: Share", "2: Confirm", "3: Questions", "4: More Details", "5: Care Plan"].map((label, idx) => (
            <div
              key={idx}
              className="absolute px-2 py-1 text-sm font-semibold text-white bg-teal-600 rounded-full -top-8 group"
              style={{ left: `${(idx * 100) / 4}%`, transform: 'translateX(-50%)' }}
            >
              {idx + 1}
              <span className="absolute hidden px-2 py-1 text-xs text-white transform -translate-x-1/2 bg-teal-800 rounded group-hover:block -top-10 left-1/2">
                {label.split(': ')[1]}
              </span>
            </div>
          ))}
        </div>
        <h2 className="mt-4 text-2xl font-semibold text-center text-teal-100">{getStepIndicator()}</h2>
      </header>

      <main className="flex-grow w-full max-w-5xl px-6 py-10 mx-auto">
        {overlayVisible && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-gradient-to-b from-teal-900/80 to-indigo-900/80 animate-fade-in" onClick={handleOverlayClick}>
            <div className="p-8 text-center transition-all transform bg-white shadow-2xl rounded-2xl hover:scale-105">
              <h1 className="mb-4 text-5xl font-bold text-teal-700">Welcome to Your Health Journey</h1>
              <p className="max-w-md mx-auto text-lg text-gray-600">
                We’re here to listen and support you. Tap anywhere to begin sharing how you’re feeling.
              </p>
            </div>
          </div>
        )}

        {errorMsg && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-60">
            <div className="w-full max-w-md p-8 text-center transition-all transform bg-white shadow-xl rounded-2xl animate-slide-up">
              <h2 className="mb-4 text-2xl font-semibold text-red-600">Sorry About That</h2>
              <p className="mb-6 text-gray-700">{errorMsg}</p>
              <Button onClick={() => setErrorMsg('')} className="px-6 py-3 text-white transition-colors bg-red-500 rounded-lg hover:bg-red-600" aria-label="Try again">
                Let’s Try Again
              </Button>
            </div>
          </div>
        )}

        {step === 'initial' && (
          <div className="flex flex-col items-center justify-center w-full space-y-8 animate-fade-in">
            <div className="relative flex items-center justify-center w-32 h-32 transition-transform transform bg-teal-100 rounded-full shadow-lg hover:scale-110">
              <Mic className="w-16 h-16 text-teal-600 animate-pulse" />
              <div className="absolute inset-0 border-4 border-teal-300 rounded-full animate-spin-slow" />
            </div>
            <h1 className="text-4xl font-bold text-center text-teal-800">We’re Here for You</h1>
            <p className="max-w-lg text-lg leading-relaxed text-center text-gray-600">
              We’re here to help you feel better. Just tell us how you’re doing, and we’ll create a plan just for you. It’s simple and personal!
            </p>
            <Button
              onClick={handleStartRecording}
              className="inline-flex items-center justify-center px-8 py-4 text-white transition-all transform bg-teal-600 rounded-full shadow-lg hover:bg-teal-700 hover:scale-105"
              aria-label="Start sharing how you feel"
            >
              <Mic className="w-6 h-6 mr-2" />
              Let’s Begin
            </Button>
            {!((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition) && (
              <div className="w-full max-w-md">
                <textarea
                  value={transcript}
                  onChange={(e) => setTranscript(e.target.value)}
                  placeholder="Type how you’re feeling here..."
                  className="w-full p-2 border rounded-lg"
                  aria-label="Type your symptoms"
                />
                <Button
                  onClick={() => {
                    setStep('review');
                    addMessage({ sender: 'patient', text: transcript, timestamp: new Date() });
                  }}
                  className="px-4 py-2 mt-2 text-white bg-teal-600 rounded-lg"
                  aria-label="Submit typed symptoms"
                >
                  Submit
                </Button>
              </div>
            )}
          </div>
        )}

        {(step === 'recording' || step === 'review') && loading && (
          <div className="flex flex-col items-center justify-center space-y-4 animate-fade-in">
            <Loader2 className="w-12 h-12 text-teal-600 animate-spin" />
            <p className="text-lg text-gray-700">I’m listening carefully to what you’re saying...</p>
          </div>
        )}

        {step === 'final' && !loading && guidelines && (
          <div className="w-full p-8 bg-white shadow-xl rounded-2xl animate-slide-up">
            <h2 className="mb-6 text-3xl font-bold text-center text-teal-700">Your Personalized Care Plan</h2>
            <p className="text-lg leading-relaxed text-gray-700">{guidelines}</p>
            {audioSrc && (
              <audio controls className="w-full mt-4">
                <source src={audioSrc} type="audio/mp3" />
                Your browser does not support the audio element.
              </audio>
            )}
          </div>
        )}

        {step !== 'initial' && (
          <div className="grid w-full grid-cols-1 gap-8 mt-10 lg:grid-cols-2">
            <div className="bg-white p-6 rounded-2xl shadow-lg h-[28rem] overflow-auto transition-all hover:shadow-xl">
              <h2 className="mb-4 text-2xl font-semibold text-teal-700">Our Conversation</h2>
              <div className="space-y-4">
                {conversation.map((msg, idx) => (
                  <div key={idx} className={`flex items-start ${msg.sender === 'doctor' ? 'justify-start' : 'justify-end'} animate-slide-in`}>
                    {msg.sender === 'doctor' ? (
                      <User className="flex-shrink-0 w-6 h-6 mr-3 text-teal-600" />
                    ) : (
                      <Mic className="flex-shrink-0 w-6 h-6 mr-3 text-indigo-600" />
                    )}
                    <div
                      className={`px-4 py-3 rounded-xl max-w-xs shadow-md ${
                        msg.sender === 'doctor' ? 'bg-teal-50 text-gray-800' : 'bg-indigo-50 text-gray-800'
                      }`}
                    >
                      <p className="text-sm leading-relaxed">{msg.text}</p>
                      <p className="mt-1 text-xs text-gray-500">{msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</p>
                    </div>
                  </div>
                ))}
                <div ref={conversationEndRef} />
              </div>
            </div>

            {(staticFollowUpQuestions.length > 0 || dynamicFollowUpQuestions.length > 0) && (
              <div className="bg-white p-6 rounded-2xl shadow-lg h-[28rem] overflow-auto transition-all hover:shadow-xl">
                <h2 className="mb-4 text-2xl font-semibold text-teal-700">What You’ve Shared</h2>
                {staticFollowUpQuestions.length > 0 && (
                  <div className="mb-6">
                    <h3 className="mb-2 text-lg font-medium text-gray-700">Initial Questions</h3>
                    <ul className="space-y-3">
                      {staticFollowUpQuestions.map((q, i) => (
                        <li key={`static-${i}`} className="p-3 rounded-lg bg-teal-50">
                          <span className="block font-medium text-teal-800">{q}</span>
                          {staticFollowUpAnswers[i] && <p className="mt-1 text-gray-600">You mentioned: {staticFollowUpAnswers[i]}</p>}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {dynamicFollowUpQuestions.length > 0 && (
                  <div>
                    <h3 className="mb-2 text-lg font-medium text-gray-700">Additional Details</h3>
                    <ul className="space-y-3">
                      {dynamicFollowUpQuestions.map((q, i) => (
                        <li key={`dynamic-${i}`} className="p-3 rounded-lg bg-indigo-50">
                          <span className="block font-medium text-indigo-800">{q}</span>
                          {dynamicFollowUpAnswers[i] && <p className="mt-1 text-gray-600">You mentioned: {dynamicFollowUpAnswers[i]}</p>}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {micActive && currentPrompt && (
          <div className="fixed max-w-md p-4 text-center text-gray-800 transform -translate-x-1/2 bg-teal-100 rounded-lg shadow-lg top-20 left-1/2">
            <p>{currentPrompt}</p>
          </div>
        )}
      </main>

      <footer className="w-full py-6 text-center text-teal-100 bg-teal-700">
        <p className="text-sm">You’re our priority. This isn’t a replacement for seeing a doctor in person.</p>
      </footer>

      {micActive && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
          <div className="flex items-center justify-center w-40 h-40 bg-teal-100 rounded-full shadow-2xl animate-pulse">
            <Mic className="w-20 h-20 text-teal-600" />
          </div>
        </div>
      )}

      {isManuallyPaused && (
        <div
          className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-gradient-to-b from-teal-900/80 to-indigo-900/80 animate-fade-in"
          onClick={() => {
            setIsManuallyPaused(false);
            speak("Let’s continue where we left off.");
          }}
        >
          <h1 className="mb-4 text-4xl font-bold text-white">Paused</h1>
          <p className="text-lg text-teal-100">Tap anywhere to resume when you’re ready.</p>
        </div>
      )}

      {loading && (step === 'followupLoading' || step === 'dynamicFollowupLoading' || step === 'final') && (
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-gradient-to-b from-teal-900/70 to-indigo-900/70">
          <Loader2 className="w-16 h-16 text-teal-300 animate-spin" />
          <p className="mt-6 text-xl font-semibold text-white">{getLoadingMessage()}</p>
        </div>
      )}

      {step !== 'initial' && (
        <Button
          onClick={() => {
            if (isManuallyPaused) {
              setIsManuallyPaused(false);
              speak("Let’s continue where we left off.");
            } else {
              setIsManuallyPaused(true);
              speak("Pausing now. Tap 'Resume' whenever you’re ready.");
            }
          }}
          className="fixed px-4 py-2 text-white transition-all bg-teal-600 rounded-full shadow-lg bottom-4 left-4 hover:bg-teal-700"
          aria-label={isManuallyPaused ? "Resume the process" : "Pause the process"}
        >
          {isManuallyPaused ? 'Resume' : 'Pause'}
        </Button>
      )}

      {step !== 'initial' && (
        <Button
          onClick={handleRestart}
          className="fixed px-4 py-2 text-white transition-all bg-teal-600 rounded-full shadow-lg bottom-4 right-4 hover:bg-teal-700"
          aria-label="Restart the health journey"
        >
          Restart
        </Button>
      )}
    </div>
  );
};

export default App;
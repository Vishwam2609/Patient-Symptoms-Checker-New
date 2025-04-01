import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Button } from './components/ui/button';
import { Mic, Loader2, User, Download } from 'lucide-react';

const API_BASE_URL = 'https://a4c9-2405-201-2032-c081-6107-4aad-81ec-9057.ngrok-free.app';

type AppStep =
  | 'initial'
  | 'recording'
  | 'review'
  | 'followupLoading'
  | 'followup'
  | 'dynamicFollowupLoading'
  | 'dynamicFollowup'
  | 'final';

interface Message {
  sender: 'doctor' | 'patient';
  text: string;
  timestamp: Date;
}

/* ----------- Voice Helpers ----------- */
const startSpeechRecognition = (
  onResult: (transcript: string) => void,
  onError: (err: any) => void,
  toggleMic?: (active: boolean) => void
) => {
  const SpeechRecognition =
    (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  if (!SpeechRecognition) {
    onError("Speech recognition not supported in this browser.");
    return null;
  }
  const recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    if (toggleMic) toggleMic(true);
  };

  recognition.onresult = (event: any) => {
    if (toggleMic) toggleMic(false);
    const transcript = event.results[0][0].transcript;
    onResult(transcript);
  };

  recognition.onerror = (event: any) => {
    if (toggleMic) toggleMic(false);
    onError(event.error);
  };

  recognition.start();
  return recognition;
};

const speakAndListen = (
  message: string,
  onResult: (result: string) => void,
  addMessage: (msg: Message) => void,
  toggleMic: (active: boolean) => void
) => {
  const utterance = new SpeechSynthesisUtterance(message);
  utterance.rate = 0.9;
  utterance.pitch = 1.0;
  utterance.onend = () => {
    startSpeechRecognition(
      (result) => {
        addMessage({ sender: 'patient', text: result, timestamp: new Date() });
        onResult(result);
      },
      (error) => {
        onResult("");
      },
      toggleMic
    );
  };
  speechSynthesis.speak(utterance);
  addMessage({ sender: 'doctor', text: message, timestamp: new Date() });
};

/* ----------- Voice Input with Retry & Pause/Resume ----------- */
const useVoiceInput = (
  addMessage: (msg: Message) => void,
  toggleMic: (active: boolean) => void,
  onResume: () => void
) => {
  const [paused, setPaused] = useState<boolean>(false);
  const resumeRef = useRef<() => void>(() => {});

  const voiceInput = useCallback(
    async (message: string): Promise<string> => {
      let attempts = 0;
      const TIMEOUT_MS = 15000;
      while (attempts < 3) {
        try {
          const result: string = await new Promise((resolve, reject) => {
            const timeoutId = setTimeout(() => {
              reject(new Error("No input"));
            }, TIMEOUT_MS);
            speakAndListen(
              message,
              (result) => {
                clearTimeout(timeoutId);
                result ? resolve(result) : reject(new Error("No input"));
              },
              addMessage,
              toggleMic
            );
          });
          return result;
        } catch (err) {
          attempts++;
          if (attempts < 3) {
            const retryMsg = `I didn't catch that. Could you please repeat? (Attempt ${attempts} of 3)`;
            const utterance = new SpeechSynthesisUtterance(retryMsg);
            utterance.rate = 0.9;
            utterance.pitch = 1.0;
            speechSynthesis.speak(utterance);
            addMessage({ sender: 'doctor', text: retryMsg, timestamp: new Date() });
          } else {
            const pauseMsg = "It seems you're having trouble responding. Please click to resume when you're ready.";
            const utterance = new SpeechSynthesisUtterance(pauseMsg);
            utterance.rate = 0.9;
            utterance.pitch = 1.0;
            speechSynthesis.speak(utterance);
            addMessage({ sender: 'doctor', text: pauseMsg, timestamp: new Date() });
            setPaused(true);
            await new Promise<void>((resolve) => {
              resumeRef.current = resolve;
            });
            setPaused(false);
            attempts = 0;
          }
        }
      }
      return "";
    },
    [addMessage, toggleMic]
  );

  const resume = () => {
    resumeRef.current();
  };

  return { voiceInput, paused, resume };
};

/* ----------- API Functions ----------- */
async function extractSymptoms(transcript: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/extract_symptoms`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transcript }),
  });
  if (!response.ok) {
    throw new Error("Symptom extraction failed");
  }
  const data = await response.json();
  return data.key_symptom;
}

async function generateFollowupQuestions(
  reviewed_transcript: string,
  key_symptom: string,
  static_followup: { question: string; answer: string }[]
): Promise<string[]> {
  const response = await fetch(`${API_BASE_URL}/generate_followup_questions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reviewed_transcript, key_symptom, static_followup }),
  });
  if (!response.ok) {
    throw new Error("Dynamic follow-up question generation failed");
  }
  const data = await response.json();
  return data.follow_up_questions;
}

async function generateGuidelines(
  transcript: string,
  key_symptom: string,
  static_followup: { question: string; answer: string }[],
  dynamic_followup: { question: string; answer: string }[]
): Promise<{ guidelines: string; audio: string }> {
  const response = await fetch(`${API_BASE_URL}/generate_guidelines`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transcript, key_symptom, follow_up: [...static_followup, ...dynamic_followup] }),
  });
  if (!response.ok) {
    throw new Error("Guideline generation failed");
  }
  const data = await response.json();
  return { guidelines: data.guidelines, audio: data.audio };
}

/* ----------- Static Follow-up Questions Mapping ----------- */
const staticFollowUpQuestionsMapping: { [key: string]: string[] } = {
  fever: [
    "When did you first notice your fever, and has your temperature been consistently high or fluctuating?",
    "Are you experiencing any other symptoms like chills, sweating, or body aches?",
    "Have you had any recent exposures such as travel or contact with someone ill?"
  ],
  coughing: [
    "When did your cough start, and is it persistent or intermittent?",
    "Is your cough dry, or are you producing any phlegm?",
    "Do you have any other breathing difficulties such as shortness of breath or chest tightness?"
  ],
  headache: [
    "When did your headache begin, and how would you describe its intensity and duration?",
    "Is the headache concentrated in one area or more generalized?",
    "Are you experiencing nausea, sensitivity to light or sound, or any visual disturbances?"
  ],
  backpain: [
    "When did your back pain start, and is it in a specific area such as your lower back?",
    "Does the pain get worse with movement or remain constant?",
    "Have you engaged in any strenuous activities recently that might have strained your back?"
  ],
  toothache: [
    "When did your toothache begin, and is the pain constant or does it come and go?",
    "How would you describe the pain, and does it spread to nearby areas?",
    "Have you noticed any other dental issues like gum swelling or increased sensitivity?"
  ]
};

/* ----------- Main App Component ----------- */
const App: React.FC = () => {
  const [step, setStep] = useState<AppStep>('initial');
  const [transcript, setTranscript] = useState<string>('');
  const [extractedSymptom, setExtractedSymptom] = useState<string>('');
  const [staticFollowUpQuestions, setStaticFollowUpQuestions] = useState<string[]>([]);
  const [staticFollowUpAnswers, setStaticFollowUpAnswers] = useState<string[]>([]);
  const [dynamicFollowUpQuestions, setDynamicFollowUpQuestions] = useState<string[]>([]);
  const [dynamicFollowUpAnswers, setDynamicFollowUpAnswers] = useState<string[]>([]);
  const [currentStaticIndex, setCurrentStaticIndex] = useState<number>(0);
  const [currentDynamicIndex, setCurrentDynamicIndex] = useState<number>(0);
  const [guidelines, setGuidelines] = useState<string>('');
  const [audioUrl, setAudioUrl] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [overlayVisible, setOverlayVisible] = useState<boolean>(true);
  const [welcomeSpoken, setWelcomeSpoken] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string>('');
  const [conversation, setConversation] = useState<Message[]>([]);
  const [micActive, setMicActive] = useState<boolean>(false);
  const [waitingIntervalId, setWaitingIntervalId] = useState<NodeJS.Timeout | null>(null);
  const [paused, setPaused] = useState<boolean>(false);
  const resumeRef = useRef<() => void>(() => {});
  const conversationEndRef = useRef<HTMLDivElement>(null);

  const { voiceInput, paused: voicePaused, resume } = useVoiceInput(
    (msg: Message) => addMessage(msg),
    setMicActive,
    () => resume()
  );

  const addMessage = (msg: Message) => {
    setConversation((prev) => [...prev, msg]);
  };

  // Auto-scroll conversation panel
  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation]);

  // Every 20 seconds while loading, speak a longer friendly wait message.
  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null;
    if (loading) {
      intervalId = setInterval(() => {
        const waitMsg =
          "Please wait a moment. We are carefully processing your information to provide the best guidance for your health. Thank you for your patience.";
        const utterance = new SpeechSynthesisUtterance(waitMsg);
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        speechSynthesis.speak(utterance);
      }, 20000);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [loading]);

  // Helper to get the appropriate loading message text for overlay (displayed larger).
  const getLoadingMessage = (): string => {
    if (step === 'followupLoading') return "Preparing follow-up questions...";
    if (step === 'dynamicFollowupLoading') return "Generating additional follow-up questions...";
    if (step === 'final' && !guidelines) return "Generating home care plan...";
    return "";
  };

  /* ----------- Header & Progress Bar ----------- */
  const Header = () => (
    <header className="relative w-full px-6 py-4 overflow-hidden shadow-lg bg-gradient-to-r from-blue-600 to-blue-800">
      <svg
        className="absolute bottom-0 left-0 right-0 text-white transition-all opacity-40"
        style={{ transform: 'translateY(50%)' }}
        viewBox="0 0 1440 320"
        preserveAspectRatio="none"
      >
        <path
          fill="currentColor"
          fillOpacity="1"
          d="M0,64L40,69.3C80,75,160,85,240,117.3C320,149,400,203,480,224C560,245,640,235,720,208C800,181,880,139,960,133.3C1040,128,1120,160,1200,165.3C1280,171,1360,149,1400,138.7L1440,128L1440,320L1400,320C1360,320,1280,320,1200,320C1120,320,1040,320,960,320C880,320,800,320,720,320C640,320,560,320,480,320C400,320,320,320,240,320C160,320,80,320,40,320L0,320Z"
        />
      </svg>
      <h1 className="relative z-10 text-3xl font-extrabold text-white drop-shadow-lg">
        Your Personalized Telehealth Experience
      </h1>
      {/* Progress Bar */}
      <div className="relative z-10 mt-4">
        <div className="w-full h-2 bg-gray-200 rounded-full">
          <div
            className="h-2 transition-all duration-500 bg-blue-400 rounded-full"
            style={{
              width:
                step === 'initial'
                  ? '0%'
                  : step === 'recording' || step === 'review'
                  ? '33%'
                  : step === 'followup' || step === 'followupLoading' || step === 'dynamicFollowup' || step === 'dynamicFollowupLoading'
                  ? '66%'
                  : '100%'
            }}
          />
        </div>
      </div>
    </header>
  );

  /* ----------- Overlay & Welcome ----------- */
  const handleOverlayClick = () => {
    if (!welcomeSpoken) {
      speakWelcome();
    }
    setOverlayVisible(false);
  };

  const speakWelcome = () => {
    const welcomeText =
      "Welcome! We're here to help you. Begin by describing your symptoms and our virtual assistant will guide you to a personalized home care plan. Then, click on 'Start Recording' to begin.";
    const utterance = new SpeechSynthesisUtterance(welcomeText);
    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    speechSynthesis.speak(utterance);
    addMessage({ sender: 'doctor', text: welcomeText, timestamp: new Date() });
    setWelcomeSpoken(true);
  };

  const handleStartRecording = async () => {
    setStep('recording');
    try {
      const result = await voiceInput("Please describe your symptoms after the beep.");
      setTranscript(result);
      addMessage({ sender: 'doctor', text: "Your voice has been recorded.", timestamp: new Date() });
      setStep('review');
    } catch (err) {
      // Handled in voiceInput
    }
  };

  /* ----------- Review Step ----------- */
  useEffect(() => {
    if (step === 'review') {
      (async () => {
        try {
          const reviewMsg = `You said: ${transcript}. If this is correct, say "yes", otherwise say "no".`;
          const result = await voiceInput(reviewMsg);
          if (result.toLowerCase().includes("yes")) {
            setStep('followupLoading');
          } else {
            setStep('recording');
            const retryResult = await voiceInput("Let's try again. Please describe your symptoms after the beep.");
            setTranscript(retryResult);
            setStep('review');
          }
        } catch (err) {
          // Handled in voiceInput
        }
      })();
    }
  }, [step, transcript]);

  /* ----------- Follow-Up Loading ----------- */
  useEffect(() => {
    if (step === 'followupLoading') {
      speechSynthesis.cancel();
      const promptMsg = "Please wait while we prepare your follow-up questions.";
      const utterance = new SpeechSynthesisUtterance(promptMsg);
      utterance.rate = 0.9;
      utterance.pitch = 1.0;
      speechSynthesis.speak(utterance);
      addMessage({ sender: 'doctor', text: promptMsg, timestamp: new Date() });
      setLoading(true);
      (async () => {
        try {
          const symptom = await extractSymptoms(transcript);
          const cleaned = symptom.toLowerCase().trim().replace(/[^a-z]/g, "");
          setExtractedSymptom(cleaned);
          const mappingKey = Object.keys(staticFollowUpQuestionsMapping).find(
            key => cleaned === key || cleaned.includes(key) || key.includes(cleaned)
          );
          if (mappingKey) {
            const questions = staticFollowUpQuestionsMapping[mappingKey];
            setStaticFollowUpQuestions(questions);
            setStaticFollowUpAnswers(Array(questions.length).fill(""));
            setCurrentStaticIndex(0);
            setStep('followup');
          } else {
            setStep('dynamicFollowupLoading');
          }
        } catch (error) {
          const errMsg = "We had trouble understanding your symptoms. Please try again.";
          setErrorMsg(errMsg);
          speechSynthesis.speak(new SpeechSynthesisUtterance(errMsg));
          setStep('initial');
        }
        setLoading(false);
      })();
    }
  }, [step, transcript]);

  /* ----------- Static Follow-Up ----------- */
  useEffect(() => {
    if (step === 'followup' && staticFollowUpQuestions.length > 0) {
      (async () => {
        try {
          const question = staticFollowUpQuestions[currentStaticIndex];
          const result = await voiceInput(`Dear patient, ${question}`);
          const newAnswers = [...staticFollowUpAnswers];
          newAnswers[currentStaticIndex] = result;
          setStaticFollowUpAnswers(newAnswers);
          if (currentStaticIndex + 1 < staticFollowUpQuestions.length) {
            setCurrentStaticIndex(currentStaticIndex + 1);
          } else {
            setStep('dynamicFollowupLoading');
          }
        } catch (err) {
          // Handled via voiceInput retry logic.
        }
      })();
    }
  }, [step, currentStaticIndex, staticFollowUpQuestions, staticFollowUpAnswers]);

  /* ----------- Dynamic Follow-Up ----------- */
  useEffect(() => {
    if (step === 'dynamicFollowupLoading') {
      speechSynthesis.cancel();
      const dynLoadingMsg = "Dear patient, please wait while we generate additional follow-up questions.";
      const utterance = new SpeechSynthesisUtterance(dynLoadingMsg);
      utterance.rate = 0.9;
      utterance.pitch = 1.0;
      speechSynthesis.speak(utterance);
      addMessage({ sender: 'doctor', text: dynLoadingMsg, timestamp: new Date() });
      setLoading(true);
      (async () => {
        try {
          const staticQA = staticFollowUpQuestions.map((q, i) => ({ question: q, answer: staticFollowUpAnswers[i] }));
          const questions = await generateFollowupQuestions(transcript, extractedSymptom, staticQA);
          setDynamicFollowUpQuestions(questions);
          setDynamicFollowUpAnswers(Array(questions.length).fill(""));
          setCurrentDynamicIndex(0);
          setStep('dynamicFollowup');
        } catch (error) {
          const errMsg = "We couldn't generate additional follow-up questions. Moving on.";
          speechSynthesis.speak(new SpeechSynthesisUtterance(errMsg));
          addMessage({ sender: 'doctor', text: errMsg, timestamp: new Date() });
          setStep('final');
        }
        setLoading(false);
      })();
    } else if (step === 'dynamicFollowup' && dynamicFollowUpQuestions.length > 0) {
      (async () => {
        try {
          const question = dynamicFollowUpQuestions[currentDynamicIndex];
          const result = await voiceInput(`Dear patient, ${question}`);
          const newDynAnswers = [...dynamicFollowUpAnswers];
          newDynAnswers[currentDynamicIndex] = result;
          setDynamicFollowUpAnswers(newDynAnswers);
          if (currentDynamicIndex + 1 < dynamicFollowUpQuestions.length) {
            setCurrentDynamicIndex(currentDynamicIndex + 1);
          } else {
            setStep('final');
          }
        } catch (err) {
          // Handled via voiceInput retry logic.
        }
      })();
    }
  }, [
    step,
    dynamicFollowUpQuestions,
    dynamicFollowUpAnswers,
    currentDynamicIndex,
    transcript,
    extractedSymptom,
    staticFollowUpQuestions,
    staticFollowUpAnswers
  ]);

  /* ----------- Final Guidelines ----------- */
  useEffect(() => {
    if (step === 'final') {
      setLoading(true);
      // Speak an initial final message.
      const finalMsg = "Dear patient, please wait while we generate your home care plan.";
      const finalUtterance = new SpeechSynthesisUtterance(finalMsg);
      finalUtterance.rate = 0.9;
      finalUtterance.pitch = 1.0;
      speechSynthesis.speak(finalUtterance);
      addMessage({ sender: 'doctor', text: finalMsg, timestamp: new Date() });
      
      const staticQA = staticFollowUpQuestions.map((q, i) => ({ question: q, answer: staticFollowUpAnswers[i] }));
      const dynamicQA = dynamicFollowUpQuestions.map((q, i) => ({ question: q, answer: dynamicFollowUpAnswers[i] }));
      (async () => {
        try {
          const result = await generateGuidelines(transcript, extractedSymptom, staticQA, dynamicQA);
          setGuidelines(result.guidelines);
          const guidelinesUtterance = new SpeechSynthesisUtterance(result.guidelines);
          guidelinesUtterance.rate = 0.9;
          guidelinesUtterance.pitch = 1.0;
          speechSynthesis.speak(guidelinesUtterance);
          addMessage({ sender: 'doctor', text: result.guidelines, timestamp: new Date() });
          setAudioUrl(result.audio);
        } catch (error) {
          const errMsg = "Sorry, we had trouble generating your home care plan. Please try again later.";
          speechSynthesis.speak(new SpeechSynthesisUtterance(errMsg));
          addMessage({ sender: 'doctor', text: errMsg, timestamp: new Date() });
        }
        setLoading(false);
      })();
    }
  }, [
    step,
    transcript,
    extractedSymptom,
    staticFollowUpQuestions,
    staticFollowUpAnswers,
    dynamicFollowUpQuestions,
    dynamicFollowUpAnswers
  ]);

  /* ----------- Step Indicator ----------- */
  const getStepIndicator = () => {
    switch (step) {
      case 'initial':
        return '';
      case 'recording':
      case 'review':
        return 'Step 1: Symptom Description';
      case 'followupLoading':
      case 'followup':
      case 'dynamicFollowupLoading':
      case 'dynamicFollowup':
        return 'Step 2: Follow-Up Questions';
      case 'final':
        return 'Step 3: Home Care Plan';
      default:
        return '';
    }
  };

  /* ----------- Download Plan (Simulated) ----------- */
  const handleDownloadPlan = () => {
    alert('Download Plan functionality is not yet implemented!');
  };

  return (
    <div className="relative min-h-screen transition-colors bg-gradient-to-br from-blue-50 to-gray-100">
      <Header />

      {/* Large central overlay mic icon based on mic status */}
      {micActive && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-30">
          <div className="flex items-center justify-center w-32 h-32 bg-blue-100 rounded-full shadow-2xl animate-pulse">
            <Mic className="w-16 h-16 text-blue-600" />
          </div>
        </div>
      )}

      {/* Loading overlay for intermediate steps */}
      {loading && (step === 'followupLoading' || step === 'dynamicFollowupLoading' || (step === 'final' && loading)) && (
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black bg-opacity-50">
          <Loader2 className="w-12 h-12 text-blue-600 animate-spin" />
          <p className="mt-4 text-2xl font-bold text-white">{getLoadingMessage()}</p>
        </div>
      )}

      {/* Pause overlay when waiting for resume */}
      {paused && (
        <div
          className="fixed inset-0 z-50 flex flex-col items-center justify-center transition-all bg-black bg-opacity-60 animate-fadeIn"
          onClick={() => {
            setPaused(false);
            resume();
          }}
        >
          <h1 className="text-4xl font-bold text-white">Click to Resume</h1>
          <p className="mt-2 text-lg text-gray-200">When you're ready, tap to continue.</p>
        </div>
      )}

      {overlayVisible && (
        <div
          className="absolute inset-0 z-50 flex flex-col items-center justify-center transition-all bg-black bg-opacity-60 animate-fadeIn"
          style={{ paddingTop: '30%' }}
          onClick={handleOverlayClick}
        >
          <div className="text-center">
            <h1 className="text-4xl font-bold text-white">Tap to Begin</h1>
            <p className="mt-2 text-lg text-gray-200">
              Please click on the "Start Recording" button to begin your consultation.
            </p>
          </div>
        </div>
      )}

      {errorMsg && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
          <div className="w-full max-w-sm p-6 text-center bg-white rounded-lg shadow-xl">
            <h2 className="mb-4 text-xl font-semibold">Alert</h2>
            <p className="mb-6">{errorMsg}</p>
            <Button
              onClick={() => setErrorMsg('')}
              className="px-6 py-3 text-white transition-colors bg-red-500 rounded-lg hover:bg-red-600"
            >
              Close
            </Button>
          </div>
        </div>
      )}

      <div className="container px-4 py-8 mx-auto transition-all">
        <div className="overflow-hidden bg-white shadow-2xl rounded-xl">
          <div className="px-8 pt-8 pb-4 border-b border-gray-200">
            <div className="text-sm font-semibold text-gray-600 transition-all">{getStepIndicator()}</div>
          </div>
          <div className="p-8">
            {step === 'initial' && (
              <div className="space-y-6 text-center">
                <div className="flex justify-center">
                  <div className="flex items-center justify-center w-24 h-24 bg-blue-100 rounded-full shadow-lg animate-bounce">
                    <Mic className="w-12 h-12 text-blue-600" />
                  </div>
                </div>
                <h1 className="text-3xl font-extrabold text-gray-900">
                  Welcome! We're Here to Help You.
                </h1>
                <p className="max-w-xl mx-auto text-lg text-gray-700">
                  Begin by describing your symptoms, and our virtual assistant will guide you to a personalized home care plan.
                </p>
                <Button
                  onClick={handleStartRecording}
                  className="inline-flex items-center justify-center px-8 py-3 mt-4 text-white transition-colors bg-blue-600 rounded-lg shadow-lg hover:bg-blue-700 focus:ring-4 focus:ring-blue-300"
                >
                  <Mic className="w-5 h-5 mr-2" />
                  Start Recording
                </Button>
                <p className="max-w-md mx-auto text-sm text-gray-500">
                  Your responses will be processed securely. We value your privacy and comfort.
                </p>
              </div>
            )}

            {(step === 'recording' || step === 'review') && loading && (
              <div className="flex flex-col items-center justify-center space-y-2">
                <Loader2 className="w-10 h-10 text-blue-600 animate-spin" />
                <p className="text-base text-gray-600">Processing your voice...</p>
              </div>
            )}

            {step === 'final' && !loading && guidelines && (
              <div className="space-y-6 text-center">
                <div className="p-6 rounded-lg shadow-lg bg-gray-50">
                  <h2 className="mb-3 text-xl font-bold text-gray-800">Your Home Care Plan</h2>
                  <pre className="leading-relaxed text-left text-gray-800 whitespace-pre-wrap">
                    {guidelines}
                  </pre>
                  <Button
                    onClick={handleDownloadPlan}
                    className="inline-flex items-center justify-center px-6 py-2 mt-4 text-white transition-colors bg-green-600 rounded-md shadow hover:bg-green-700"
                  >
                    <Download className="w-5 h-5 mr-2" />
                    Download Plan
                  </Button>
                </div>
                <p className="mt-2 text-sm text-gray-600">
                  These guidelines are informational. Please consult your healthcare provider for further advice.
                </p>
              </div>
            )}

            {step !== 'initial' && (
              <div className="grid grid-cols-1 gap-6 mt-8 lg:grid-cols-2" style={{ minHeight: '200px' }}>
                <div className="p-4 overflow-y-auto rounded-lg shadow-inner bg-gray-50 max-h-80">
                  <h2 className="mb-4 text-xl font-semibold text-gray-800">Conversation</h2>
                  <div className="space-y-3">
                    {conversation.map((msg, idx) => (
                      <div
                        key={idx}
                        className={`flex items-start transition-all animate-[fadeIn_0.3s_ease-in-out] ${msg.sender === 'doctor' ? 'justify-start' : 'justify-end'}`}
                      >
                        {msg.sender === 'doctor' ? (
                          <div className="flex items-center">
                            <User className="w-6 h-6 mr-2 text-blue-600" />
                          </div>
                        ) : (
                          <div className="flex items-center">
                            <Mic className="w-6 h-6 mr-2 text-green-600" />
                          </div>
                        )}
                        <div className={`px-4 py-2 rounded-lg max-w-xs ${msg.sender === 'doctor' ? 'bg-blue-100 text-gray-800' : 'bg-green-100 text-gray-800'}`}>
                          <p className="text-sm leading-relaxed">{msg.text}</p>
                          <p className="mt-1 text-xs text-gray-500">
                            {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </p>
                        </div>
                      </div>
                    ))}
                    <div ref={conversationEndRef} />
                  </div>
                </div>

                {(staticFollowUpQuestions.length > 0 || dynamicFollowUpQuestions.length > 0) && (
                  <div className="p-4 overflow-y-auto bg-gray-100 rounded-lg shadow-inner max-h-80">
                    <h2 className="mb-4 text-xl font-semibold text-gray-800">Follow-Up Questions & Answers</h2>
                    {staticFollowUpQuestions.length > 0 && (
                      <div className="mb-4">
                        <h3 className="text-lg font-medium text-gray-700">Static Follow-Up:</h3>
                        <ul className="pl-5 mt-2 list-disc">
                          {staticFollowUpQuestions.map((q, i) => (
                            <li key={`static-${i}`} className="mb-1">
                              <span className="font-medium text-gray-800">{q}</span>
                              {staticFollowUpAnswers[i] && (
                                <p className="ml-4 text-gray-600">Answer: {staticFollowUpAnswers[i]}</p>
                              )}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {dynamicFollowUpQuestions.length > 0 && (
                      <div>
                        <h3 className="text-lg font-medium text-gray-700">Dynamic Follow-Up:</h3>
                        <ul className="pl-5 mt-2 list-disc">
                          {dynamicFollowUpQuestions.map((q, i) => (
                            <li key={`dynamic-${i}`} className="mb-1">
                              <span className="font-medium text-gray-800">{q}</span>
                              {dynamicFollowUpAnswers[i] && (
                                <p className="ml-4 text-gray-600">Answer: {dynamicFollowUpAnswers[i]}</p>
                              )}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      <footer className="py-4 mt-6 text-sm text-center text-gray-500 bg-gray-200">
        We care about you. Your well-being is our highest priority. This is not a substitute for professional medical advice.
      </footer>
    </div>
  );
};

export default App;
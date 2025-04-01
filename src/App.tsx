import React, { useState, useEffect, useRef } from 'react';
import { Button } from './components/ui/button';
import { Mic, Loader2, CheckCircle, User } from 'lucide-react';

const API_BASE_URL = 'https://3e7e-2405-201-2032-c081-6107-4aad-81ec-9057.ngrok-free.app';

// Static follow-up questions mapping.
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
}

/* ----------- Voice Helpers ----------- */
// Start speech recognition and call onResult with transcript.
const startSpeechRecognition = (
  onResult: (transcript: string) => void,
  onError: (err: any) => void
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
  recognition.start();

  recognition.onresult = (event: any) => {
    const transcript = event.results[0][0].transcript;
    onResult(transcript);
  };

  recognition.onerror = (event: any) => {
    onError(event.error);
  };

  return recognition;
};

// Speak a message using TTS, log it, then listen for a response.
const speakAndListen = (
  message: string,
  onResult: (result: string) => void,
  addMessage: (msg: Message) => void
) => {
  const utterance = new SpeechSynthesisUtterance(message);
  utterance.rate = 0.9;
  utterance.pitch = 1.0;
  utterance.onend = () => {
    startSpeechRecognition(
      (result) => {
        addMessage({ sender: 'patient', text: result });
        onResult(result);
      },
      (error) => {
        const errorMsg = "I'm sorry, I didn't catch that. Could you please repeat?";
        const errorUtterance = new SpeechSynthesisUtterance(errorMsg);
        errorUtterance.rate = 0.9;
        errorUtterance.pitch = 1.0;
        speechSynthesis.speak(errorUtterance);
        addMessage({ sender: 'doctor', text: errorMsg });
        // Retry the same prompt.
        speakAndListen(message, onResult, addMessage);
      }
    );
  };
  speechSynthesis.speak(utterance);
  addMessage({ sender: 'doctor', text: message });
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
  const conversationEndRef = useRef<HTMLDivElement>(null);

  const addMessage = (msg: Message) => {
    setConversation((prev) => [...prev, msg]);
  };

  // Auto-scroll conversation panel to the bottom when new messages arrive.
  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation]);

  /* ----------- Header & Overlay ----------- */
  const Header = () => (
    <header className="w-full px-6 py-4 shadow-lg bg-gradient-to-r from-blue-600 to-blue-800">
      <h1 className="text-2xl font-bold text-white">Your Personalized Telehealth Experience</h1>
    </header>
  );

  const speakWelcome = () => {
    const welcomeUtterance = new SpeechSynthesisUtterance(
      "Welcome, dear patient! We're here to help you feel your best. Describe your symptoms and let us guide you to a personalized home care plan. Then, click on Start Recording to begin."
    );
    welcomeUtterance.rate = 0.9;
    welcomeUtterance.pitch = 1.0;
    speechSynthesis.speak(welcomeUtterance);
    addMessage({ 
      sender: 'doctor', 
      text: "Welcome, dear patient! We're here to help you feel your best. Describe your symptoms and let us guide you to a personalized home care plan. Then, click on 'Start Recording' to begin."
    });
    setWelcomeSpoken(true);
  };

  const handleOverlayClick = () => {
    if (!welcomeSpoken) {
      speakWelcome();
    }
    setOverlayVisible(false);
  };

  // Start recording (voice capture) on click.
  const handleStartRecording = () => {
    setMicActive(true);
    speechSynthesis.cancel();
    setStep('recording');
    speakAndListen(
      "Please describe your symptoms after the beep.", 
      (result) => {
        setTranscript(result);
        setMicActive(false);
        addMessage({ sender: 'doctor', text: "Your voice has been recorded." });
        setStep('review');
      }, 
      addMessage
    );
  };

  /* ----------- Review Screen (Voice Confirmation) ----------- */
  useEffect(() => {
    if (step === 'review') {
      const reviewMsg = `You said: ${transcript}. If this is correct, say "yes", otherwise say "no".`;
      speakAndListen(reviewMsg, (result) => {
        if (result.toLowerCase().includes("yes")) {
          setStep('followupLoading');
        } else {
          setStep('recording');
          speakAndListen("Let's try again. Please describe your symptoms after the beep.", (newResult) => {
            setTranscript(newResult);
            setStep('review');
          }, addMessage);
        }
      }, addMessage);
    }
  }, [step, transcript]);

  /* ----------- Extracting Symptom & Preparing Follow-Up ----------- */
  useEffect(() => {
    if (step === 'followupLoading') {
      speechSynthesis.cancel();
      const promptMsg = "Please wait while we prepare your follow-up questions.";
      const utterance = new SpeechSynthesisUtterance(promptMsg);
      utterance.rate = 0.9;
      utterance.pitch = 1.0;
      speechSynthesis.speak(utterance);
      addMessage({ sender: 'doctor', text: promptMsg });
      setLoading(true);
      (async () => {
        try {
          const symptom = await extractSymptoms(transcript);
          const cleaned = symptom.toLowerCase().trim().replace(/[^a-z]/g, "");
          setExtractedSymptom(cleaned);
          // Flexible matching: check if the cleaned symptom or a substring matches any mapping key.
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

  /* ----------- Static Follow-Up (Voice-based) ----------- */
  useEffect(() => {
    if (step === 'followup' && staticFollowUpQuestions.length > 0) {
      const question = staticFollowUpQuestions[currentStaticIndex];
      speakAndListen(`Dear patient, ${question}`, (result) => {
        const newAnswers = [...staticFollowUpAnswers];
        newAnswers[currentStaticIndex] = result;
        setStaticFollowUpAnswers(newAnswers);
        if (currentStaticIndex + 1 < staticFollowUpQuestions.length) {
          setCurrentStaticIndex(currentStaticIndex + 1);
        } else {
          setStep('dynamicFollowupLoading');
        }
      }, addMessage);
    }
  }, [step, currentStaticIndex, staticFollowUpQuestions, staticFollowUpAnswers]);

  /* ----------- Dynamic Follow-Up (Voice-based) ----------- */
  useEffect(() => {
    if (step === 'dynamicFollowupLoading') {
      speechSynthesis.cancel();
      const dynLoadingMsg = "Dear patient, please wait while we generate additional follow-up questions.";
      speechSynthesis.speak(new SpeechSynthesisUtterance(dynLoadingMsg));
      addMessage({ sender: 'doctor', text: dynLoadingMsg });
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
          addMessage({ sender: 'doctor', text: errMsg });
          setStep('final');
        }
        setLoading(false);
      })();
    } else if (step === 'dynamicFollowup' && dynamicFollowUpQuestions.length > 0) {
      const question = dynamicFollowUpQuestions[currentDynamicIndex];
      speakAndListen(`Dear patient, ${question}`, (result) => {
        const newDynAnswers = [...dynamicFollowUpAnswers];
        newDynAnswers[currentDynamicIndex] = result;
        setDynamicFollowUpAnswers(newDynAnswers);
        if (currentDynamicIndex + 1 < dynamicFollowUpQuestions.length) {
          setCurrentDynamicIndex(currentDynamicIndex + 1);
        } else {
          setStep('final');
        }
      }, addMessage);
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

  /* ----------- Final Guidelines Generation (Voice-based) ----------- */
  useEffect(() => {
    if (step === 'final') {
      setLoading(true);
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
          addMessage({ sender: 'doctor', text: result.guidelines });
          setAudioUrl(result.audio);
        } catch (error) {
          const errMsg = "Sorry, we had trouble generating your home care plan. Please try again later.";
          speechSynthesis.speak(new SpeechSynthesisUtterance(errMsg));
          addMessage({ sender: 'doctor', text: errMsg });
        }
        setLoading(false);
      })();
    }
  }, [step, transcript, extractedSymptom, staticFollowUpQuestions, staticFollowUpAnswers, dynamicFollowUpQuestions, dynamicFollowUpAnswers]);

  /* ----------- Step Indicator for better UX ----------- */
  const getStepIndicator = () => {
    switch (step) {
      case 'initial':
        return 'Step 0: Start';
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-gray-100">
      <Header />
      {overlayVisible && (
        <div
          className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black cursor-pointer bg-opacity-60 animate-fadeIn"
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

      <div className="container px-4 py-8 mx-auto">
        <div className="overflow-hidden bg-white shadow-2xl rounded-xl">
          <div className="px-8 pt-8 pb-4 border-b border-gray-200">
            {/* Step Indicator */}
            <div className="text-sm font-semibold text-gray-600">
              {getStepIndicator()}
            </div>
          </div>
          <div className="p-8">
            {step === 'initial' && (
              <div className="space-y-6 text-center">
                <div className="flex justify-center">
                  <div className="flex items-center justify-center w-24 h-24 bg-blue-100 rounded-full shadow-lg">
                    <Mic className="w-12 h-12 text-blue-600" />
                  </div>
                </div>
                <h1 className="text-3xl font-extrabold text-gray-900">
                  Welcome, dear patient!
                </h1>
                <p className="text-lg text-gray-700">
                  We’re here to help you feel your best. Describe your symptoms and let us guide you to a personalized home care plan.
                </p>
                <Button
                  onClick={handleStartRecording}
                  className="px-8 py-3 mt-4 text-white transition-colors bg-blue-600 rounded-lg shadow-lg hover:bg-blue-700"
                >
                  <Mic className="w-5 h-5 mr-2" /> Start Recording
                </Button>
              </div>
            )}

            {(step === 'recording' || step === 'review') && (
              <div className="space-y-6 text-center">
                {step === 'recording' && (
                  <div className="flex flex-col items-center">
                    <div className="flex items-center">
                      <div className="w-4 h-4 mr-2 bg-red-500 rounded-full animate-pulse"></div>
                      <span className="text-base text-gray-700">Recording...</span>
                    </div>
                  </div>
                )}
                {loading ? (
                  <div className="flex flex-col items-center justify-center">
                    <Loader2 className="w-10 h-10 text-blue-600 animate-spin" />
                    <p className="text-base text-gray-600">Processing your voice...</p>
                  </div>
                ) : (
                  step === 'review' && (
                    <p className="text-lg text-gray-700">
                      Listening...
                    </p>
                  )
                )}
              </div>
            )}

            {(step === 'followupLoading' || step === 'dynamicFollowupLoading') && (
              <div className="flex flex-col items-center justify-center">
                <Loader2 className="w-10 h-10 text-blue-600 animate-spin" />
                <p className="mt-2 text-lg text-gray-700">
                  Preparing follow-up questions...
                </p>
              </div>
            )}

            {step === 'final' && (
              <div className="space-y-6 text-center">
                {loading ? (
                  <div className="flex flex-col items-center justify-center">
                    <Loader2 className="w-10 h-10 text-blue-600 animate-spin" />
                    <p className="mt-2 text-lg text-gray-700">
                      Generating your home care plan...
                    </p>
                  </div>
                ) : (
                  <div className="p-6 rounded-lg shadow-lg bg-gray-50">
                    <pre className="text-lg text-gray-800 whitespace-pre-wrap">{guidelines}</pre>
                  </div>
                )}
                <p className="mt-2 text-sm text-gray-600">
                  Dear patient, these guidelines are informational. Please consult your healthcare provider for further advice.
                </p>
              </div>
            )}

            {/* Conversation & Q&A Layout */}
            {step !== 'initial' && (
              <div className="grid grid-cols-1 gap-6 mt-8 lg:grid-cols-2">
                {/* Conversation Panel */}
                <div className="p-4 overflow-y-auto rounded-lg shadow-inner bg-gray-50 max-h-80">
                  <h2 className="mb-4 text-xl font-semibold text-gray-800">Conversation</h2>
                  <div className="space-y-3">
                    {conversation.map((msg, idx) => (
                      <div
                        key={idx}
                        className={`flex items-start ${msg.sender === 'doctor' ? 'justify-start' : 'justify-end'}`}
                      >
                        {/* Doctor's Icon on left, Patient's Icon on right */}
                        {msg.sender === 'doctor' && (
                          <User className="w-6 h-6 mr-2 text-blue-600" />
                        )}
                        {msg.sender === 'patient' && (
                          <Mic className="w-6 h-6 mr-2 text-green-600" />
                        )}
                        <div
                          className={`px-4 py-2 rounded-lg max-w-xs ${
                            msg.sender === 'doctor' ? 'bg-blue-100 text-gray-800' : 'bg-green-100 text-gray-800'
                          }`}
                        >
                          <p className="text-sm">{msg.text}</p>
                        </div>
                      </div>
                    ))}
                    <div ref={conversationEndRef} />
                  </div>
                </div>

                {/* Follow-up Q&A Panel */}
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
      <footer className="py-4 mt-6 text-sm text-center text-gray-500">
        We care about you. Your well-being is our highest priority.
      </footer>
    </div>
  );
};

export default App;
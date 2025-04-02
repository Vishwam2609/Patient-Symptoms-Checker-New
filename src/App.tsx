import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Button } from './components/ui/button';
import { Mic, Loader2, User } from 'lucide-react';

// Adjust the API_BASE_URL to your server
const API_BASE_URL = 'https://cce4-2405-201-2032-c081-5ac2-16de-dea6-4b6a.ngrok-free.app';

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

/** Speech Utility: Cancel any existing speech before speaking */
function speak(text: string) {
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.9; // Slightly slower for a natural pace
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
}

/** Initiates the browser's speech recognition */
const startSpeechRecognition = (
  onResult: (transcript: string) => void,
  onError: (err: any) => void,
  toggleMic?: (active: boolean) => void
) => {
  const SpeechRecognition =
    (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  if (!SpeechRecognition) {
    onError("Sorry, it looks like your browser doesn’t support speech recognition. Could you try another device?");
    return null;
  }
  const recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => toggleMic && toggleMic(true);
  recognition.onresult = (event: any) => {
    toggleMic && toggleMic(false);
    const transcript = event.results[0][0].transcript;
    onResult(transcript);
  };
  recognition.onerror = (event: any) => {
    toggleMic && toggleMic(false);
    onError(event.error);
  };

  recognition.start();
  return recognition;
};

/** Speaks a message, then listens for a response */
const speakAndListen = (
  message: string,
  onResult: (result: string) => void,
  addMessage: (msg: Message) => void,
  toggleMic: (active: boolean) => void
) => {
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(message);
  utterance.rate = 0.9;
  utterance.pitch = 1.0;
  utterance.onend = () =>
    startSpeechRecognition(
      (result) => {
        addMessage({ sender: 'patient', text: result, timestamp: new Date() });
        onResult(result);
      },
      () => onResult(""),
      toggleMic
    );
  window.speechSynthesis.speak(utterance);
  addMessage({ sender: 'doctor', text: message, timestamp: new Date() });
};

/** Custom hook for voice input with retries and pause/resume */
const useVoiceInput = (
  addMessage: (msg: Message) => void,
  toggleMic: (active: boolean) => void,
  onResume: () => void
) => {
  const [paused, setPaused] = useState<boolean>(false);
  const resumeRef = useRef<() => void>(() => {});

  const voiceInput = useCallback(
    async (promptMsg: string): Promise<string> => {
      let attempts = 0;
      const TIMEOUT_MS = 15000;

      while (attempts < 3) {
        try {
          const result: string = await new Promise((resolve, reject) => {
            const timeoutId = setTimeout(() => reject(new Error("No input")), TIMEOUT_MS);
            speakAndListen(
              promptMsg,
              (res) => {
                clearTimeout(timeoutId);
                res ? resolve(res) : reject(new Error("No input"));
              },
              addMessage,
              toggleMic
            );
          });
          return result;
        } catch (err) {
          attempts++;
          if (attempts < 3) {
            const retryMsg = `I’m sorry, I didn’t catch that. Could you say it again? This is try ${attempts + 1} of 3.`;
            speak(retryMsg);
            addMessage({ sender: 'doctor', text: retryMsg, timestamp: new Date() });
          } else {
            const pauseMsg = "It seems like we’re having trouble hearing you. Don’t worry, take your time, and tap the screen when you’re ready to continue.";
            speak(pauseMsg);
            addMessage({ sender: 'doctor', text: pauseMsg, timestamp: new Date() });
            setPaused(true);
            await new Promise<void>((resolve) => (resumeRef.current = resolve));
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
    onResume();
  };

  return { voiceInput, paused, resume };
};

/** API Functions */
async function extractSymptoms(transcript: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/extract_symptoms`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transcript }),
  });
  if (!response.ok) throw new Error("Symptom extraction failed");
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
  if (!response.ok) throw new Error("Dynamic follow-up question generation failed");
  const data = await response.json();
  return data.follow_up_questions;
}

/** Main App Component */
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
  const [loading, setLoading] = useState<boolean>(false);
  const [overlayVisible, setOverlayVisible] = useState<boolean>(true);
  const [welcomeSpoken, setWelcomeSpoken] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string>('');
  const [conversation, setConversation] = useState<Message[]>([]);
  const [micActive, setMicActive] = useState<boolean>(false);
  const isConfirmingRef = useRef<boolean>(false);
  const isAskingStaticRef = useRef<boolean>(false);
  const isAskingDynamicRef = useRef<boolean>(false);
  const isLoadingDynamicRef = useRef<boolean>(false);
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const waitingIntervalRef = useRef<number | null>(null);

  const { voiceInput, paused: voicePaused, resume } = useVoiceInput(
    (msg: Message) => addMessage(msg),
    setMicActive,
    () => resume()
  );

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation]);

  const addMessage = (msg: Message) => setConversation((prev) => [...prev, msg]);

  const getStepIndicator = (): string => {
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
  };

  const getProgressPercentage = (): string => {
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
  };

  const getLoadingMessage = (): string => {
    switch (step) {
      case 'followupLoading': return "We’re preparing some thoughtful questions to better understand how you’re doing.";
      case 'dynamicFollowupLoading': return "Just a moment—we’re tailoring a few more questions specifically for you.";
      case 'final': return "We’re putting together a care plan just for you to help you feel better soon.";
      default: return "We’re working on something special to support you...";
    }
  };

  /** Handle Waiting Message During Loading */
  useEffect(() => {
    if (loading && (step === 'followupLoading' || step === 'dynamicFollowupLoading' || step === 'final')) {
      waitingIntervalRef.current = window.setInterval(() => {
        const waitingMsg = "We’re still working on this for you. Hang in there, we’ll be with you shortly!";
        speak(waitingMsg);
        addMessage({ sender: 'doctor', text: waitingMsg, timestamp: new Date() });
      }, 20000); // Every 20 seconds
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

  /** Overlay & Welcome */
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
    addMessage({ sender: 'doctor', text: "Thank you for sharing that with me. I’ve noted what you said.", timestamp: new Date() });
    setStep('review');
  };

  /** Review Step */
  useEffect(() => {
    if (step === 'review' && !isConfirmingRef.current) {
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
  }, [step, transcript, voiceInput]);

  /** Follow-Up Loading */
  useEffect(() => {
    if (step === 'followupLoading') {
      window.speechSynthesis.cancel();
      speak("I’m just getting some questions ready to learn more about how you’re feeling. Bear with me for a moment.");
      addMessage({ sender: 'doctor', text: "I’m just getting some questions ready to learn more about how you’re feeling. Bear with me for a moment.", timestamp: new Date() });
      setLoading(true);

      (async () => {
        try {
          const symptom = await extractSymptoms(transcript);
          const cleanedSymptom = symptom.toLowerCase().trim().replace(/[^a-z]/g, "");
          setExtractedSymptom(cleanedSymptom);

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
  }, [step, transcript]);

  /** Static Follow-Up */
  useEffect(() => {
    if (step === 'followup' && staticFollowUpQuestions.length > 0 && !isAskingStaticRef.current) {
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
  }, [step, currentStaticIndex, staticFollowUpQuestions, staticFollowUpAnswers, voiceInput]);

  /** Dynamic Follow-Up */
  useEffect(() => {
    if (step === 'dynamicFollowupLoading' && !isLoadingDynamicRef.current) {
      isLoadingDynamicRef.current = true;
      window.speechSynthesis.cancel();
      speak("I’m putting together a few more questions to help me understand your situation even better. Just a moment.");
      addMessage({ sender: 'doctor', text: "I’m putting together a few more questions to help me understand your situation even better. Just a moment.", timestamp: new Date() });
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
          speak("I’m sorry, we couldn’t generate more questions, but I’ll do my best with what we have. Let’s move forward.");
          addMessage({ sender: 'doctor', text: "I’m sorry, we couldn’t generate more questions, but I’ll do my best with what we have. Let’s move forward.", timestamp: new Date() });
          setStep('final');
        } finally {
          setLoading(false);
          isLoadingDynamicRef.current = false;
        }
      })();
    } else if (step === 'dynamicFollowup' && dynamicFollowUpQuestions.length > 0 && !isAskingDynamicRef.current) {
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
  }, [step, dynamicFollowUpQuestions, currentDynamicIndex, transcript, extractedSymptom, staticFollowUpQuestions, staticFollowUpAnswers, voiceInput]);

  /** Final Guidelines */
  useEffect(() => {
    if (step === 'final' && !loading) {
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
          key_symptom: extractedSymptom,
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
  }, [step, transcript, extractedSymptom, staticFollowUpQuestions, staticFollowUpAnswers, dynamicFollowUpQuestions, dynamicFollowUpAnswers]);

  return (
    <div className="flex flex-col w-full min-h-screen font-sans bg-gradient-to-br from-teal-50 via-blue-100 to-indigo-50">
      <header className="w-full px-6 py-6 shadow-lg bg-gradient-to-r from-teal-600 to-indigo-700">
        <h1 className="text-4xl font-bold tracking-tight text-center text-white">Your Health, Our Care</h1>
        <div className="relative w-full h-4 mt-4 overflow-hidden bg-teal-200 rounded-full">
          <div
            className="absolute top-0 left-0 h-full transition-all duration-500 ease-in-out bg-gradient-to-r from-teal-400 to-indigo-500"
            style={{ width: getProgressPercentage() }}
          />
          {["1", "2", "3", "4", "5"].map((num, idx) => (
            <div
              key={num}
              className="absolute px-2 py-1 text-sm font-semibold text-white bg-teal-600 rounded-full -top-8"
              style={{ left: `${(idx * 100) / 4}%`, transform: 'translateX(-50%)' }}
            >
              {num}
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
              <Button onClick={() => setErrorMsg('')} className="px-6 py-3 text-white transition-colors bg-red-500 rounded-lg hover:bg-red-600">
                Let’s Start Over
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
            >
              <Mic className="w-6 h-6 mr-2" />
              Let’s Begin
            </Button>
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

      {voicePaused && (
        <div
          className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-gradient-to-b from-teal-900/80 to-indigo-900/80 animate-fade-in"
          onClick={() => resume()}
        >
          <h1 className="mb-4 text-4xl font-bold text-white">Take All the Time You Need</h1>
          <p className="text-lg text-teal-100">Whenever you’re ready, just tap here to continue.</p>
        </div>
      )}

      {loading && (step === 'followupLoading' || step === 'dynamicFollowupLoading' || step === 'final') && (
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-gradient-to-b from-teal-900/70 to-indigo-900/70">
          <Loader2 className="w-16 h-16 text-teal-300 animate-spin" />
          <p className="mt-6 text-xl font-semibold text-white">{getLoadingMessage()}</p>
        </div>
      )}

      {/* Custom Tailwind Animations */}
      <style jsx global>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideUp {
          from { transform: translateY(20px); opacity: 0; }
          to { transform: translateY(0); opacity: 1; }
        }
        @keyframes slideIn {
          from { transform: translateX(-20px); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
        @keyframes spinSlow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .animate-fade-in {
          animation: fadeIn 0.5s ease-in-out;
        }
        .animate-slide-up {
          animation: slideUp 0.5s ease-in-out;
        }
        .animate-slide-in {
          animation: slideIn 0.5s ease-in-out;
        }
        .animate-spin-slow {
          animation: spinSlow 4s linear infinite;
        }
      `}</style>
    </div>
  );
};

export default App;
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Button } from './components/ui/button'; // Assuming Shadcn UI component
import { Mic, Loader2, User, Activity } from 'lucide-react';
import './styles.css';

// Adjust the API_BASE_URL to match your server endpoint
const API_BASE_URL = 'https://c508-34-125-160-155.ngrok-free.app';

// Define application steps
type AppStep =
  | 'initial'
  | 'recording'
  | 'review'
  | 'startingInterview'
  | 'interviewing'
  | 'submittingAnswer'
  | 'final';

// Define message structure
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

// Define structures for conversation state
interface StaticFollowupItem {
  question: string;
  answer: string | null;
}

interface DynamicFollowupItem {
  question: string;
  answer: string;
}

interface ConversationState {
  transcript: string;
  symptom_summary: SymptomSummary;
  static_followup: StaticFollowupItem[];
  dynamic_followup_history: DynamicFollowupItem[];
  current_question: string | null;
}

// Custom interface for SpeechRecognitionEvent
interface CustomSpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

// Custom interface for SpeechRecognitionErrorEvent to resolve TypeScript error
interface CustomSpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

// Define SpeechRecognition globally
const SpeechRecognitionConstructor =
  (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

// Utility Functions
function speak(text: string, onEndCallback?: () => void) {
  if (!text) return;
  try {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    if (onEndCallback) utterance.onend = onEndCallback;
    utterance.onerror = (event) => {
      console.error('SpeechSynthesis Error:', event);
      if (onEndCallback) onEndCallback();
    };
    window.speechSynthesis.speak(utterance);
  } catch (error) {
    console.error('Error initiating speech synthesis:', error);
    if (onEndCallback) onEndCallback();
  }
}

const startSpeechRecognition = (
  onResult: (transcript: string) => void,
  onError: (errorType: string, errorMessage: string) => void,
  setMicActiveUI: (active: boolean) => void
): any | null => {
  if (!SpeechRecognitionConstructor) {
    onError(
      'unsupported',
      'Speech recognition not supported by this browser. Please type your response if possible.'
    );
    return null;
  }

  const recognition = new SpeechRecognitionConstructor();
  recognition.lang = 'en-US';
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;
  recognition.continuous = true;

  let finalTranscript = '';
  let silenceTimer: NodeJS.Timeout | null = null;
  const SILENCE_DURATION = 2000;

  recognition.onstart = () => {
    console.log('Speech recognition started');
    setMicActiveUI(true);
    if (silenceTimer) clearTimeout(silenceTimer);
  };

  recognition.onresult = (event: CustomSpeechRecognitionEvent) => {
    console.log('Speech result received');
    if (silenceTimer) clearTimeout(silenceTimer);

    let interimTranscript = '';
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        finalTranscript += event.results[i][0].transcript.trim() + ' ';
      } else {
        interimTranscript += event.results[i][0].transcript;
      }
    }
    console.log('Interim:', interimTranscript, 'Final:', finalTranscript);

    silenceTimer = setTimeout(() => {
      console.log('Silence detected, stopping recognition.');
      recognition.stop();
    }, SILENCE_DURATION);
  };

  recognition.onend = () => {
    console.log('Speech recognition ended');
    setMicActiveUI(false);
    if (silenceTimer) clearTimeout(silenceTimer);
    const result = finalTranscript.trim();
    if (result) onResult(result);
    else onError('no-speech', 'No speech detected or recognition ended abruptly.');
  };

  recognition.onerror = (event: CustomSpeechRecognitionErrorEvent) => {
    console.error('SpeechRecognition Error:', event.error, event.message);
    setMicActiveUI(false);
    if (silenceTimer) clearTimeout(silenceTimer);
    onError(event.error, event.message);
  };

  try {
    recognition.start();
    silenceTimer = setTimeout(() => {
      console.log('Initial silence timeout, stopping recognition.');
      recognition.stop();
    }, SILENCE_DURATION * 2);
  } catch (e) {
    console.error('Failed to start speech recognition:', e);
    onError('start-failed', 'Could not start voice input. Please check microphone permissions.');
    return null;
  }

  return recognition;
};

// Custom Hook
const useVoiceInteraction = (
  addMessage: (msg: Message) => void,
  setMicActiveUI: (active: boolean) => void,
  setErrorMsg: (msg: string) => void,
  isManuallyPaused: boolean
) => {
  const recognitionRef = useRef<any>(null);

  const cleanupRecognition = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.onstart = null;
      recognitionRef.current.onresult = null;
      recognitionRef.current.onerror = null;
      recognitionRef.current.onend = null;
      recognitionRef.current.stop();
      recognitionRef.current = null;
      setMicActiveUI(false);
    }
  }, [setMicActiveUI]);

  const askAndListen = useCallback(
    (prompt: string): Promise<string> => {
      return new Promise((resolve, reject) => {
        if (isManuallyPaused) {
          console.log('Interaction paused, rejecting request.');
          return reject('Interaction is manually paused.');
        }

        cleanupRecognition();
        addMessage({ sender: 'doctor', text: prompt, timestamp: new Date() });

        speak(prompt, () => {
          if (isManuallyPaused) {
            console.log('Interaction paused after speaking, rejecting listen.');
            return reject('Interaction is manually paused.');
          }
          console.log('Speak finished, starting recognition...');
          recognitionRef.current = startSpeechRecognition(
            (transcript) => {
              console.log('Recognition success:', transcript);
              if (transcript) {
                addMessage({ sender: 'patient', text: transcript, timestamp: new Date() });
                resolve(transcript);
              } else {
                addMessage({ sender: 'patient', text: '[No audible response]', timestamp: new Date() });
                reject('No audible response recorded.');
              }
              cleanupRecognition();
            },
            (errorType, errorMessage) => {
              console.error('Recognition error:', errorType, errorMessage);
              let userMessage = `Sorry, I had trouble understanding. ${errorMessage}`;
              if (errorType === 'no-speech') userMessage = "I didn't hear anything. Could you please speak up?";
              else if (errorType === 'audio-capture')
                userMessage = 'Hmm, I couldn’t capture audio. Please check your microphone connection and permissions.';
              else if (errorType === 'not-allowed')
                userMessage = 'I need microphone access to hear you. Please grant permission in your browser settings.';
              setErrorMsg(userMessage);
              addMessage({ sender: 'doctor', text: userMessage, timestamp: new Date() });
              speak(userMessage);
              reject(errorMessage);
              cleanupRecognition();
            },
            setMicActiveUI
          );
          if (!recognitionRef.current) {
            reject('Failed to initialize speech recognition.');
            cleanupRecognition();
          }
        });
      });
    },
    [addMessage, setMicActiveUI, setErrorMsg, isManuallyPaused, cleanupRecognition]
  );

  useEffect(() => {
    return () => {
      window.speechSynthesis.cancel();
      cleanupRecognition();
    };
  }, [cleanupRecognition]);

  return { askAndListen, cancelListening: cleanupRecognition };
};

// API Functions
async function startInterview(transcript: string): Promise<{
  status: string;
  conversation_state: ConversationState;
  audio_response: string | null;
  guidelines?: string;
  summary?: string;
  error?: string;
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/start_interview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
      body: JSON.stringify({ transcript }),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: `HTTP error ${response.status}` }));
      console.error('Start Interview API Error:', response.status, errorData);
      return { status: 'error', error: errorData?.error || `Request failed with status ${response.status}`, conversation_state: null!, audio_response: null };
    }
    const data = await response.json();
    console.log('Start Interview Response:', data);
    return data;
  } catch (error) {
    console.error('Network or fetch error in startInterview:', error);
    return { status: 'error', error: `Network error: ${error instanceof Error ? error.message : String(error)}`, conversation_state: null!, audio_response: null };
  }
}

async function submitAnswerAndContinue(
  currentState: ConversationState,
  answer: string
): Promise<{
  status: string;
  conversation_state: ConversationState;
  audio_response: string | null;
  guidelines?: string;
  summary?: string;
  error?: string;
}> {
  if (!currentState || !currentState.current_question) {
    console.error('Cannot submit answer, invalid state:', currentState);
    return { status: 'error', error: 'Internal error: Missing current question in state.', conversation_state: currentState, audio_response: null };
  }
  try {
    const response = await fetch(`${API_BASE_URL}/interview_step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
      body: JSON.stringify({ conversation_state: currentState, answer }),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: `HTTP error ${response.status}` }));
      console.error('Interview Step API Error:', response.status, errorData);
      return { status: 'error', error: errorData?.error || `Request failed with status ${response.status}`, conversation_state: currentState, audio_response: null };
    }
    const data = await response.json();
    console.log('Interview Step Response:', data);
    return data;
  } catch (error) {
    console.error('Network or fetch error in submitAnswerAndContinue:', error);
    return { status: 'error', error: `Network error: ${error instanceof Error ? error.message : String(error)}`, conversation_state: currentState, audio_response: null };
  }
}

// Main App Component
const App: React.FC = () => {
  const [step, setStep] = useState<AppStep>('initial');
  const [transcript, setTranscript] = useState<string>('');
  const [conversationState, setConversationState] = useState<ConversationState | null>(null);
  const [guidelines, setGuidelines] = useState<string>('');
  const [summary, setSummary] = useState<string>('');
  const [audioSrc, setAudioSrc] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingMessage, setLoadingMessage] = useState<string>('');
  const [overlayVisible, setOverlayVisible] = useState<boolean>(true);
  const [welcomeSpoken, setWelcomeSpoken] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string>('');
  const [conversation, setConversation] = useState<Message[]>([]);
  const [micActiveUI, setMicActiveUI] = useState<boolean>(false);
  const [isManuallyPaused, setIsManuallyPaused] = useState<boolean>(false);

  const isProcessingRef = useRef<boolean>(false);
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const waitingIntervalRef = useRef<number | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const currentQuestion = useMemo(() => conversationState?.current_question, [conversationState]);

  const addMessage = useCallback((msg: Message) => {
    setConversation((prev) => [...prev, msg]);
  }, []);

  const { askAndListen, cancelListening } = useVoiceInteraction(addMessage, setMicActiveUI, setErrorMsg, isManuallyPaused);

  const resetState = useCallback(() => {
    setStep('initial');
    setTranscript('');
    setConversationState(null);
    setGuidelines('');
    setSummary('');
    setAudioSrc('');
    setConversation([]);
    setOverlayVisible(true);
    setWelcomeSpoken(false);
    setErrorMsg('');
    setIsManuallyPaused(false);
    setLoading(false);
    setLoadingMessage('');
    isProcessingRef.current = false;
    if (waitingIntervalRef.current) clearInterval(waitingIntervalRef.current);
    window.speechSynthesis.cancel();
    cancelListening();
  }, [cancelListening]);

  const handleRestart = useCallback(() => {
    resetState();
  }, [resetState]);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation]);

  const stepIndicator = useMemo(() => {
    switch (step) {
      case 'initial': return 'Welcome';
      case 'recording': return 'Step 1: Share Your Story';
      case 'review': return 'Step 2: Confirm Your Statement';
      case 'startingInterview': return 'Getting Ready...';
      case 'interviewing': return 'Step 3: Answering Questions';
      case 'submittingAnswer': return 'Processing Your Answer...';
      case 'final': return 'Step 4: Your Care Plan';
      default: return '';
    }
  }, [step]);

  const progressPercentage = useMemo(() => {
    switch (step) {
      case 'initial': return '0%';
      case 'recording': return '15%';
      case 'review': return '30%';
      case 'startingInterview': return '40%';
      case 'interviewing': return '50%';
      case 'submittingAnswer': return '75%';
      case 'final': return '100%';
      default: return '0%';
    }
  }, [step]);

  useEffect(() => {
    if (loading && (step === 'startingInterview' || step === 'submittingAnswer' || (step === 'final' && !guidelines))) {
      let initialMessage = 'Hang tight, we’re working on it...';
      if (step === 'startingInterview') initialMessage = 'Analyzing your statement and preparing first question...';
      if (step === 'submittingAnswer') initialMessage = 'Thinking about your answer and preparing the next step...';
      if (step === 'final' && !guidelines) initialMessage = 'Creating your personalized care plan...';
      setLoadingMessage(initialMessage);

      waitingIntervalRef.current = window.setInterval(() => {
        const waitingMsg = 'Still working on this for you, thanks for your patience!';
        setLoadingMessage(waitingMsg);
        speak(waitingMsg);
        addMessage({ sender: 'doctor', text: waitingMsg, timestamp: new Date() });
      }, 25000);
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
  }, [loading, step, guidelines, addMessage]);

  const handleOverlayClick = () => {
    if (!welcomeSpoken) {
      const welcomeMsg =
        'Hello there! We’re so glad you’re here. Whenever you’re ready, tell us how you’re feeling by clicking "Let\'s Begin", and we’ll create a care plan just for you.';
      speak(welcomeMsg);
      addMessage({ sender: 'doctor', text: welcomeMsg, timestamp: new Date() });
      setWelcomeSpoken(true);
    }
    setOverlayVisible(false);
  };

  const handleStartRecording = async () => {
    if (isProcessingRef.current || isManuallyPaused) return;
    isProcessingRef.current = true;
    setStep('recording');
    setTranscript('');
    setErrorMsg('');

    try {
      const initialPrompt = 'Whenever you’re ready, please tell me how you’re feeling today. What’s been going on?';
      const result = await askAndListen(initialPrompt);
      setTranscript(result);
      setStep('review');
    } catch (error) {
      console.error('Error during initial recording:', error);
      speak('Let’s try that initial step again. Click "Let’s Begin" when ready.');
      addMessage({ sender: 'doctor', text: 'Let’s try that initial step again.', timestamp: new Date() });
      setStep('initial');
    } finally {
      isProcessingRef.current = false;
    }
  };

  useEffect(() => {
    if (step === 'review' && transcript && !isProcessingRef.current && !isManuallyPaused) {
      isProcessingRef.current = true;
      setErrorMsg('');

      const confirmTranscript = async () => {
        try {
          const reviewMsg = `Okay, I heard you say: "${transcript}". Is that correct? Please say "yes" or "no".`;
          const confirmation = await askAndListen(reviewMsg);

          if (confirmation.toLowerCase().includes('yes')) {
            speak('Great! Let me analyze that for a moment.');
            addMessage({ sender: 'doctor', text: 'Great! Let’s move forward.', timestamp: new Date() });
            setStep('startingInterview');
          } else {
            speak('My apologies. Let’s try recording that again.');
            addMessage({ sender: 'doctor', text: 'Okay, let’s try recording again.', timestamp: new Date() });
            setStep('initial');
          }
        } catch (error) {
          console.error('Error during transcript confirmation:', error);
          speak('I’m having trouble confirming. Let’s try recording again.');
          addMessage({ sender: 'doctor', text: 'Let’s try recording again.', timestamp: new Date() });
          setStep('initial');
        } finally {
          isProcessingRef.current = false;
        }
      };
      confirmTranscript();
    }
  }, [step, transcript, askAndListen, isManuallyPaused]);

  useEffect(() => {
    if (step === 'startingInterview' && transcript && !isProcessingRef.current && !isManuallyPaused) {
      isProcessingRef.current = true;
      setLoading(true);
      setLoadingMessage('Analyzing your statement...');
      setErrorMsg('');

      const performStartInterview = async () => {
        const result = await startInterview(transcript);

        if (result.status === 'error' || !result.conversation_state) {
          setErrorMsg(result.error || 'Failed to start the interview process. Please try restarting.');
          speak(result.error || 'Sorry, something went wrong starting our chat. Please try restarting.');
          addMessage({ sender: 'doctor', text: result.error || 'Something went wrong starting.', timestamp: new Date() });
          resetState();
        } else {
          setConversationState(result.conversation_state);
          setAudioSrc(result.audio_response || '');

          if (result.status === 'completed') {
            setGuidelines(result.guidelines || '');
            setSummary(result.summary || '');
            setStep('final');
            addMessage({ sender: 'doctor', text: result.summary || 'Care plan generated.', timestamp: new Date() });
          } else if (result.conversation_state.current_question) {
            setStep('interviewing');
          } else {
            console.error('Interview started "in_progress" but no current_question received.');
            setErrorMsg('Something went wrong during interview setup.');
            speak('Sorry, there was an issue setting up the questions.');
            addMessage({ sender: 'doctor', text: 'There was an issue setting up questions.', timestamp: new Date() });
            resetState();
          }
        }
        setLoading(false);
        isProcessingRef.current = false;
      };
      performStartInterview();
    }
  }, [step, transcript, isManuallyPaused, resetState]);

  useEffect(() => {
    if (step === 'interviewing' && currentQuestion && !isProcessingRef.current && !isManuallyPaused) {
      isProcessingRef.current = true;
      setErrorMsg('');

      const handleQuestion = async () => {
        try {
          const answer = await askAndListen(currentQuestion);

          if (answer && conversationState) {
            setStep('submittingAnswer');
            setLoading(true);
            setLoadingMessage('Processing your answer...');

            const result = await submitAnswerAndContinue(conversationState, answer);

            if (result.status === 'error' || !result.conversation_state) {
              setErrorMsg(result.error || 'Failed to process your answer. Please try restarting.');
              speak(result.error || 'Sorry, something went wrong processing that. Please try restarting.');
              addMessage({ sender: 'doctor', text: result.error || 'Something went wrong.', timestamp: new Date() });
              resetState();
            } else {
              setConversationState(result.conversation_state);
              setAudioSrc(result.audio_response || '');

              if (result.status === 'completed') {
                setGuidelines(result.guidelines || '');
                setSummary(result.summary || '');
                setStep('final');
                addMessage({ sender: 'doctor', text: result.summary || 'Thank you. I’ve completed your care plan.', timestamp: new Date() });
              } else if (result.conversation_state.current_question) {
                setStep('interviewing');
              } else {
                console.error('Interview step "in_progress" but no next question.');
                setErrorMsg('Something went wrong fetching the next question.');
                speak('Sorry, there was an issue getting the next step.');
                addMessage({ sender: 'doctor', text: 'Issue getting next step.', timestamp: new Date() });
                resetState();
              }
            }
          }
        } catch (error) {
          console.error('Error during interview question handling:', error);
          if (!errorMsg) {
            speak('I encountered an issue. Let’s pause for a moment.');
            addMessage({ sender: 'doctor', text: 'Encountered an issue.', timestamp: new Date() });
          }
        } finally {
          setLoading(false);
          isProcessingRef.current = false;
        }
      };
      handleQuestion();
    }
  }, [step, currentQuestion, conversationState, askAndListen, isManuallyPaused, audioSrc, errorMsg, resetState]);

  useEffect(() => {
    if (step === 'final' && audioSrc && audioRef.current) {
      if (audioRef.current.currentSrc !== audioSrc) {
        audioRef.current.src = audioSrc;
        audioRef.current.load();
      }
      const playTimeout = setTimeout(() => {
        if (audioRef.current) {
          audioRef.current.play().catch((err) => console.error('Audio auto-play failed:', err));
        }
      }, 300);
      return () => clearTimeout(playTimeout);
    }
  }, [step, audioSrc]);

  const handleTogglePause = () => {
    if (isManuallyPaused) {
      setIsManuallyPaused(false);
      speak('Okay, let’s continue where we left off.');
      addMessage({ sender: 'doctor', text: 'Resuming...', timestamp: new Date() });
    } else {
      window.speechSynthesis.cancel();
      cancelListening();
      setIsManuallyPaused(true);
      speak('Pausing now. Tap "Resume" or click the screen when you’re ready to continue.');
      addMessage({ sender: 'doctor', text: 'Paused. Tap "Resume" or screen to continue.', timestamp: new Date() });
    }
  };

  return (
    <div className="flex flex-col w-full min-h-screen font-sans bg-gradient-to-br from-teal-50 via-blue-100 to-indigo-50">
      <header className="sticky top-0 z-40 w-full px-6 py-4 shadow-md bg-gradient-to-r from-teal-600 to-indigo-700">
        <h1 className="text-3xl font-bold tracking-tight text-center text-white">Your Health, Our Care</h1>
        <div className="relative w-full h-3 max-w-3xl mx-auto mt-3 mb-1 overflow-hidden rounded-full bg-teal-200/50">
          <div
            className="absolute top-0 left-0 h-full transition-all duration-500 ease-out rounded-full bg-gradient-to-r from-green-400 to-teal-500"
            style={{ width: progressPercentage }}
          />
        </div>
        <h2 className="text-xl font-semibold text-center text-teal-100">{stepIndicator}</h2>
      </header>

      <main className="flex-grow w-full max-w-5xl px-4 py-8 mx-auto sm:px-6" role="main" aria-live="polite">
        {overlayVisible && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center cursor-pointer bg-gradient-to-b from-teal-900/80 to-indigo-900/80 animate-fade-in"
            onClick={handleOverlayClick}
          >
            <div className="p-8 text-center transition-all transform bg-white shadow-2xl rounded-2xl hover:scale-105">
              <h1 className="mb-4 text-4xl font-bold text-teal-700">Welcome!</h1>
              <p className="max-w-md mx-auto text-lg text-gray-600">
                Ready to begin? Tap anywhere or click "Let’s Begin" below.
              </p>
            </div>
          </div>
        )}

        {errorMsg && (
          <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black bg-opacity-60 p-4">
            <div className="w-full max-w-md p-6 text-center bg-white shadow-xl rounded-2xl animate-slide-up">
              <h2 className="mb-3 text-xl font-semibold text-red-600">An Issue Occurred</h2>
              <p className="mb-5 text-gray-700">{errorMsg}</p>
              <Button
                onClick={() => {
                  setErrorMsg('');
                  resetState();
                }}
                className="px-5 py-2 text-white bg-red-500 rounded-lg hover:bg-red-600"
                aria-label="Restart after error"
              >
                Restart Process
              </Button>
            </div>
          </div>
        )}

        {step === 'initial' && !overlayVisible && (
          <div className="flex flex-col items-center justify-center w-full pt-10 space-y-6 text-center animate-fade-in">
            <div className="relative flex items-center justify-center transition-transform transform bg-teal-100 rounded-full shadow-lg w-28 h-28 hover:scale-110">
              <Mic className="text-teal-600 w-14 h-14" />
            </div>
            <h1 className="text-3xl font-bold text-teal-800">Let’s Talk About How You’re Feeling</h1>
            <p className="max-w-lg text-base leading-relaxed text-gray-600">
              Click the button below and tell us what’s been bothering you. Speak naturally, we’re here to listen.
            </p>
            <Button
              onClick={handleStartRecording}
              disabled={isProcessingRef.current || isManuallyPaused}
              className="inline-flex items-center justify-center px-6 py-3 text-lg text-white transition-transform bg-teal-600 rounded-full shadow-lg hover:bg-teal-700 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Start sharing how you feel"
            >
              <Mic className="w-5 h-5 mr-2" />
              Let’s Begin
            </Button>
          </div>
        )}

        {loading && (step === 'startingInterview' || step === 'submittingAnswer' || (step === 'final' && !guidelines)) && (
          <div className="fixed inset-0 z-[55] flex flex-col items-center justify-center text-center bg-gradient-to-b from-teal-900/70 to-indigo-900/70 p-4">
            <Loader2 className="w-12 h-12 mb-4 text-teal-300 animate-spin" />
            <p className="text-lg font-semibold text-white">{loadingMessage || 'Processing...'}</p>
          </div>
        )}

        {step === 'final' && !loading && (guidelines || summary) && (
          <div className="w-full p-6 bg-white shadow-xl sm:p-8 rounded-2xl animate-slide-up">
            <h2 className="mb-5 text-3xl font-bold text-center text-teal-700">Your Personalized Care Plan</h2>
            {summary && (
              <div className="p-4 mb-6 border-l-4 border-teal-500 rounded-r-lg bg-teal-50">
                <h3 className="mb-2 text-xl font-semibold text-teal-600">Quick Summary</h3>
                <p className="text-base leading-relaxed text-gray-700">{summary}</p>
                {audioSrc && (
                  <div className="mt-4">
                    <audio ref={audioRef} controls className="w-full h-10">
                      <source src={audioSrc} type="audio/mp3" />
                      Your browser does not support the audio element.
                    </audio>
                  </div>
                )}
              </div>
            )}
            {guidelines && (
              <div className="mt-4">
                <h3 className="mb-2 text-xl font-semibold text-teal-600">Detailed Guidelines</h3>
                <div className="space-y-2 text-base leading-relaxed text-gray-700 whitespace-pre-wrap">{guidelines}</div>
              </div>
            )}
            <div className="mt-8 text-center">
              <Button onClick={handleRestart} className="px-6 py-2 text-white bg-teal-600 rounded-lg hover:bg-teal-700">
                Start Over
              </Button>
            </div>
          </div>
        )}

        {step !== 'initial' && step !== 'final' && !overlayVisible && (
          <div className="mt-6 text-center">
            <p className="text-gray-600">
              {step === 'recording'
                ? 'Listening for your initial statement...'
                : step === 'review'
                ? 'Please confirm the transcript...'
                : step === 'interviewing' && currentQuestion
                ? `Waiting for your answer to: "${currentQuestion}"`
                : 'Processing...'}
            </p>
          </div>
        )}

        {step !== 'initial' && !overlayVisible && (
          <div className="grid w-full grid-cols-1 gap-6 mt-8 lg:grid-cols-5">
            <div className="lg:col-span-3 bg-white p-4 sm:p-6 rounded-2xl shadow-lg h-[30rem] flex flex-col overflow-hidden transition-all hover:shadow-xl">
              <h2 className="flex-shrink-0 mb-4 text-xl font-semibold text-teal-700">Conversation Log</h2>
              <div className="flex-grow pr-2 space-y-3 overflow-y-auto">
                {conversation.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`flex items-end text-sm ${msg.sender === 'doctor' ? 'justify-start' : 'justify-end'} animate-slide-in`}
                  >
                    {msg.sender === 'doctor' && <User className="flex-shrink-0 w-5 h-5 mr-2 text-teal-600" />}
                    <div
                      className={`px-3 py-2 rounded-xl max-w-[80%] shadow-sm ${
                        msg.sender === 'doctor' ? 'bg-teal-50 text-gray-800 rounded-bl-none' : 'bg-indigo-50 text-gray-800 rounded-br-none'
                      }`}
                    >
                      <p className="leading-snug">{msg.text}</p>
                    </div>
                    {msg.sender === 'patient' && <User className="flex-shrink-0 w-5 h-5 ml-2 text-indigo-600" />}
                  </div>
                ))}
                {loading && step === 'submittingAnswer' && (
                  <div className="flex items-center justify-center p-2">
                    <Loader2 className="w-5 h-5 text-teal-600 animate-spin" />
                    <p className="ml-2 text-sm text-gray-500">Processing answer...</p>
                  </div>
                )}
                <div ref={conversationEndRef} />
              </div>
            </div>

            <div className="lg:col-span-2 bg-white p-4 sm:p-6 rounded-2xl shadow-lg h-[30rem] overflow-auto transition-all hover:shadow-xl">
              <h2 className="mb-4 text-xl font-semibold text-teal-700">Current Summary</h2>
              {conversationState?.symptom_summary ? (
                <div className="space-y-2 text-sm">
                  <p>
                    <strong className="font-medium text-teal-700">Main Symptom:</strong>{' '}
                    {conversationState.symptom_summary.key_symptom || 'Not identified'}
                  </p>
                  <p>
                    <strong className="font-medium text-teal-700">Severity:</strong>{' '}
                    {conversationState.symptom_summary.severity || 'Not specified'}
                  </p>
                  <p>
                    <strong className="font-medium text-teal-700">Onset/Duration:</strong>{' '}
                    {conversationState.symptom_summary.onset_duration || 'Not specified'}
                  </p>
                  <p>
                    <strong className="font-medium text-teal-700">Location:</strong>{' '}
                    {conversationState.symptom_summary.location || 'Not specified'}
                  </p>
                  <p>
                    <strong className="font-medium text-teal-700">Character:</strong>{' '}
                    {conversationState.symptom_summary.character || 'Not specified'}
                  </p>
                  <p>
                    <strong className="font-medium text-teal-700">Other Symptoms:</strong>{' '}
                    {conversationState.symptom_summary.associated_symptoms || 'None mentioned'}
                  </p>
                </div>
              ) : (
                <p className="text-gray-500">Summary will appear here after your initial statement.</p>
              )}
              {conversationState?.static_followup &&
                conversationState.static_followup.some((item) => item.answer) && (
                  <div className="pt-4 mt-4 border-t">
                    <h3 className="mb-2 text-base font-semibold text-teal-700">Standard Questions Answered</h3>
                    <ul className="space-y-1 text-sm">
                      {conversationState.static_followup
                        .filter((item) => item.answer)
                        .map((item, idx) => (
                          <li key={`static-ans-${idx}`}>
                            <strong>Q:</strong> {item.question} <br />
                            <strong>A:</strong> {item.answer}
                          </li>
                        ))}
                    </ul>
                  </div>
                )}
            </div>
          </div>
        )}

        {micActiveUI && (
          <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black bg-opacity-40 pointer-events-none">
            <div className="flex items-center justify-center w-32 h-32 bg-teal-100 rounded-full shadow-2xl animate-pulse">
              <Mic className="w-16 h-16 text-teal-600" />
            </div>
          </div>
        )}

        {isManuallyPaused && (
          <div
            className="fixed inset-0 z-[80] flex flex-col items-center justify-center text-center cursor-pointer bg-gradient-to-b from-gray-900/80 to-black/80 animate-fade-in p-4"
            onClick={handleTogglePause}
          >
            <Activity className="w-16 h-16 mb-4 text-teal-300 animate-pulse" />
            <h1 className="mb-2 text-3xl font-bold text-white">Paused</h1>
            <p className="text-lg text-teal-100">Tap anywhere or click "Resume" to continue.</p>
          </div>
        )}
      </main>

      <footer className="w-full py-4 text-center text-gray-600 bg-teal-100/50">
        <p className="text-xs">
          This tool provides information but is not a substitute for professional medical advice. Always consult a doctor for diagnosis and treatment.
        </p>
      </footer>

      {step !== 'initial' && !overlayVisible && (
        <>
          <Button
            onClick={handleTogglePause}
            className={`fixed z-[90] px-4 py-2 text-white transition-colors rounded-full shadow-lg bottom-4 left-4 ${
              isManuallyPaused ? 'bg-green-500 hover:bg-green-600' : 'bg-yellow-500 hover:bg-yellow-600'
            }`}
            aria-label={isManuallyPaused ? 'Resume the process' : 'Pause the process'}
            disabled={loading || step === 'final'}
          >
            {isManuallyPaused ? 'Resume' : 'Pause'}
          </Button>

          <Button
            onClick={handleRestart}
            className="fixed z-[90] px-4 py-2 text-white transition-colors bg-red-500 rounded-full shadow-lg bottom-4 right-4 hover:bg-red-600"
            aria-label="Restart the health journey"
            disabled={loading}
          >
            Restart
          </Button>
        </>
      )}
    </div>
  );
};

export default App;
import React, { useState, useRef, useEffect } from 'react';

interface AudioRecorderProps {
  onRecordingComplete: (audioBlob: Blob) => void;
  autoStart?: boolean;
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({ onRecordingComplete, autoStart = false }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };
      mediaRecorderRef.current.onstop = () => {
        if (timerRef.current) {
          clearInterval(timerRef.current);
          setRecordingTime(0);
        }
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        onRecordingComplete(audioBlob);
      };
      mediaRecorderRef.current.start();
      setIsRecording(true);
      timerRef.current = window.setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("We couldn't access your microphone. Please check your permissions and try again.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  useEffect(() => {
    if (autoStart) {
      startRecording();
    }
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [autoStart]);

  return (
    <div className="flex flex-col items-center">
      {isRecording && (
        <div className="flex items-center mb-4 space-x-3">
          <span className="w-4 h-4 bg-red-500 rounded-full animate-pulse"></span>
          <span className="font-semibold text-red-600 text-lg">Recording... {recordingTime}s</span>
        </div>
      )}
      {isRecording ? (
        <button
          onClick={stopRecording}
          className="px-6 py-3 text-white bg-red-500 rounded-full shadow-lg hover:bg-red-600 transition-all duration-200"
        >
          Stop Recording
        </button>
      ) : (
        !autoStart && (
          <button
            onClick={startRecording}
            className="px-6 py-3 text-white bg-blue-500 rounded-full shadow-lg hover:bg-blue-600 transition-all duration-200"
          >
            Start Recording
          </button>
        )
      )}
    </div>
  );
};
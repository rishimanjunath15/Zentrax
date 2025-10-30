import whisper
import speech_recognition as sr
import numpy as np
import torch
import io
import wave


class WhisperHandler:
    """
    Enhanced speech recognition using OpenAI's Whisper v3 mini model.
    Provides better accuracy and offline capabilities compared to Google API.
    """
    
    def __init__(self, model_name="base", device=None):
        """
        Initialize Whisper model.
        
        Args:
            model_name: Model size - 'tiny', 'base', 'small', 'medium', 'large'
                       'base' is recommended for good balance of speed and accuracy
            device: 'cuda' for GPU, 'cpu' for CPU, None for auto-detect
        """
        print(f"Loading Whisper {model_name} model...")
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = whisper.load_model(model_name, device=device)
        print(f"✅ Whisper model loaded on {device}")
        
        # Initialize speech recognizer for audio capture
        self.recognizer = sr.Recognizer()
    
    def transcribe_audio(self, audio_data, language="en"):
        """
        Transcribe audio data using Whisper.
        
        Args:
            audio_data: AudioData object from speech_recognition
            language: Language code (e.g., 'en' for English)
        
        Returns:
            Transcribed text as string
        """
        try:
            # Convert AudioData to numpy array
            audio_array = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_float,
                language=language,
                fp16=(self.device == "cuda"),
                task="transcribe"
            )
            
            return result["text"].strip()
            
        except Exception as e:
            print(f"Whisper transcription error: {e}")
            return ""
    
    def transcribe_from_file(self, audio_file_path, language="en"):
        """
        Transcribe audio from a file.
        
        Args:
            audio_file_path: Path to audio file
            language: Language code
        
        Returns:
            Transcribed text as string
        """
        try:
            result = self.model.transcribe(
                audio_file_path,
                language=language,
                fp16=(self.device == "cuda")
            )
            return result["text"].strip()
        except Exception as e:
            print(f"File transcription error: {e}")
            return ""
    
    def listen_and_transcribe(self, microphone, timeout=5, phrase_time_limit=5, language="en"):
        """
        Listen from microphone and transcribe using Whisper.
        
        Args:
            microphone: Microphone source
            timeout: Listening timeout in seconds
            phrase_time_limit: Maximum phrase duration
            language: Language code
        
        Returns:
            Transcribed text as string
        """
        try:
            with microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            return self.transcribe_audio(audio, language=language)
            
        except sr.WaitTimeoutError:
            return ""
        except Exception as e:
            print(f"Listen error: {e}")
            return ""
    
    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            "device": self.device,
            "model_type": type(self.model).__name__,
            "is_multilingual": self.model.is_multilingual
        }


class HybridRecognizer:
    """
    Hybrid recognizer that can use both Whisper and Google Speech API.
    Falls back to Google if Whisper fails.
    """
    
    def __init__(self, use_whisper=True, whisper_model="base"):
        """
        Initialize hybrid recognizer.
        
        Args:
            use_whisper: Whether to use Whisper as primary recognizer
            whisper_model: Whisper model size to use
        """
        self.use_whisper = use_whisper
        self.recognizer = sr.Recognizer()
        
        if use_whisper:
            try:
                self.whisper = WhisperHandler(model_name=whisper_model)
                print("✅ Hybrid mode: Whisper (primary) + Google (fallback)")
            except Exception as e:
                print(f"⚠️ Whisper initialization failed: {e}")
                print("Falling back to Google Speech API only")
                self.use_whisper = False
                self.whisper = None
        else:
            self.whisper = None
            print("✅ Using Google Speech API only")
    
    def recognize(self, audio_data, language="en"):
        """
        Recognize speech using Whisper with Google fallback.
        
        Args:
            audio_data: AudioData object
            language: Language code
        
        Returns:
            Recognized text as string
        """
        # Try Whisper first
        if self.use_whisper and self.whisper:
            text = self.whisper.transcribe_audio(audio_data, language=language)
            if text:
                return text
        
        # Fallback to Google
        try:
            return self.recognizer.recognize_google(audio_data).lower()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print(f"Google API error: {e}")
            return ""
    
    def listen_and_recognize(self, microphone, timeout=5, phrase_time_limit=5, language="en"):
        """
        Listen and recognize with automatic fallback.
        
        Args:
            microphone: Microphone source
            timeout: Listening timeout
            phrase_time_limit: Maximum phrase duration
            language: Language code
        
        Returns:
            Recognized text
        """
        try:
            with microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            return self.recognize(audio, language=language)
            
        except sr.WaitTimeoutError:
            return ""
        except Exception as e:
            print(f"Recognition error: {e}")
            return ""

import speech_recognition as sr
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import threading
import time
import os
import sys
import pywhatkit
from datetime import datetime
from queue import Queue, Empty

# Optional non-PyAudio audio capture fallback
try:
    import sounddevice as _sd  # used only if PyAudio/sr.Microphone fails
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    _sd = None
    SOUNDDEVICE_AVAILABLE = False

# --- Safe imports / fallbacks for missing modules ---
try:
	# try to import real implementations if present
	from hill_climb_game import HillClimbGame
except Exception:
	# Minimal stub so main program can run without the actual game module
	class HillClimbGame:
		def __init__(self):
			self.running = False
		def run(self):
			# simple loop placeholder (non-blocking use expected in main)
			self.running = True
			while self.running:
				time.sleep(0.1)
		def handle_gesture(self, gesture):
			print(f"[HillClimbGame stub] Gesture received: {gesture}")

try:
	from whisper_handler import HybridRecognizer
except Exception:
	# Fallback HybridRecognizer using speech_recognition's Google API
	class HybridRecognizer:
		def __init__(self, use_whisper=False, whisper_model="base"):
			import speech_recognition as sr
			self.recognizer = sr.Recognizer()
			self.use_whisper = False  # fallback doesn't use Whisper
		def recognize(self, audio):
			# audio: instance of speech_recognition.AudioData
			try:
				# Prefer the recognizer's built-in Google API as a lightweight fallback.
				return self.recognizer.recognize_google(audio)
			except Exception:
				return ""


class VoiceGestureControl:
    def __init__(self, use_whisper=True, whisper_model="base"):
        # ---------------- Initialization ----------------
        # Initialize Whisper-based hybrid recognizer
        self.hybrid_recognizer = HybridRecognizer(use_whisper=use_whisper, whisper_model=whisper_model)
        self.recognizer = self.hybrid_recognizer.recognizer

        # detect microphone (may return None if PyAudio missing)
        self.mic_index = self.get_working_microphone()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.is_awake = False
        self.active_mode = "voice"
        self.running = True
        self.listening = True
        self.gesture_cooldown = 0
        self.game = None

        # Add configurable wake phrase (change here if you want a different trigger)
        self.wake_phrase = "hello"

        # audio queue for non-blocking transcription
        self.audio_queue = Queue()
        self.audio_worker = threading.Thread(target=self._audio_worker, daemon=True)

        # Voice commands
        self.voice_commands = {
            "open browser": self.open_browser,
            "close window": self.close_window,
            "minimize": self.minimize_window,
            "maximize": self.maximize_window,
            "volume up": self.volume_up,
            "volume down": self.volume_down,
            "scroll up": self.scroll_up,
            "scroll down": self.scroll_down,
            "take screenshot": self.take_screenshot,
            "exit program": self.exit_program,
            "play hill climb": self.start_hill_climb,
        }

        # Contacts
        self.contacts = {
            "john": "+1234567890",
            "jane": "+1987654321",
            "rishi": "+919876543210",
            "mom": "+919988776655",
            "dad": "+112233445566"
        }

    # ---------------- Hill Climb Integration ----------------
    def start_hill_climb(self):
        print("🎮 Starting Hill Climb Game...")
        self.game = HillClimbGame()
        threading.Thread(target=self.game.run, daemon=True).start()

    # ---------------- Microphone Handling ----------------
    def get_working_microphone(self):
        try:
            mic_list = sr.Microphone.list_microphone_names()
        except (AttributeError, ModuleNotFoundError, OSError) as e:
            # PyAudio missing -> speech_recognition cannot list devices here.
            # Don't force a hard "NO_MIC" exit; return None to attempt default mic usage
            print("PyAudio is not installed or not available. Voice input may still work via system default or a fallback recorder.")
            print("Install on Windows (recommended):")
            print("  python -m pip install pipwin")
            print("  python -m pipwin install pyaudio")
            print("Or download a prebuilt wheel from:")
            print("  https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
            return None

        if not mic_list:
            print("No audio devices found.")
            return None

        # Prefer explicit matches; otherwise use default microphone
        preferred_keywords = ["microphone", "realtek", "amd", "mic", "audio", "internal", "default"]

        for i, name in enumerate(mic_list):
            if any(k in name.lower() for k in preferred_keywords):
                try:
                    with sr.Microphone(device_index=i) as test_source:
                        if getattr(test_source, "stream", None) is not None:
                            print(f"✅ Using microphone {i}: {name}")
                            return i
                except Exception:
                    continue

        # If no preferred device found, try to verify any device works and return its index
        for i, name in enumerate(mic_list):
            try:
                with sr.Microphone(device_index=i) as test_source:
                    if getattr(test_source, "stream", None) is not None:
                        print(f"✅ Using fallback mic {i}: {name}")
                        return i
            except Exception:
                continue

        # As a last resort, use system default microphone (None)
        print("No specific microphone validated; using system default microphone.")
        return None

    # ---------------- Audio worker (non-blocking) ----------------
    def _audio_worker(self):
        while self.running:
            try:
                audio = self.audio_queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                text = self.hybrid_recognizer.recognize(audio) or ""
                if text:
                    self._handle_recognized_text(text.lower())
            except Exception as e:
                print(f"[Audio worker error]: {e}")

    # ---------------- Voice Control ----------------
    def listen_for_commands(self):
        print("Voice recognition started...")

        # If PyAudio truly absent we still try default path and fallback recorder.
        # Pre-adjust ambient noise once (use default mic when mic_index is None)
        try:
            if self.mic_index is None:
                try:
                    with sr.Microphone() as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
                except Exception:
                    # If sr.Microphone isn't usable (PyAudio missing), skip ambient adjust and continue:
                    print("sr.Microphone unavailable; will try alternative recorder if needed.")
            else:
                with sr.Microphone(device_index=self.mic_index) as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
        except Exception:
            pass

        self.audio_worker.start()

        while self.running and self.listening:
            try:
                audio = None
                try:
                    if self.mic_index is None:
                        # Try default sr.Microphone
                        with sr.Microphone() as source:
                            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    else:
                        with sr.Microphone(device_index=self.mic_index) as source:
                            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                except Exception as e:
                    # sr.Microphone failed (likely PyAudio missing). Try sounddevice fallback.
                    if SOUNDDEVICE_AVAILABLE:
                        print("sr.Microphone unavailable; using sounddevice fallback to capture audio.")
                        audio = self.record_with_sounddevice(duration=5, fs=16000)
                        if audio is None:
                            print("Fallback recorder failed; waiting before retrying.")
                            time.sleep(1)
                            continue
                    else:
                        print(f"Voice input unavailable: {e}")
                        # wait and retry instead of exiting the loop
                        time.sleep(1)
                        continue

                # enqueue for background processing
                if audio:
                    self.audio_queue.put(audio)
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"[Voice Error]: {e}")
                time.sleep(1)

    def _handle_recognized_text(self, text):
        print(f"Recognized: {text}")
        if not self.is_awake:
            # use configurable wake phrase
            if self.wake_phrase in text:
                self.is_awake = True
                print("Zentrax is awake!")
            return

        if "go to sleep" in text or "deactivate" in text:
            self.is_awake = False
            return
        if "switch to gesture mode" in text:
            self.active_mode = "gesture"
            return
        if "switch to voice mode" in text:
            self.active_mode = "voice"
            return

        if self.active_mode == "voice":
            if text.startswith("play music"):
                song = text.replace("play music", "").strip()
                if song: self.play_music(song)
                return
            elif text.startswith("send whatsapp message to"):
                self.handle_whatsapp(text)
                return
            for command, func in self.voice_commands.items():
                if command in text:
                    func()
                    break

    # ---------------- WhatsApp ----------------
    def handle_whatsapp(self, text):
        try:
            parts = text.split("saying", 1)
            if len(parts) != 2: return
            contact_phrase = parts[0].replace("send whatsapp message to", "").strip()
            message = parts[1].strip()
            for name, number in self.contacts.items():
                if name in contact_phrase:
                    now = datetime.now()
                    pywhatkit.sendwhatmsg(number, message, now.hour, (now.minute + 1) % 60)
        except Exception as e:
            print(f"WhatsApp error: {e}")

    # ---------------- Gesture Control ----------------
    def process_gestures(self):
        print("Gesture recognition started...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                image = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image_rgb)

                if self.is_awake and self.active_mode == "gesture" and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        gesture = self.recognize_gesture(hand_landmarks)
                        if gesture and time.time() > self.gesture_cooldown:
                            self.execute_gesture(gesture)
                            self.gesture_cooldown = time.time() + 0.2  # faster for real-time

                cv2.imshow('Hand Tracking', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    # ---------------- Gesture Recognition ----------------
    def recognize_gesture(self, landmarks):
        points = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
        if self.is_open_palm(points): return "open_palm"
        if self.is_closed_fist(points): return "closed_fist"
        if self.is_thumbs_up(points): return "thumbs_up"
        if self.is_thumbs_down(points): return "thumbs_down"
        return None

    def is_open_palm(self, points):
        return all(points[tip][1] < points[mcp][1] for tip, mcp in zip([8,12,16,20],[5,9,13,17]))
    def is_closed_fist(self, points):
        return all(points[tip][1] > points[mcp][1] for tip, mcp in zip([8,12,16,20],[5,9,13,17]))
    def is_thumbs_up(self, points): return points[4][1] < points[3][1]
    def is_thumbs_down(self, points): return points[4][1] > points[3][1]

    # ---------------- Execute Gestures ----------------
    def execute_gesture(self, gesture):
        if self.game and getattr(self.game, "running", False):
            self.game.handle_gesture(gesture)
            print(f"[Game] Gesture: {gesture}")
            return
        # fallback system gestures
        try:
            if gesture == "open_palm": self.maximize_window()
            elif gesture == "closed_fist": self.minimize_window()
            elif gesture == "thumbs_up": self.volume_up()
            elif gesture == "thumbs_down": self.volume_down()
        except Exception as e:
            print(f"[Gesture Error]: {e}")

    # ---------------- System Controls ----------------
    def open_browser(self): os.system("start https://www.google.com")
    def close_window(self): pyautogui.hotkey('alt', 'f4')
    def minimize_window(self): pyautogui.hotkey('win', 'down')
    def maximize_window(self): pyautogui.hotkey('win', 'up')
    def volume_up(self): pyautogui.press('volumeup')
    def volume_down(self): pyautogui.press('volumedown')
    def scroll_up(self): pyautogui.scroll(10)
    def scroll_down(self): pyautogui.scroll(-10)
    def take_screenshot(self):
        path = os.path.join(os.path.expanduser("~"), "Desktop", f"screenshot_{int(time.time())}.png")
        pyautogui.screenshot().save(path)
        print(f"Saved screenshot: {path}")
    def exit_program(self): self.running = False
    def play_music(self, name): pywhatkit.playonyt(name)

    def record_with_sounddevice(self, duration=5, fs=16000):
        """Fallback recorder using sounddevice -> returns sr.AudioData or None."""
        if not SOUNDDEVICE_AVAILABLE:
            return None
        try:
            # record mono int16
            audio = _sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            _sd.wait()
            audio_bytes = audio.tobytes()
            return sr.AudioData(audio_bytes, fs, 2)
        except Exception as e:
            print(f"[sounddevice fallback error]: {e}")
            return None

    # ---------------- Main Run ----------------
    def run(self):
        print("Starting Voice & Gesture Control...")
        voice_thread = threading.Thread(target=self.listen_for_commands, daemon=True)
        voice_thread.start()
        self.process_gestures()
        self.running = False
        voice_thread.join(timeout=2)
        print("Shutdown complete.")


if __name__ == "__main__":
    controller = VoiceGestureControl()
    controller.run()

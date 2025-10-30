import speech_recognition as sr
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import threading
import time
import os
import sys
import traceback
import pywhatkit
from datetime import datetime
from hill_climb_game import HillClimbGame
from whisper_handler import HybridRecognizer


class VoiceGestureControl:
    def __init__(self, use_whisper=True, whisper_model="base"):
        # ---------------- Initialization ----------------
        # Initialize Whisper-based hybrid recognizer
        self.hybrid_recognizer = HybridRecognizer(use_whisper=use_whisper, whisper_model=whisper_model)
        self.recognizer = self.hybrid_recognizer.recognizer
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
        mic_list = sr.Microphone.list_microphone_names()
        preferred_keywords = ["microphone", "realtek", "amd", "mic", "audio"]

        for i, name in enumerate(mic_list):
            if any(k in name.lower() for k in preferred_keywords):
                try:
                    with sr.Microphone(device_index=i) as test_source:
                        if test_source.stream is not None:
                            print(f"✅ Using microphone {i}: {name}")
                            return i
                except:
                    continue
        # Fallback
        for i, name in enumerate(mic_list):
            try:
                with sr.Microphone(device_index=i) as test_source:
                    if test_source.stream is not None:
                        print(f"✅ Using fallback mic {i}: {name}")
                        return i
            except:
                continue
        print("No working microphone found!")
        sys.exit(1)

    # ---------------- Voice Control ----------------
    def listen_for_commands(self):
        print("Voice recognition started...")
        while self.running and self.listening:
            try:
                with sr.Microphone(device_index=self.mic_index) as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                try:
                    # Use Whisper hybrid recognizer for better accuracy
                    text = self.hybrid_recognizer.recognize(audio).lower()
                    if not text:
                        continue
                    print(f"Recognized: {text}")

                    # Wake word
                    if not self.is_awake:
                        if "hello zentrax" in text:
                            self.is_awake = True
                            print("Zentrax is awake!")
                        continue

                    # Sleep or switch modes
                    if "go to sleep" in text or "deactivate" in text:
                        self.is_awake = False
                        continue
                    if "switch to gesture mode" in text:
                        self.active_mode = "gesture"
                        continue
                    if "switch to voice mode" in text:
                        self.active_mode = "voice"
                        continue

                    # Voice commands
                    if self.active_mode == "voice":
                        if text.startswith("play music"):
                            song = text.replace("play music", "").strip()
                            if song: self.play_music(song)
                            continue
                        elif text.startswith("send whatsapp message to"):
                            self.handle_whatsapp(text)
                            continue
                        for command, func in self.voice_commands.items():
                            if command in text:
                                func()
                                break

                except sr.UnknownValueError:
                    pass
            except Exception as e:
                print(f"[Voice Error]: {e}")
                time.sleep(1)

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

        try:
            while self.running:
                success, image = cap.read()
                if not success: break
                image = cv2.flip(image, 1)
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
                if cv2.waitKey(5) & 0xFF == ord('q'):
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
        if self.game and self.game.running:
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

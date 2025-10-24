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

class VoiceGestureControl:
    def __init__(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()

        # Discover microphones
        mic_list = sr.Microphone.list_microphone_names()
        if not mic_list:
            print("No microphone found. Please connect one.")
            sys.exit(1)

        print("Available microphones:")
        for i, mic in enumerate(mic_list):
            print(f"{i}: {mic}")

        # Pick default microphone
        self.mic_index = self.get_working_microphone()

        # Initialize hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Command mappings
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
            "exit program": self.exit_program
            # "play music" and "send whatsapp message" will be handled separately due to dynamic arguments
            # "go to sleep", "switch to gesture mode", "switch to voice mode" also handled separately
        }

        # Define your contacts for WhatsApp messages (Name: Phone Number with country code)
        self.contacts = {
            "john": "+1234567890",  # Example: Replace with actual phone numbers
            "jane": "+1987654321",
            "rishi": "+919876543210", # Example for India
            "mom": "+919988776655", # New contact example
            "dad": "+112233445566"  # Another new contact example
        }

        # Gesture mappings / cooldown
        self.last_gesture = None
        self.gesture_cooldown = 0

        # Control flags
        self.running = True
        self.listening = True

        # Mouse tracking state
        self.tracking_mouse = False
        self.tracking_end_time = 0

        # New: Wake word and mode control
        self.is_awake = False # System is asleep, only listens for wake word
        self.active_mode = "voice" # Default active mode when awake ("voice" or "gesture")

    # ---------------- Microphone Handling ----------------
    def get_working_microphone(self):
        """Find a working microphone automatically."""
        mic_list = sr.Microphone.list_microphone_names()
        preferred_mic_keywords = ["microphone", "realtek", "amd", "mic", "audio"]

        # Prefer real microphones
        for i, name in enumerate(mic_list):
            low = name.lower()
            if any(k in low for k in preferred_mic_keywords):
                try:
                    with sr.Microphone(device_index=i) as test_source:
                        if test_source.stream is not None:
                            print(f"✅ Using microphone {i}: {name}")
                            return i
                except Exception as e:
                    print(f"❌ Mic {i} ({name}) failed: {e}")

        # Fallback: first working mic
        for i, name in enumerate(mic_list):
            try:
                with sr.Microphone(device_index=i) as test_source:
                    if test_source.stream is not None:
                        print(f"✅ Using fallback mic {i}: {name}")
                        return i
            except Exception:
                continue

        print("No working microphone found!")
        sys.exit(1)

    # ---------------- Voice ----------------
    def listen_for_commands(self):
        print("Voice recognition started. Speak commands...")
        while self.running and self.listening:
            try:
                with sr.Microphone(device_index=self.mic_index) as source:
                    if source.stream is None:
                        print(f"[Mic Error] Stream failed at index {self.mic_index}")
                        self.mic_index = self.get_working_microphone()
                        continue

                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    print("Listening...")
                    try:
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    except sr.WaitTimeoutError:
                        print("Listen timed out, retrying...")
                        continue

                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Recognized: {text}")

                    # --- Wake Word Logic ---
                    if not self.is_awake:
                        if "hello zentrax" in text:
                            self.is_awake = True
                            print("Zentrax is awake. Defaulting to voice mode. Say 'switch to gesture mode' to change.")
                            self.active_mode = "voice" # Reset to default mode on wake
                        else:
                            print("Zzz... (Waiting for 'Hello Zentrax')")
                        continue # Skip further command processing if not awake

                    # --- Commands when awake ---
                    if "go to sleep" in text or "deactivate" in text:
                        self.is_awake = False
                        print("Zentrax is going to sleep. Say 'Hello Zentrax' to wake me up.")
                        continue # Skip further command processing

                    if "switch to gesture mode" in text:
                        self.active_mode = "gesture"
                        print("Switched to gesture control mode.")
                        continue
                    elif "switch to voice mode" in text:
                        self.active_mode = "voice"
                        print("Switched to voice control mode.")
                        continue

                    # Only process voice commands if in voice mode
                    if self.active_mode == "voice":
                        # Handle "play music" command specifically
                        if text.startswith("play music"):
                            music_name = text.replace("play music", "").strip()
                            if music_name:
                                print(f"Executing voice command: play music '{music_name}'")
                                try:
                                    self.play_music(music_name)
                                except Exception as e:
                                    print(f"Error executing command 'play music': {e}")
                            else:
                                print("Please specify a music name after 'play music'.")
                            continue # Command handled, move to next listen cycle
                        # Handle "send whatsapp message" command specifically
                        elif text.startswith("send whatsapp message to"):
                            try:
                                # Expected format: "send whatsapp message to [contact name] saying [your message]"
                                parts = text.split("saying", 1)
                                if len(parts) == 2:
                                    contact_phrase = parts[0].replace("send whatsapp message to", "").strip()
                                    message = parts[1].strip()

                                    if contact_phrase and message:
                                        # Attempt to find the contact name in a case-insensitive manner
                                        found_contact_name = None
                                        for name_key in self.contacts:
                                            if name_key in contact_phrase: # Check if the spoken phrase contains a known contact name
                                                found_contact_name = name_key
                                                break
                                        
                                        if found_contact_name:
                                            print(f"Executing voice command: send WhatsApp to '{found_contact_name}' with message '{message}'")
                                            self.send_whatsapp_message(found_contact_name, message)
                                        else:
                                            print(f"Contact '{contact_phrase}' not recognized. Please add them to your contacts list.")
                                    else:
                                        print("Could not parse contact name or message for WhatsApp. Please use the format: 'send WhatsApp message to [contact name] saying [your message]'.")
                                else:
                                    print("Please use the format: 'send WhatsApp message to [contact name] saying [your message]'.")
                            except Exception as e:
                                print(f"Error parsing WhatsApp command: {e}\n{traceback.format_exc()}")
                            continue # Command handled, move to next listen cycle
                        else:
                            # Handle other predefined commands
                            command_executed = False
                            for command, function in self.voice_commands.items():
                                if command in text:
                                    print(f"Executing voice command: {command}")
                                    try:
                                        function()
                                    except Exception as e:
                                        print(f"Error executing command '{command}': {e}")
                                    command_executed = True
                                    break
                            if not command_executed:
                                print(f"No matching voice command found for: {text}")
                    else:
                        print(f"Currently in gesture mode. Voice commands are inactive. Recognized: {text}")

                except sr.UnknownValueError:
                    if self.is_awake:
                        print("Could not understand audio")
                    else:
                        print("Zzz... (Could not understand audio while waiting for wake word)")
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                except Exception as e:
                    print(f"Unexpected error during recognition: {e}\n{traceback.format_exc()}")

            except Exception as e:
                print(f"Error in voice recognition loop: {e}\n{traceback.format_exc()}")
                time.sleep(1)

    # ---------------- Gestures ----------------
    def process_gestures(self):
        print("Gesture recognition started. Show gestures to camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        try:
            while self.running:
                success, image = cap.read()
                if not success:
                    print("Error: Could not read from webcam")
                    break

                image = cv2.flip(image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image_rgb)

                # Display current mode on screen
                if not self.is_awake:
                    cv2.putText(image, "Zzz... Say 'Hello Zentrax'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                elif self.active_mode == "voice":
                    cv2.putText(image, "Voice Mode Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                elif self.active_mode == "gesture":
                    cv2.putText(image, "Gesture Mode Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)


                if self.is_awake and self.active_mode == "gesture":
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                            gesture = self.recognize_gesture(hand_landmarks)
                            current_time = time.time()

                            if gesture == "pointing" and current_time > self.gesture_cooldown:
                                print("Detected gesture: pointing -> starting mouse tracking for 5s")
                                self.tracking_mouse = True
                                self.tracking_end_time = current_time + 5.0
                                self.gesture_cooldown = current_time + 1.0

                            elif gesture and current_time > self.gesture_cooldown and gesture != "pointing":
                                print(f"Detected gesture: {gesture}")
                                self.execute_gesture(gesture)
                                self.gesture_cooldown = current_time + 1.0

                    if self.tracking_mouse and time.time() < self.tracking_end_time:
                        if results.multi_hand_landmarks:
                            landmarks = results.multi_hand_landmarks[0].landmark
                            index_finger = landmarks[8]
                            screen_w, screen_h = pyautogui.size()
                            x = int(index_finger.x * screen_w)
                            y = int(index_finger.y * screen_h)
                            pyautogui.moveTo(x, y)
                    elif self.tracking_mouse and time.time() >= self.tracking_end_time:
                        self.tracking_mouse = False
                        print("Mouse tracking stopped")
                elif self.is_awake and self.active_mode == "voice":
                    # If in voice mode, still show hand landmarks but don't execute gestures
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                # If not awake, just show camera feed with "Zzz..." message

                cv2.imshow('Hand Tracking', image)
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    print("Quit requested via 'q' key.")
                    self.running = False
                    break

        except KeyboardInterrupt:
            print("KeyboardInterrupt received - stopping.")
            self.running = False
        except Exception as e:
            print(f"Exception in gesture loop: {e}\n{traceback.format_exc()}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    # ---------------- Gesture recognition heuristics ----------------
    def recognize_gesture(self, landmarks):
        points = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

        if self.is_open_palm(points):
            return "open_palm"
        if self.is_closed_fist(points):
            return "closed_fist"
        if self.is_pointing(points):
            return "pointing"
        if self.is_thumbs_up(points):
            return "thumbs_up"
        if self.is_thumbs_down(points):
            return "thumbs_down"
        if self.is_swipe_left(points):
            return "swipe_left"
        if self.is_swipe_right(points):
            return "swipe_right"
        return None

    def is_open_palm(self, points):
        fingertips = [8, 12, 16, 20]
        mcps = [5, 9, 13, 17]
        for tip, mcp in zip(fingertips, mcps):
            if points[tip][1] >= points[mcp][1]:
                return False
        return points[4][0] > points[3][0]

    def is_closed_fist(self, points):
        fingertips = [8, 12, 16, 20]
        mcps = [5, 9, 13, 17]
        return all(points[tip][1] > points[mcp][1] for tip, mcp in zip(fingertips, mcps))

    def is_pointing(self, points):
        if points[8][1] >= points[5][1]:
            return False
        for tip, mcp in zip([12, 16, 20], [9, 13, 17]):
            if points[tip][1] < points[mcp][1]:
                return False
        return True

    def is_thumbs_up(self, points):
        if points[4][1] >= points[3][1]:
            return False
        return all(points[tip][1] > points[mcp][1] for tip, mcp in zip([8, 12, 16, 20], [5, 9, 13, 17]))

    def is_thumbs_down(self, points):
        if points[4][1] <= points[3][1]:
            return False
        return all(points[tip][1] > points[mcp][1] for tip, mcp in zip([8, 12, 16, 20], [5, 9, 13, 17]))

    def is_swipe_left(self, points):
        return points[0][0] - points[5][0] > 0.12

    def is_swipe_right(self, points):
        return points[5][0] - points[0][0] > 0.12

    def execute_gesture(self, gesture):
        try:
            if gesture == "open_palm":
                print("Executing: maximize")
                self.maximize_window()
            elif gesture == "closed_fist":
                print("Executing: minimize")
                self.minimize_window()
            elif gesture == "thumbs_up":
                print("Executing: volume up")
                self.volume_up()
            elif gesture == "thumbs_down":
                print("Executing: volume down")
                self.volume_down()
            elif gesture == "swipe_left":
                print("Executing: Swipe Left (Alt+Left)")
                pyautogui.hotkey('alt', 'left')
            elif gesture == "swipe_right":
                print("Executing: Swipe Right (Alt+Right)")
                pyautogui.hotkey('alt', 'right')
        except Exception as e:
            print(f"Error executing gesture '{gesture}': {e}\n{traceback.format_exc()}")

    # ---------- System Control Functions ----------
    def open_browser(self):
        try:
            os.system("start https://www.google.com")
        except Exception as e:
            print(f"Failed to open browser: {e}")

    def close_window(self):
        pyautogui.hotkey('alt', 'f4')

    def minimize_window(self):
        pyautogui.hotkey('win', 'down')

    def maximize_window(self):
        pyautogui.hotkey('win', 'up')

    def volume_up(self):
        pyautogui.press('volumeup')

    def volume_down(self):
        pyautogui.press('volumedown')

    def scroll_up(self):
        pyautogui.scroll(10)

    def scroll_down(self):
        pyautogui.scroll(-10)

    def take_screenshot(self):
        try:
            screenshot = pyautogui.screenshot()
            path = os.path.join(os.path.expanduser("~"), "Desktop", f"screenshot_{int(time.time())}.png")
            screenshot.save(path)
            print(f"Screenshot saved to {path}")
        except Exception as e:
            print(f"Failed to take screenshot: {e}")

    def exit_program(self):
        print("Exit command received. Stopping.")
        self.running = False

    def play_music(self, music_name):
        """Opens and plays a YouTube video for the given music name."""
        print(f"Attempting to play music: {music_name}")
        try:
            pywhatkit.playonyt(music_name)
            print(f"Playing '{music_name}' on YouTube.")
        except Exception as e:
            print(f"Failed to play music '{music_name}': {e}")

    def send_whatsapp_message(self, contact_name, message):
        """Sends a WhatsApp message to a contact using pywhatkit."""
        phone_number = self.contacts.get(contact_name.lower())
        if not phone_number:
            print(f"Contact '{contact_name}' not found in your contacts list. Please add them.")
            return

        print(f"Scheduling WhatsApp message to {contact_name} ({phone_number}): '{message}'")
        try:
            now = datetime.now()
            send_hour = now.hour
            send_minute = now.minute + 1
            if send_minute >= 60:
                send_minute -= 60
                send_hour = (send_hour + 1) % 24

            pywhatkit.sendwhatmsg(phone_number, message, send_hour, send_minute, wait_time=25, tab_close=False, close_time=5)
            print(f"WhatsApp message scheduled to be sent to {contact_name} at {send_hour:02d}:{send_minute:02d}.")
            print("Please ensure WhatsApp Web is logged in on your default browser.")
            print("The browser tab will remain open for you to verify the message was sent.")
        except Exception as e:
            print(f"Failed to send WhatsApp message to {contact_name}: {e}")
            print("Common issues: WhatsApp Web not logged in, incorrect phone number format (include country code), or internet connectivity.")

    def run(self):
        print("Starting Voice and Gesture Control")
        voice_thread = threading.Thread(target=self.listen_for_commands, name="VoiceThread")
        voice_thread.daemon = True
        voice_thread.start()
        try:
            self.process_gestures()
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught in main run.")
            self.running = False
        finally:
            self.running = False
            voice_thread.join(timeout=2)
            print("Shutdown complete.")

if __name__ == "__main__":
    controller = VoiceGestureControl()
    controller.run()

import whisper
import torch
import sounddevice as sd
import numpy as np
import pyttsx3
from scipy.io.wavfile import write
from threading import Thread, Event
from queue import Queue

class AudioHandler:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.engine = None
        self.sample_rate = 16000
        self.is_speaking = Event()
        self.queue = Queue()
        self.speech_thread = None
        self.initialize_engine()
        self.start_speech_thread()

    def initialize_engine(self):
        """Initialize the text-to-speech engine"""
        try:
            self.engine = pyttsx3.init()
            # Configure the engine
            self.engine.setProperty('rate', 150)    # Speed of speech
            self.engine.setProperty('volume', 1.0)  # Volume level
            
            # Test if the engine works
            voices = self.engine.getProperty('voices')
            if voices:
                self.engine.setProperty('voice', voices[0].id)  # Set the first available voice
            
            # Try a test speech
            print("Initializing text-to-speech engine...")
            self.engine.say("Text to speech engine initialized")
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error initializing speech engine: {e}")

    def start_speech_thread(self):
        """Start the speech processing thread"""
        if self.speech_thread is None or not self.speech_thread.is_alive():
            self.speech_thread = Thread(target=self._process_speech_queue, daemon=True)
            self.speech_thread.start()

    def _process_speech_queue(self):
        """Process speech requests from the queue"""
        while True:
            text = self.queue.get()
            if text is None:
                break
                
            try:
                self.is_speaking.set()
                
                # Reinitialize the engine for each new speech request
                if self.engine is not None:
                    self.engine.stop()
                self.initialize_engine()
                
                self.engine.say(text)
                self.engine.runAndWait()
                
            except Exception as e:
                print(f"Error in speech processing: {e}")
                
            finally:
                self.is_speaking.clear()
                self.queue.task_done()

    def speak(self, text):
        """Add text to the speech queue"""
        if text:
            print(f"Adding to speech queue: {text}")  # Debug print
            # Clear any pending items in the queue
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except:
                    break
            
            # Add new text to queue
            self.queue.put(text)
            
            # Ensure speech thread is running
            self.start_speech_thread()

    def record_audio(self, duration=5, filename="temp_recording.wav"):
        """Record audio for a specified duration"""
        print("Recording...")
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        audio_int16 = (recording * 32767).astype(np.int16)
        write(filename, self.sample_rate, audio_int16)
        print("Recording finished!")
        return filename

    def transcribe_audio(self, audio_file):
        """Transcribe audio using Whisper"""
        try:
            result = self.model.transcribe(audio_file)
            return result["text"].strip()
        except Exception as e:
            print(f"Error in transcription: {e}")
            return ""

    def _del_(self):
        """Cleanup when the object is destroyed"""
        if self.queue is not None:
            self.queue.put(None)  # Signal the thread to exit
        if self.engine is not None:
            try:
                self.engine.stop()
            except:
                pass
class ExerciseManager:
    def __init__(self):
        self.exercises = {
            'breathing': [
                {
                    'name': 'Deep Breathing',
                    'instruction': 'Take a deep breath in through your nose for 4 counts, hold for 4, then exhale for 4.',
                    'duration': 5
                },
                {
                    'name': 'Diaphragmatic Breathing',
                    'instruction': 'Place one hand on your chest and one on your belly. Breathe so that your belly expands more than your chest.',
                    'duration': 5
                }
            ],
            'articulation': [
                {
                    'name': 'Tongue Twisters',
                    'instruction': 'Practice saying: Peter Piper picked a peck of pickled peppers',
                    'duration': 5
                }
            ],
            'tongue': [
                {
                    'name': 'Tongue Exercise',
                    'instruction': 'Stick out your tongue and move it up and down, then side to side.',
                    'duration': 5
                }
            ],
            'naming': [
                {
                    'name': 'Object Naming',
                    'instruction': 'Name common objects in your surroundings.',
                    'duration': 5
                }
            ],
            'sentence': [
                {
                    'name': 'Sentence Formation',
                    'instruction': 'Create complete sentences using given words.',
                    'duration': 5
                }
            ],
            'speech_sounds': [
                {
                    'name': 'S Sound Practice',
                    'instruction': 'Practice the "S" sound with these words: Sun, Snake, Star, Smile',
                    'duration': 5,
                    'words': ['sun', 'snake', 'star', 'smile']
                },
                {
                    'name': 'R Sound Practice',
                    'instruction': 'Practice the "R" sound with these words: Red, Rain, Run, River',
                    'duration': 5,
                    'words': ['red', 'rain', 'run', 'river']
                },
                {
                    'name': 'TH Sound Practice',
                    'instruction': 'Practice the "TH" sound with these words: Think, The, That, Three',
                    'duration': 5,
                    'words': ['think', 'the', 'that', 'three']
                }
            ]
        }

    def get_exercise_types(self):
        """Return all available exercise types"""
        return list(self.exercises.keys())  # Make sure this returns the list of keys

    def get_exercises_for_type(self, exercise_type):
        """Return exercises for a specific type"""
        return self.exercises.get(exercise_type, [])
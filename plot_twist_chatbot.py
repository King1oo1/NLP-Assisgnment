import nltk
from textblob import TextBlob
import re
import random
import json
import os
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import speech_recognition as sr
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import threading
import queue
import time
from datetime import datetime
from colorama import Fore, Style, init
from pathlib import Path

# Initialize colorama
init(autoreset=True)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ChatbotBase:
    def __init__(self, name="Chatbot"):
        self.name = name
        self.conversation_is_active = True
        self.conversation_mode = "text"  # "text" or "voice"
        
    def greeting(self):
        print(f'{Fore.CYAN}Hello I am {self.name}{Style.RESET_ALL}')
    
    def farewell(self):
        print(f'{Fore.YELLOW}Goodbye!{Style.RESET_ALL}')
    
    def receive_input(self):
        """Receive input based on current mode"""
        if self.conversation_mode == "voice":
            return self.receive_voice_input()
        else:
            return input(f"{Fore.GREEN}> {Style.RESET_ALL}")
    
    def switch_mode(self, mode):
        """Switch between text and voice mode"""
        if mode in ["text", "voice"]:
            self.conversation_mode = mode
            return f"Switched to {mode} mode."
        return "Invalid mode. Please choose 'text' or 'voice'."

class PlotTwistChatbot(ChatbotBase):
    def __init__(self, name="Plot Twist Storyteller"):
        super().__init__(name)
        
        # Story state management
        self.story_state = {
            "genre": "adventure",
            "mood": "neutral",
            "characters": [],
            "locations": [],
            "items": [],
            "story_arc": [],
            "preferences": {},
            "voice_emotion": "neutral"
        }
        
        # Story branch management for backtracking
        self.story_branches = []
        self.current_branch_index = -1
        self.conversation_history = []
        self.user_style_preferences = {}
        
        # Voice recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.microphone_available = False
        self.whisper_model = None
        self.recognition_confidence_threshold = 0.6
        
        # BART model for enhanced storytelling
        self.bart_model = None
        self.bart_tokenizer = None
        self.initialize_bart_model()
        
        # Voice recording settings
        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        
        # Adaptive storytelling parameters
        self.story_coherence_score = 0
        self.user_engagement_level = 0
        
        # Create saved stories folder if it doesn't exist
        self.saved_stories_folder = Path("saved_stories")
        self.saved_stories_folder.mkdir(exist_ok=True)
        
        # Initialize voice recognition system
        self.check_microphone_availability()
    
    def initialize_bart_model(self):
        """Initialize BART model for enhanced story generation"""
        try:
            print(f"{Fore.YELLOW}Loading BART model for enhanced storytelling...{Style.RESET_ALL}")
            self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
            self.bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
            print(f"{Fore.GREEN}BART model loaded successfully!{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Could not load BART model: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Using fallback story generation.{Style.RESET_ALL}")
    
    def check_microphone_availability(self):
        """Check if microphone is available without initializing it"""
        try:
            print(f"{Fore.CYAN}Checking microphone availability...{Style.RESET_ALL}")
            
            # List available microphones
            mic_list = sr.Microphone.list_microphone_names()
            if mic_list:
                print(f"{Fore.GREEN}✓ Microphone detected: {mic_list[0]}{Style.RESET_ALL}")
                self.microphone_available = True
            else:
                print(f"{Fore.YELLOW}⚠ No microphones detected{Style.RESET_ALL}")
                self.microphone_available = False
                
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Could not check microphone: {e}{Style.RESET_ALL}")
            self.microphone_available = False
    
    def initialize_microphone(self):
        """Initialize microphone when needed"""
        try:
            print(f"{Fore.CYAN}Initializing microphone...{Style.RESET_ALL}")
            self.microphone = sr.Microphone()
            
            # Quick test to ensure microphone works
            with self.microphone as source:
                print(f"{Fore.GREEN}✓ Microphone ready!{Style.RESET_ALL}")
                return True
                
        except Exception as e:
            print(f"{Fore.RED}✗ Failed to initialize microphone: {e}{Style.RESET_ALL}")
            self.microphone = None
            self.microphone_available = False
            return False
    
    def receive_voice_input(self):
        """Receive voice input with error correction and confidence feedback"""
        
        # Check if microphone is available
        if not self.microphone_available:
            print(f"{Fore.YELLOW}Voice mode not available. Please use text input.{Style.RESET_ALL}")
            return input(f"{Fore.GREEN}> {Style.RESET_ALL}")
        
        # Initialize microphone if not already done
        if self.microphone is None:
            if not self.initialize_microphone():
                print(f"{Fore.YELLOW}Falling back to text input.{Style.RESET_ALL}")
                self.conversation_mode = "text"
                return input(f"{Fore.GREEN}> {Style.RESET_ALL}")
        
        print(f"{Fore.CYAN}🎤 Listening... (speak now){Style.RESET_ALL}")
        
        try:
            # Use the microphone context manager properly
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen with timeout
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
                # Try multiple recognition methods
                recognized_text = None
                confidence = 0
                
                # Method 1: Google Speech Recognition
                try:
                    recognized_text = self.recognizer.recognize_google(audio)
                    confidence = 0.7
                    print(f"{Fore.GREEN}✓ Google recognition successful{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.YELLOW}⚠ Google recognition failed: {e}{Style.RESET_ALL}")
                
                # Method 2: Whisper fallback (only if Google fails)
                if not recognized_text:
                    try:
                        print(f"{Fore.CYAN}Trying Whisper recognition...{Style.RESET_ALL}")
                        if self.whisper_model is None:
                            print(f"{Fore.YELLOW}Loading Whisper model...{Style.RESET_ALL}")
                            self.whisper_model = whisper.load_model("base")
                        
                        # Save audio to temp file for whisper
                        temp_file = "temp_audio.wav"
                        with open(temp_file, "wb") as f:
                            f.write(audio.get_wav_data())
                        
                        result = self.whisper_model.transcribe(temp_file)
                        recognized_text = result["text"].strip()
                        confidence = 0.6
                        
                        # Clean up temp file
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            
                        print(f"{Fore.GREEN}✓ Whisper recognition successful{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}✗ Whisper failed: {e}{Style.RESET_ALL}")
                
                if recognized_text and len(recognized_text.strip()) > 0:
                    # Extract voice emotion
                    emotion = self.extract_voice_emotion(audio)
                    self.story_state["voice_emotion"] = emotion
                    
                    # Show confidence feedback
                    confidence_bar = self.create_confidence_bar(confidence)
                    print(f"{Fore.GREEN}✅ Recognized: \"{recognized_text}\"")
                    print(f"{Fore.BLUE}Confidence: {confidence_bar} ({confidence:.2%}){Style.RESET_ALL}")
                    print(f"{Fore.CYAN}Voice emotion: {emotion}{Style.RESET_ALL}")
                    
                    # Low confidence handling
                    if confidence < self.recognition_confidence_threshold:
                        print(f"{Fore.YELLOW}⚠ Low confidence. Say 're-record' to try again.{Style.RESET_ALL}")
                        if "re-record" in recognized_text.lower():
                            return self.receive_voice_input()
                    
                    return recognized_text
                else:
                    print(f"{Fore.RED}❌ Could not understand audio. Please try again or type your input.{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Falling back to text input.{Style.RESET_ALL}")
                    return input(f"{Fore.GREEN}> {Style.RESET_ALL}")
                    
        except sr.WaitTimeoutError:
            print(f"{Fore.YELLOW}⏰ No speech detected. Please try again.{Style.RESET_ALL}")
            return ""
        except sr.UnknownValueError:
            print(f"{Fore.RED}❌ Could not understand audio.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Falling back to text input.{Style.RESET_ALL}")
            return input(f"{Fore.GREEN}> {Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}✗ Voice recognition error: {str(e)[:100]}...{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Switching to text mode.{Style.RESET_ALL}")
            self.conversation_mode = "text"
            return input(f"{Fore.GREEN}> {Style.RESET_ALL}")
    
    def extract_voice_emotion(self, audio):
        """Extract emotion from voice characteristics"""
        try:
            # Analyze audio features
            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
            
            # Simple feature extraction
            volume = np.abs(audio_data).mean()
            zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
            
            # Determine emotion based on features
            if volume > 10000 and zero_crossings > 1000:
                return "excited"
            elif volume < 3000:
                return "calm"
            elif zero_crossings > 1500:
                return "nervous"
            else:
                return "neutral"
        except:
            return "neutral"
    
    def create_confidence_bar(self, confidence):
        """Create visual confidence bar"""
        bars = 20
        filled = int(confidence * bars)
        return f"[{'█' * filled}{'░' * (bars - filled)}]"
    
    def greeting(self):
        greeting_msg = f"""
{Fore.CYAN}🎭🎤 Welcome to PLOT TWIST - Enhanced Interactive Storyteller{Style.RESET_ALL}

I'll be your Game Master for an interactive adventure with new features!

{Fore.MAGENTA}Features:{Style.RESET_ALL}
• Text/Voice Mode : Speak or Write your text (with emotion detection!)
• Branch Backtracking: Revisit previous choices
• Personalized Stories: Learns your preferences
• Enhanced Narratives: More coherent storytelling
• Save/Load System: Continue your adventures later!

{Fore.CYAN}Commands:{Style.RESET_ALL}
• Type/Say: "{Fore.GREEN}go back{Style.RESET_ALL}" to revisit previous choice
• Type/Say: "{Fore.BLUE}save story{Style.RESET_ALL}" to save current progress
• Type/Say: "{Fore.GREEN}load story{Style.RESET_ALL}" to load a saved adventure
• Type/Say: "{Fore.YELLOW}explain{Style.RESET_ALL}" to understand story logic
• Type/Say: "{Fore.GREEN}switch to voice{Style.RESET_ALL}" or "{Fore.GREEN}switch to text{Style.RESET_ALL}"
• Type/Say: "{Fore.RED}quit{Style.RESET_ALL}" to exit

{Fore.WHITE}Where would you like to begin?{Style.RESET_ALL}
        """
        print(greeting_msg)
        
        # Ask if user wants to load a saved story
        load_choice = input(f"{Fore.CYAN}Load a saved story? (yes/no) [default: no]: {Style.RESET_ALL}").lower()
        if load_choice == "yes":
            result = self.load_story_progress()
            if result:
                print(result["response"])
            return
        
        # Ask for mode preference only if microphone is available
        if self.microphone_available:
            mode_choice = input(f"{Fore.CYAN}Start with voice mode? (yes/no) [default: no]: {Style.RESET_ALL}").lower()
            if mode_choice == "yes":
                self.conversation_mode = "voice"
                print(f"{Fore.GREEN}✓ Voice mode activated! Say 'switch to text' to change.{Style.RESET_ALL}")
            else:
                print(f"{Fore.CYAN}Starting in text mode. Say 'switch to voice' to change.{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠ Voice mode not available (no microphone detected). Starting in text mode.{Style.RESET_ALL}")
    
    def process_input(self, user_input):
        """Process user input using NLP techniques"""
        
        # Check for special commands first
        command_response = self.process_special_commands(user_input)
        if command_response:
            return command_response
        
        # Process regular input
        processed = {
            "raw_text": user_input,
            "sentiment": self.analyze_sentiment(user_input),
            "keywords": self.extract_keywords(user_input),
            "action_type": self.classify_input(user_input),
            "voice_emotion": self.story_state["voice_emotion"]
        }
        
        # Update user preferences based on input
        self.update_user_preferences(processed)
        
        # Store for backtracking
        self.save_story_branch()
        
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "input": processed,
            "mode": self.conversation_mode
        })
        
        return processed
    
    def process_special_commands(self, user_input):
        """Process voice/text commands"""
        lower_input = user_input.lower()
        
        # Mode switching
        if "switch to voice" in lower_input:
            if self.microphone_available:
                if self.microphone is None:
                    if not self.initialize_microphone():
                        return {"command": "mode_switch", "response": f"{Fore.RED}✗ Could not initialize microphone. Staying in text mode.{Style.RESET_ALL}"}
                self.conversation_mode = "voice"
                return {"command": "mode_switch", "response": f"{Fore.GREEN}✓ Switched to voice mode. Speak your commands!{Style.RESET_ALL}"}
            else:
                return {"command": "mode_switch", "response": f"{Fore.YELLOW}⚠ Voice mode not available. No microphone detected.{Style.RESET_ALL}"}
        
        elif "switch to text" in lower_input:
            self.conversation_mode = "text"
            return {"command": "mode_switch", "response": f"{Fore.GREEN}✓ Switched to text mode. Type your commands!{Style.RESET_ALL}"}
        
        # Backtracking
        elif any(cmd in lower_input for cmd in ["go back", "previous", "backtrack", "undo", "last choice"]):
            return self.handle_backtracking()
        
        # Story saving
        elif any(cmd in lower_input for cmd in ["save story", "save progress", "save game"]):
            return self.save_story_progress()
        
        # Story loading
        elif any(cmd in lower_input for cmd in ["load story", "load progress", "load game", "continue story"]):
            return self.load_story_progress()
        
        # List saved stories
        elif any(cmd in lower_input for cmd in ["list stories", "show saves", "saved games"]):
            return self.list_saved_stories()
        
        # Explain logic
        elif any(cmd in lower_input for cmd in ["explain", "why", "how", "logic"]):
            return self.explain_story_logic()
        
        # Voice emotion trigger
        elif any(emotion in lower_input for emotion in ["surprise", "excited", "shocked"]):
            self.story_state["voice_emotion"] = "excited"
            return {"command": "emotion_trigger", "response": "I sense excitement! Let's add a surprise plot twist..."}
        
        return None
    
    def handle_backtracking(self):
        """Handle story branch backtracking"""
        if len(self.story_branches) > 1:
            # Go back one branch
            self.story_branches.pop()
            previous_branch = self.story_branches[-1]
            self.story_state = previous_branch["story_state"]
            
            response = f"""
{Fore.YELLOW}⏪ Backtracking to previous choice...{Style.RESET_ALL}

{Fore.CYAN}You are now at:{Style.RESET_ALL}
Location: {previous_branch['location']}
Items: {', '.join(previous_branch['items'])}

What would you like to do differently?
            """
            return {"command": "backtrack", "response": response}
        else:
            return {"command": "backtrack", "response": "No previous branches to go back to."}
    
    def save_story_branch(self):
        """Save current story state as a branch"""
        branch = {
            "timestamp": datetime.now().isoformat(),
            "story_state": self.story_state.copy(),
            "location": self.story_state["locations"][-1] if self.story_state["locations"] else "Beginning",
            "items": self.story_state["items"].copy(),
            "story_arc_snapshot": self.story_state["story_arc"][-3:] if len(self.story_state["story_arc"]) >= 3 else self.story_state["story_arc"]
        }
        self.story_branches.append(branch)
    
    def save_story_progress(self):
        """Save current story progress to file in saved_stories folder"""
        filename = f"story_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.saved_stories_folder / filename
        
        save_data = {
            "story_state": self.story_state,
            "branches": self.story_branches,
            "preferences": self.user_style_preferences,
            "history": self.conversation_history[-10:],
            "saved_at": datetime.now().isoformat(),
            "locations": self.story_state["locations"],
            "items": self.story_state["items"],
            "story_summary": self.generate_story_summary()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
            response = f"""
{Fore.GREEN}✅ Story saved successfully!{Style.RESET_ALL}
{Fore.CYAN}File: {filename}{Style.RESET_ALL}
{Fore.CYAN}Path: {filepath}{Style.RESET_ALL}
{Fore.YELLOW}Story Summary:{Style.RESET_ALL}
{self.generate_story_summary()}
            """
        except Exception as e:
            response = f"{Fore.RED}❌ Error saving story: {e}{Style.RESET_ALL}"
        
        return {"command": "save", "response": response}
    
    def load_story_progress(self, filename=None):
        """Load story progress from a saved file"""
        # List available saves if no filename provided
        if not filename:
            saved_files = list(self.saved_stories_folder.glob("*.json"))
            
            if not saved_files:
                return {"command": "load", "response": f"{Fore.YELLOW}No saved stories found in '{self.saved_stories_folder}'{Style.RESET_ALL}"}
            
            # Sort by modification time (newest first)
            saved_files.sort(key=os.path.getmtime, reverse=True)
            
            print(f"{Fore.CYAN}📚 Available saved stories:{Style.RESET_ALL}")
            for i, file in enumerate(saved_files[:10], 1):  # Show only first 10
                mtime = datetime.fromtimestamp(os.path.getmtime(file))
                print(f"{Fore.YELLOW}{i}. {file.name}{Style.RESET_ALL} - Saved: {mtime.strftime('%Y-%m-%d %H:%M')}")
            
            try:
                choice = input(f"{Fore.GREEN}Enter number to load (or 'cancel'): {Style.RESET_ALL}")
                if choice.lower() == 'cancel':
                    return {"command": "load", "response": "Load cancelled."}
                
                idx = int(choice) - 1
                if 0 <= idx < len(saved_files):
                    filename = saved_files[idx]
                else:
                    return {"command": "load", "response": f"{Fore.RED}Invalid choice.{Style.RESET_ALL}"}
            except ValueError:
                return {"command": "load", "response": f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}"}
        
        filepath = self.saved_stories_folder / filename if isinstance(filename, str) else filename
        
        try:
            with open(filepath, 'r') as f:
                save_data = json.load(f)
            
            # Restore all saved data
            self.story_state = save_data.get("story_state", self.story_state)
            self.story_branches = save_data.get("branches", [])
            self.user_style_preferences = save_data.get("preferences", {})
            self.conversation_history = save_data.get("history", [])
            
            # Update coherence score
            self.story_coherence_score = self.calculate_coherence_score()
            
            response = f"""
{Fore.GREEN}✅ Story loaded successfully!{Style.RESET_ALL}
{Fore.CYAN}File: {filepath.name}{Style.RESET_ALL}
{Fore.YELLOW}Story Summary:{Style.RESET_ALL}
{save_data.get('story_summary', 'No summary available')}
{Fore.CYAN}Loaded Data:{Style.RESET_ALL}
• Locations visited: {len(save_data.get('locations', []))}
• Items collected: {len(save_data.get('items', []))}
• Story branches: {len(self.story_branches)}
• Saved at: {save_data.get('saved_at', 'Unknown')}
            """
            return {"command": "load", "response": response}
            
        except FileNotFoundError:
            return {"command": "load", "response": f"{Fore.RED}❌ File '{filename}' not found.{Style.RESET_ALL}"}
        except json.JSONDecodeError:
            return {"command": "load", "response": f"{Fore.RED}❌ File is corrupted or invalid JSON.{Style.RESET_ALL}"}
        except Exception as e:
            return {"command": "load", "response": f"{Fore.RED}❌ Error loading story: {e}{Style.RESET_ALL}"}
    
    def list_saved_stories(self):
        """List all saved stories"""
        saved_files = list(self.saved_stories_folder.glob("*.json"))
        
        if not saved_files:
            return {"command": "list", "response": f"{Fore.YELLOW}No saved stories found in '{self.saved_stories_folder}'{Style.RESET_ALL}"}
        
        # Sort by modification time (newest first)
        saved_files.sort(key=os.path.getmtime, reverse=True)
        
        response_lines = [f"{Fore.CYAN}📚 Saved Stories ({len(saved_files)} found):{Style.RESET_ALL}"]
        
        for i, file in enumerate(saved_files[:15], 1):  # Show only first 15
            # Try to get basic info from the save file
            try:
                with open(file, 'r') as f:
                    save_data = json.load(f)
                locations = len(save_data.get('locations', []))
                items = len(save_data.get('items', []))
                summary = save_data.get('story_summary', 'No summary')[:50] + "..."
                mtime = datetime.fromtimestamp(os.path.getmtime(file))
                
                response_lines.append(
                    f"{Fore.YELLOW}{i}. {file.name}{Style.RESET_ALL}\n"
                    f"   {Fore.GREEN}Saved: {mtime.strftime('%Y-%m-%d %H:%M')} | "
                    f"Locations: {locations} | Items: {items}{Style.RESET_ALL}\n"
                    f"   {Fore.CYAN}{summary}{Style.RESET_ALL}"
                )
            except:
                mtime = datetime.fromtimestamp(os.path.getmtime(file))
                response_lines.append(f"{Fore.YELLOW}{i}. {file.name} (Saved: {mtime.strftime('%Y-%m-%d %H:%M')}){Style.RESET_ALL}")
        
        response_lines.append(f"\n{Fore.GREEN}Say 'load story' followed by the number to load.{Style.RESET_ALL}")
        
        return {"command": "list", "response": "\n".join(response_lines)}
    
    def generate_story_summary(self):
        """Generate a summary of the current story"""
        if not self.story_state["locations"]:
            return "Your adventure is just beginning!"
        
        summary = f"""Adventure Summary:
• Genre: {self.story_state['genre']}
• Mood: {self.story_state['mood']}
• Locations visited: {len(set(self.story_state['locations']))}
• Items collected: {len(self.story_state['items'])}
• Recent location: {self.story_state['locations'][-1] if self.story_state['locations'] else 'Beginning'}
• Key items: {', '.join(self.story_state['items'][-3:]) if self.story_state['items'] else 'None'}
• Story coherence: {self.story_coherence_score:.1f}/10"""
        
        return summary
    
    def explain_story_logic(self):
        """Explain why the story changed based on user input"""
        if not self.conversation_history:
            return {"command": "explain", "response": "The story is just beginning!"}
        
        last_input = self.conversation_history[-1]["input"]
        keywords = last_input.get("keywords", [])
        sentiment = last_input.get("sentiment", "neutral")
        emotion = last_input.get("voice_emotion", "neutral")
        
        explanations = []
        
        if keywords:
            for keyword in keywords:
                if keyword in ['forest', 'castle', 'cave']:
                    explanations.append(f"Added {keyword} location because you mentioned it.")
                elif keyword in ['sword', 'key', 'treasure']:
                    explanations.append(f"Included {keyword} because you showed interest.")
        
        if sentiment != "neutral":
            explanations.append(f"Adjusted tone to be more {sentiment} based on your input.")
        
        if emotion != "neutral":
            explanations.append(f"Added elements matching your {emotion} voice tone.")
        
        if not explanations:
            explanations.append("The story continues based on your choices.")
        
        explanation_text = f"""
{Fore.CYAN}📖 Story Logic Explanation:{Style.RESET_ALL}

{Fore.YELLOW}Why the story changed:{Style.RESET_ALL}
{chr(10).join(f'• {exp}' for exp in explanations)}

{Fore.GREEN}Current narrative coherence: {self.calculate_coherence_score():.1f}/10{Style.RESET_ALL}
        """
        
        return {"command": "explain", "response": explanation_text}
    
    def analyze_sentiment(self, text):
        """Enhanced sentiment analysis"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Combine text sentiment with voice emotion
            if hasattr(self, 'story_state') and self.story_state.get("voice_emotion"):
                voice_emotion = self.story_state["voice_emotion"]
                if voice_emotion == "excited" and polarity > 0:
                    return "very_positive"
                elif voice_emotion == "nervous" and polarity < 0:
                    return "very_negative"
            
            if polarity > 0.3:
                return "very_positive"
            elif polarity > 0.1:
                return "positive"
            elif polarity < -0.3:
                return "very_negative"
            elif polarity < -0.1:
                return "negative"
            else:
                return "neutral"
        except:
            return "neutral"
    
    def extract_keywords(self, text):
        """Extract important keywords"""
        story_keywords = [
            'forest', 'castle', 'cave', 'river', 'mountain', 'village', 'town',
            'sword', 'key', 'map', 'treasure', 'gold', 'dragon', 'wizard', 'monster',
            'fight', 'attack', 'run', 'hide', 'explore', 'search', 'look',
            'take', 'get', 'grab', 'open', 'enter', 'go', 'move', 'walk',
            'magic', 'spell', 'secret', 'mystery', 'danger', 'safe', 'help'
        ]
        
        found_keywords = []
        words = re.findall(r'\b\w+\b', text.lower())
        
        for word in words:
            if word in story_keywords:
                found_keywords.append(word)
        
        return list(set(found_keywords))
    
    def classify_input(self, text):
        """Classify the type of user input"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['go', 'move', 'walk', 'enter', 'explore', 'travel']):
            return "movement"
        elif any(word in text_lower for word in ['attack', 'fight', 'kill', 'defeat', 'hit', 'battle']):
            return "combat"
        elif any(word in text_lower for word in ['take', 'get', 'grab', 'pick', 'collect']):
            return "item_interaction"
        elif any(word in text_lower for word in ['talk', 'ask', 'tell', 'say', 'speak']):
            return "dialogue"
        else:
            return "narrative"
    
    def update_user_preferences(self, processed_input):
        """Learn and update user style preferences"""
        action_type = processed_input["action_type"]
        sentiment = processed_input["sentiment"]
        
        # Update preferences based on actions
        if action_type not in self.user_style_preferences:
            self.user_style_preferences[action_type] = 0
        
        self.user_style_preferences[action_type] += 1
        
        # Update mood preference
        if "mood_preference" not in self.user_style_preferences:
            self.user_style_preferences["mood_preference"] = {}
        
        if sentiment not in self.user_style_preferences["mood_preference"]:
            self.user_style_preferences["mood_preference"][sentiment] = 0
        
        self.user_style_preferences["mood_preference"][sentiment] += 1
    
    def generate_response(self, processed_input):
        """Generate story response based on input"""
        
        # Handle special commands
        if processed_input.get("command"):
            return processed_input["response"]
        
        # Generate base response based on action type
        if processed_input["action_type"] == "movement":
            base_response = self.generate_movement_response()
        elif processed_input["action_type"] == "combat":
            base_response = self.generate_combat_response()
        elif processed_input["action_type"] == "item_interaction":
            base_response = self.generate_item_response()
        else:
            base_response = self.generate_narrative_response()
        
        # Enhance with BART if available
        if self.bart_model and random.random() > 0.3:
            enhanced_response = self.enhance_with_bart(base_response, processed_input)
        else:
            enhanced_response = base_response
        
        # Adjust tone based on sentiment and voice emotion
        enhanced_response = self.adjust_tone(enhanced_response, processed_input)
        
        # Add personalized elements
        enhanced_response = self.add_personalized_elements(enhanced_response)
        
        # Add voice emotion plot twists
        if processed_input.get("voice_emotion") == "excited":
            enhanced_response = self.add_surprise_plot_twist(enhanced_response)
        
        # Store story progression
        self.story_state["story_arc"].append(enhanced_response[:150])
        
        # Update coherence score
        self.story_coherence_score = self.calculate_coherence_score()
        
        # Show real-time feedback occasionally
        if random.random() > 0.7:
            feedback = self.generate_real_time_feedback(processed_input)
            enhanced_response = f"{enhanced_response}\n\n{Fore.CYAN}💡 {feedback}{Style.RESET_ALL}"
        
        return enhanced_response
    
    def enhance_with_bart(self, base_response, processed_input):
        """Use BART model to enhance story coherence"""
        try:
            # Prepare prompt for BART
            context = " ".join(self.story_state["story_arc"][-3:]) if len(self.story_state["story_arc"]) >= 3 else ""
            prompt = f"Continue the story: {context} {processed_input['raw_text']}"
            
            # Tokenize and generate
            inputs = self.bart_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = self.bart_model.generate(
                inputs['input_ids'], 
                max_length=150,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode the generated text
            bart_response = self.bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Blend BART response with base response
            if bart_response and len(bart_response) > 20:
                return f"{bart_response} {base_response}"
            else:
                return base_response
                
        except Exception as e:
            print(f"{Fore.YELLOW}BART enhancement failed: {e}{Style.RESET_ALL}")
            return base_response
    
    def generate_movement_response(self):
        """Generate movement responses"""
        locations = [
            "a dark, ancient forest where sunlight barely penetrates the canopy",
            "a crumbling stone castle atop a misty hill",
            "a deep, echoing cave that seems to breathe with ancient magic",
            "a rushing river that carves through the landscape",
            "a quiet village with smoke rising from chimneys",
            "a mysterious temple covered in strange symbols"
        ]
        events = [
            "You journey onward and discover",
            "After some time, you arrive at",
            "Your path leads you to",
            "Through the mist, you see"
        ]
        
        selected_location = random.choice(locations)
        self.story_state["locations"].append(selected_location.split()[2] if len(selected_location.split()) > 2 else selected_location.split()[0])
        
        return f"{random.choice(events)} {selected_location}. What would you like to do here?"
    
    def generate_combat_response(self):
        """Generate combat responses"""
        enemies = [
            "a fierce dragon with scales like emeralds",
            "a band of mischievous goblins armed with crude weapons",
            "an ancient stone guardian that awakens at your approach",
            "a shadowy figure that moves like smoke",
            "a giant spider with webs covering the trees"
        ]
        outcomes = [
            "You ready your weapon and face",
            "With courage, you prepare to fight",
            "Combat instincts take over as you confront",
            "You stand your ground against"
        ]
        
        return f"{random.choice(outcomes)} {random.choice(enemies)}. What's your next move?"
    
    def generate_item_response(self):
        """Generate item responses"""
        items = [
            "a glowing sword that hums with power",
            "an ancient key covered in rust but still sturdy",
            "a mysterious map drawn on aged parchment",
            "a healing potion that shimmers with blue light",
            "a bag of gold coins that jingles promisingly",
            "a magical amulet with a pulsating gem"
        ]
        discoveries = [
            "You carefully search the area and find",
            "To your surprise, you discover",
            "Hidden just out of sight is",
            "Your keen eyes spot"
        ]
        
        selected_item = random.choice(items)
        item_name = selected_item.split()[1]
        if item_name not in self.story_state["items"]:
            self.story_state["items"].append(item_name)
        
        return f"{random.choice(discoveries)} {selected_item}. This could be useful on your journey!"
    
    def generate_narrative_response(self):
        """Generate narrative responses"""
        hooks = [
            "The story takes an unexpected turn...",
            "New possibilities unfold before you...",
            "The plot thickens as mysteries deepen...",
            "Your adventure continues with new challenges..."
        ]
        questions = [
            "What path will you choose next?",
            "How will you shape this chapter of your story?",
            "What discovery awaits you around the next corner?",
            "Where does your curiosity lead you now?"
        ]
        
        return f"{random.choice(hooks)} {random.choice(questions)}"
    
    def adjust_tone(self, response, processed_input):
        """Adjust tone based on sentiment analysis"""
        sentiment = processed_input["sentiment"]
        voice_emotion = processed_input.get("voice_emotion", "neutral")
        
        if sentiment == "very_positive" or voice_emotion == "excited":
            modifiers = ["With triumphant energy,", "Joyfully,", "Exuberantly,", "With sparkling excitement,"]
            intensity = "✨ "
        elif sentiment == "very_negative" or voice_emotion == "nervous":
            modifiers = ["With grim determination,", "Darkly,", "With heavy heart,", "Amidst growing dread,"]
            intensity = "⚡ "
        elif sentiment == "positive":
            modifiers = ["Happily,", "Optimistically,", "Brightly,", "With hope,"]
            intensity = "🌟 "
        elif sentiment == "negative":
            modifiers = ["Sadly,", "With concern,", "Warily,", "Heavily,"]
            intensity = "🌑 "
        else:
            modifiers = ["Meanwhile,", "As events unfold,", "Curiously,", "Suddenly,"]
            intensity = "🔮 "
        
        if random.random() > 0.4:
            response = f"{intensity}{random.choice(modifiers)} {response.lower()}"
        
        return response
    
    def add_personalized_elements(self, response):
        """Add elements based on learned user preferences"""
        if not self.user_style_preferences:
            return response
        
        # Find user's preferred action type
        if self.user_style_preferences:
            action_prefs = {k: v for k, v in self.user_style_preferences.items() 
                          if isinstance(v, int) and k in ['combat', 'movement', 'item_interaction', 'dialogue']}
            if action_prefs:
                preferred_action = max(action_prefs.items(), key=lambda x: x[1])
                
                if preferred_action[0] == "combat" and "fight" not in response.lower():
                    additions = ["You sense danger nearby.", "Combat instincts stir within you."]
                    response = f"{response} {random.choice(additions)}"
                
                elif preferred_action[0] == "item_interaction" and "find" not in response.lower():
                    additions = ["Something valuable might be hidden here.", "Your treasure senses tingle."]
                    response = f"{response} {random.choice(additions)}"
        
        return response
    
    def add_surprise_plot_twist(self, response):
        """Add surprise plot twist triggered by excited voice"""
        twists = [
            "\n\n🎭 PLOT TWIST: A familiar face emerges from the shadows!",
            "\n\n🎭 PLOT TWIST: The artifact begins to glow with unexpected power!",
            "\n\n🎭 PLOT TWIST: The ground shakes as a hidden chamber reveals itself!",
            "\n\n🎭 PLOT TWIST: Your companion reveals a shocking secret!",
            "\n\n🎭 PLOT TWIST: The villain you sought was closer than you imagined!"
        ]
        
        if random.random() > 0.5:
            response = f"{response}{random.choice(twists)}"
        
        return response
    
    def generate_real_time_feedback(self, processed_input):
        """Provide real-time feedback about story adjustments"""
        feedback_options = [
            f"Story adapted to your {processed_input['sentiment']} tone.",
            f"Incorporated your interest in {', '.join(processed_input['keywords'][:2]) if processed_input['keywords'] else 'adventure'}.",
            f"Narrative coherence: {self.story_coherence_score:.1f}/10",
            f"Detected {processed_input.get('voice_emotion', 'neutral')} emotion in your voice.",
        ]
        
        if self.user_style_preferences:
            action_prefs = {k: v for k, v in self.user_style_preferences.items() 
                          if isinstance(v, int) and k in ['combat', 'movement', 'item_interaction', 'dialogue']}
            if action_prefs:
                preferred = max(action_prefs.items(), key=lambda x: x[1])
                feedback_options.append(f"Your style preference: {preferred[0]}")
        
        return random.choice(feedback_options)
    
    def calculate_coherence_score(self):
        """Calculate story coherence score (0-10)"""
        if len(self.story_state["story_arc"]) < 2:
            return 5.0
        
        # Simple coherence calculation
        score = 5.0
        
        # Add points for consistent locations
        unique_locations = len(set(self.story_state["locations"]))
        total_locations = len(self.story_state["locations"])
        if total_locations > 0:
            location_score = (unique_locations / total_locations) * 3
            score += location_score
        
        # Add points for consistent items
        if self.story_state["items"]:
            score += min(2, len(self.story_state["items"]) * 0.5)
        
        return min(10, max(0, score))
    
    def farewell(self):
        print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Thank you for playing Plot Twist!{Style.RESET_ALL}")
        
        if self.story_state["locations"]:
            unique_locations = set(self.story_state["locations"])
            print(f"{Fore.GREEN}You visited: {', '.join(unique_locations)}{Style.RESET_ALL}")
        
        if self.story_state["items"]:
            print(f"{Fore.BLUE}You found: {', '.join(self.story_state['items'])}{Style.RESET_ALL}")
        
        print(f"{Fore.MAGENTA}Story Coherence Score: {self.story_coherence_score:.1f}/10{Style.RESET_ALL}")
        
        if self.user_style_preferences:
            print(f"{Fore.CYAN}Your play style:{Style.RESET_ALL}")
            for pref, count in self.user_style_preferences.items():
                if isinstance(count, int) and count > 0:
                    print(f"  {pref}: {count} times")
        
        # Save suggestion
        save_choice = input(f"\n{Fore.YELLOW}Would you like to save your story before exiting? (yes/no): {Style.RESET_ALL}").lower()
        if save_choice == "yes":
            save_result = self.save_story_progress()
            print(save_result["response"])
        
        print(f"\n{Fore.YELLOW}Saves are stored in: {self.saved_stories_folder}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Hope to continue your adventure soon!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")

def main():
    chatbot = PlotTwistChatbot()
    chatbot.greeting()
    
    # If story was loaded in greeting, skip initial generation
    if not chatbot.conversation_history:
        # Initial story generation
        initial_response = chatbot.generate_response({
            "action_type": "narrative", 
            "sentiment": "neutral",
            "keywords": [],
            "voice_emotion": "neutral"
        })
        print(f"\n{Fore.MAGENTA}{initial_response}{Style.RESET_ALL}")
    
    # Main conversation loop
    while chatbot.conversation_is_active:
        try:
            # Get user input
            user_input = chatbot.receive_input()
            
            # Check for quit command
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                chatbot.conversation_is_active = False
                break
            
            # Process and respond
            processed = chatbot.process_input(user_input)
            response = chatbot.generate_response(processed)
            
            # Print response
            print(f"\n{Fore.WHITE}{'─'*50}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}{response}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{'─'*50}{Style.RESET_ALL}")
            
        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}Story interrupted...{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"\n{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Let's continue the story...{Style.RESET_ALL}")
            response = "What would you like to do next?"
            print(f"\n{Fore.MAGENTA}{response}{Style.RESET_ALL}")
    
    chatbot.farewell()

if __name__ == "__main__":
    main()
# Plot Twist - AI-Powered Interactive Storyteller

An interactive storytelling chatbot that creates dynamic adventures based on user input using advanced NLP techniques.

## Key Features
- **Text/Voice Mode:** Speak or Write your text with emotion detection
- **Save/Load System:** Continue adventures later
- **BART Integration** Uses BART model for coherent and context-aware story generation
- **Story Branch Backtracking:** Allows going back to previous choices
- **User Preference Learning:** Adapts to user's style and preferences over time
- **Real-time Feedback:** Explains story logic and adjustments
- **Sentiment Analysis:** Adapts story tone based on user emotions
- **Keyword Detection:** Identifies important story elements
- **Text Classification:** Categorizes user actions
- **Adaptive Storytelling:** Generates personalized narratives


## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create new conda environment called "aim" with Python 3.10
conda create -n aim python=3.10

# Activate the environment
conda activate aim

# First install PyTorch (required for BART and Whisper)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install system dependencies for voice recognition
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg

# Install Python audio packages
pip install sounddevice scipy numpy

# Install all Python dependencies from requirements.txt
pip install -r requirements.txt

# Run the chatbot
python plot_twist_chatbot.py






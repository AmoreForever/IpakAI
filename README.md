# 🏦 Uzbek Bank Voice Assistant

This project is a voice assistant built on Flask that communicates with the user in Uzbek, converts voice requests to text and vice versa, converts text to voice. The user can also send a request for credit limit.

## ✨ Main Features

- 🎤 **Speech-to-Text (STT)**: Converts WAV audio files to text via the Google Gemini API.
- 🔊 **Text-to-Speech (TTS)**: Converts text to MP3 audio via the AIsha TTS API.
- 🧠 **Credit Limit Prediction**: Determines the credit limit using a pre-trained model based on the user ID.
- 👋 **Natural Language Understanding**: Recognizes greetings, thanks, questions about credit, and other contexts.
- 🔁 **Cyrillic-Latin Conversion**: Converts text in Cyrillic to Latin.
- 🔢 **Word to Number Conversion**: There is a function to convert Uzbek number words to numbers.

## 🛠 Technologies

- Python 3
- Flask
- Google Gemini API (STT)
- AIsha TTS API
- Together API
- Joblib (for model loading)
- Pandas (for working with CSV)
- dotenv (for confidential data)
- Logging (for tracking errors and processes)

## 📦 Project Setup

```bash
git clone https://github.com/yourusername/uzbek-bank-voice-assistant.git
cd uzbek-bank-voice-assistant
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt

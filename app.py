from flask import Flask, render_template, request, jsonify, send_from_directory, session
import os
import uuid
import pandas as pd
import joblib
import requests
import re
import logging
from typing import Tuple, Optional
from dotenv import load_dotenv
from together import Together
from google import genai
from google.genai import types
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY", "your_secure_secret_key"
)  # Replace with secure key
app.config["UPLOAD_FOLDER"] = "audio"

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TTS_API_KEY = os.getenv("TTS_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Initialize Together client
client = Together(api_key=TOGETHER_API_KEY)

# Create directories
os.makedirs("temp", exist_ok=True)
os.makedirs("audio", exist_ok=True)


# Load bank info, model, and test data
def load_bank_info(file_path="general_info.txt"):
    if not os.path.exists(file_path):
        return "Bank haqida ma'lumot fayli topilmadi."
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


BANK_INFO = load_bank_info()
MODEL = joblib.load("linear_regression_model.pkl")
TEST_DATA = pd.read_csv("test_data2.csv")
FEATURES = [
    "Income",
    "Rating",
    "Cards",
    "Age",
    "Education",
    "Gender",
    "Student",
    "Married",
    "Ethnicity",
    "Balance",
]


# Speech-to-Text function using Google Gemini
def speech_to_text(audio_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribes an audio file to text using Google Gemini API.

    Args:
        audio_path (str): Path to the audio file (expected to be WAV).

    Returns:
        Tuple[Optional[str], Optional[str]]: (transcript, error_message)
    """
    if not audio_path.lower().endswith(".wav"):
        logger.error(f"Invalid file format: {audio_path}. Expected WAV.")
        return None, "Audio file must be in WAV format"

    file_size = os.path.getsize(audio_path)
    logger.info(f"Sending WAV file: {audio_path}, size: {file_size} bytes")

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        transcript = ""

        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        audio_part = types.Part.from_bytes(
            data=base64.b64decode(audio_base64), mime_type="audio/wav"
        )
        model = "gemini-2.0-flash-001"
        contents = [
            types.Content(
                role="user", parts=[audio_part, types.Part.from_text(text=".")]
            )
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                ),
            ],
            system_instruction=[
                types.Part.from_text(text="transcribe given audio to uzbek language")
            ],
        )

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            for candidate in chunk.candidates:
                for part in candidate.content.parts:
                    transcript += part.text + " "
        transcript = transcript.strip()

        if not transcript:
            logger.warning("Empty transcript received")
            return None, "No speech detected in audio"
        return transcript, None
    except Exception as e:
        logger.error(f"STT Exception: {str(e)}")
        return None, str(e)


# Text-to-Speech function
def text_to_speech(
    text: str, language: str = "uz", model: str = "gulnoza"
) -> Optional[str]:
    """
    Converts text to an audio file using a TTS API.

    Args:
        text (str): Text to convert to speech.
        language (str): Language code for synthesis (default: "uz").
        model (str): TTS model name (default: "gulnoza").

    Returns:
        Optional[str]: Path to the generated audio file, or None if failed.
    """
    url = "https://back.aisha.group/api/v1/tts/post/"
    headers = {
        "x-api-key": TTS_API_KEY,
        "X-Channels": "stereo",
        "X-Quality": "64k",
        "X-Rate": "16000",
        "X-Format": "mp3",
    }
    data = {"transcript": text, "language": language, "model": model}

    try:
        response = requests.post(url, headers=headers, data=data, timeout=30)
        logger.info(f"TTS Response Status: {response.status_code}")
        logger.debug(f"TTS Response Content: {response.text}")

        if response.status_code in (200, 201):
            response_data = response.json()
            audio_url = response_data.get("audio_path")
            if not audio_url:
                logger.error("No audio_path in response")
                return None

            audio_response = requests.get(audio_url, timeout=30)
            if audio_response.status_code == 200:
                os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
                output_file = os.path.join(
                    app.config["UPLOAD_FOLDER"], f"output_{uuid.uuid4()}.mp3"
                )
                with open(output_file, "wb") as f:
                    f.write(audio_response.content)
                logger.info(f"TTS Output saved: {output_file}")
                return output_file
            else:
                logger.error(
                    f"TTS Audio Fetch Error: Status {audio_response.status_code}"
                )
                return None
        else:
            logger.error(
                f"TTS Error: Status {response.status_code}, Response: {response.text}"
            )
            return None
    except requests.exceptions.Timeout:
        logger.error("TTS request timed out")
        return None
    except Exception as e:
        logger.error(f"TTS Exception: {str(e)}")
        return None


# Helper functions for chatbot logic
def predict_limit_by_id(input_id):
    if input_id not in TEST_DATA["ID"].values:
        return f"ID {input_id} topilmadi."
    input_data = TEST_DATA[TEST_DATA["ID"] == input_id][FEATURES]
    return MODEL.predict(input_data)[0]


def is_credit_query(message):
    credit_keywords = [
        "kredit",
        "qarz",
        "limit",
        "pul olish",
        "kredit olish",
        "kredit limiti",
    ]
    return any(keyword.lower() in message.lower() for keyword in credit_keywords)


def is_greeting(message):
    greeting_keywords = ["salom", "assalom", "assalomu alaykum", "assalomu aleykum"]
    return any(keyword.lower() in message.lower() for keyword in greeting_keywords)


def is_thanks(message):
    thanks_keywords = ["rahmat", "tashakkur"]
    return any(keyword.lower() in message.lower() for keyword in thanks_keywords)


def is_credit_reason_query(message):
    reason_keywords = [
        "nima uchun",
        "negadir",
        "nima sababdan",
        "qanday qilib",
        "why",
        "how",
    ]
    return (
        any(keyword.lower() in message.lower() for keyword in reason_keywords)
        and "kredit" in message.lower()
    )


def is_bot_info_query(message):
    bot_keywords = [
        "isming",
        "kim",
        "nomi",
        "quruvchi",
        "ishlab chiqaruvchi",
        "developer",
    ]
    return any(keyword.lower() in message.lower() for keyword in bot_keywords)


def krill_to_latin(text):
    krill_to_latin_map = {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "yo",
        "ж": "j",
        "з": "z",
        "и": "i",
        "й": "y",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "x",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "ъ": "",
        "ы": "i",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
        "қ": "q",
        "ғ": "g'",
        "ҳ": "h",
        "А": "A",
        "Б": "B",
        "В": "V",
        "Г": "G",
        "Д": "D",
        "Е": "E",
        "Ё": "Yo",
        "Ж": "J",
        "З": "Z",
        "И": "I",
        "Й": "Y",
        "К": "K",
        "Л": "L",
        "М": "M",
        "Н": "N",
        "О": "O",
        "П": "P",
        "Р": "R",
        "С": "S",
        "Т": "T",
        "У": "U",
        "Ф": "F",
        "Х": "X",
        "Ц": "Ts",
        "Ч": "Ch",
        "Ш": "Sh",
        "Ъ": "",
        "Ы": "I",
        "Ь": "",
        "Э": "E",
        "Ю": "Yu",
        "Я": "Ya",
        "Қ": "Q",
        "Ғ": "G'",
        "Ҳ": "H",
    }
    return "".join(krill_to_latin_map.get(char, char) for char in text)


def uzbek_text_to_number(text):
    number_map = {
        "nol": 0,
        "bir": 1,
        "ikki": 2,
        "uch": 3,
        "to'rt": 4,
        "besh": 5,
        "olti": 6,
        "yetti": 7,
        "sakkiz": 8,
        "to'qqiz": 9,
        "o'n": 10,
        "yigirma": 20,
        "o'ttiz": 30,
        "qirq": 40,
        "ellik": 50,
        "oltmish": 60,
        "yetmish": 70,
        "sakson": 80,
        "to'qson": 90,
        "yuz": 100,
        "yuzi": 100,
        "ming": 1000,
        "million": 1000000,
    }
    text = krill_to_latin(text.lower().replace("va", "").strip())
    numeric_match = re.search(r"\b\d+\b", text)
    if numeric_match:
        try:
            return int(numeric_match.group())
        except ValueError:
            pass
    words = text.split()
    number_words = [word for word in words if word in number_map]
    if not number_words:
        return None
    if len(number_words) == 1:
        return number_map[number_words[0]]
    total = 0
    current = 0
    for word in number_words:
        num = number_map[word]
        if num in [100, 1000, 1000000]:
            if current == 0:
                current = 1
            total += current * num
            current = 0
        else:
            current += num
    total += current
    return total if total > 0 else None


def generate_response(prompt, chat_history):
    messages = chat_history + [
        {
            "role": "user",
            "content": f"Answer concisely (max 50 words) in Uzbek: {prompt}",
        }
    ]
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=messages,
            temperature=0.7,
            max_tokens=50,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Uzr, hozirda javob bera olmayman."


def process_message(user_input, chat_history, waiting_for_id, last_credit_amount):
    if not user_input:
        return (
            "Iltimos, aniq ovozli xabar yuboring yoki matn kiriting.",
            chat_history,
            waiting_for_id,
            last_credit_amount,
        )
    user_input = krill_to_latin(user_input.lower().strip())
    chat_history.append({"role": "user", "content": user_input})

    if is_greeting(user_input):
        response = "Vaalaykum assalom! Sizga qanday yordam bera olaman?"
    elif is_thanks(user_input):
        response = "Arzimaydi! Sizga yordam bera olganimdan xursandman."
    elif is_bot_info_query(user_input):
        if "ism" in user_input or "nomi" in user_input:
            response = "Mening ismim Epsilion."
        elif (
            "quruvchi" in user_input
            or "ishlab chiqaruvchi" in user_input
            or "developer" in user_input
        ):
            response = "Meni ishlab chiqaruvchimning taxallusi Neo."
        else:
            response = "Men Epsilion, Neo tomonidan yaratilganman."
    elif is_credit_reason_query(user_input) and last_credit_amount is not None:
        response = "Mening kredit scoring modelim buni bashorat qildi."
    elif waiting_for_id or any(
        is_credit_query(msg["content"]) or "topilmadi" in msg["content"]
        for msg in chat_history[-4:]
    ):
        parsed_id = uzbek_text_to_number(user_input)
        if parsed_id is not None:
            prediction = predict_limit_by_id(parsed_id)
            if isinstance(prediction, str):
                response = f"ID {parsed_id} topilmadi. Iltimos, boshqa ID kiriting."
                waiting_for_id = True
            else:
                last_credit_amount = prediction
                response = f"Sizga bir yil muddatga {prediction:.2f} dollar miqdorida kredit bera olamiz."
                waiting_for_id = False
        else:
            response = "To'g'ri ID raqamini kiriting (masalan, '1', 'bir', '127')."
            waiting_for_id = True
    else:
        if is_credit_query(user_input):
            waiting_for_id = True
            response = "Kredit limiti uchun ID raqamingizni kiriting (masalan, '1', 'bir', '127')."
        else:
            prompt = f"Quyidagi ma'lumot asosida qisqa va muloyim javob bering:\n{BANK_INFO}\n\nSavol: {user_input}"
            response = generate_response(prompt, chat_history)

    chat_history.append({"role": "assistant", "content": response})
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]
    return response, chat_history, waiting_for_id, last_credit_amount


# Routes
@app.route("/")
def index():
    session.clear()  # Clear session for new users
    return render_template("index.html")


@app.route("/process_voice", methods=["POST"])
def process_voice():
    if "audio" not in request.files:
        return jsonify({"error": "Audio file not provided"}), 400

    audio_file = request.files["audio"]
    temp_path = os.path.join("temp", f"{uuid.uuid4()}.wav")
    audio_file.save(temp_path)

    transcript, error = speech_to_text(temp_path)
    os.remove(temp_path)

    if error:
        return jsonify({"error": error}), 400

    chat_history = session.get("chat_history", [])
    waiting_for_id = session.get("waiting_for_id", False)
    last_credit_amount = session.get("last_credit_amount", None)

    response, new_chat_history, new_waiting_for_id, new_last_credit_amount = (
        process_message(transcript, chat_history, waiting_for_id, last_credit_amount)
    )

    session["chat_history"] = new_chat_history
    session["waiting_for_id"] = new_waiting_for_id
    session["last_credit_amount"] = new_last_credit_amount

    audio_path = text_to_speech(response)
    audio_url = None
    if audio_path:
        audio_url = f"/audio/{os.path.basename(audio_path)}"

    return jsonify(
        {"response": response, "audio_url": audio_url, "transcript": transcript}
    )


@app.route("/process_text", methods=["POST"])
def process_text():
    user_input = request.form.get("text")
    if not user_input:
        return jsonify({"error": "Text input not provided"}), 400

    chat_history = session.get("chat_history", [])
    waiting_for_id = session.get("waiting_for_id", False)
    last_credit_amount = session.get("last_credit_amount", None)

    response, new_chat_history, new_waiting_for_id, new_last_credit_amount = (
        process_message(user_input, chat_history, waiting_for_id, last_credit_amount)
    )

    session["chat_history"] = new_chat_history
    session["waiting_for_id"] = new_waiting_for_id
    session["last_credit_amount"] = new_last_credit_amount

    audio_path = text_to_speech(response)
    audio_url = None
    if audio_path:
        audio_url = f"/audio/{os.path.basename(audio_path)}"

    return jsonify({"response": response, "audio_url": audio_url})


@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)

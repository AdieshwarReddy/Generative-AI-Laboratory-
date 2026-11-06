
# Generative AI Laboratory
# Experiment 8: Real-Time Speech Transcription

import os
import csv
from datetime import datetime

import speech_recognition as sr


# ---------------------------------------
# Step 1: Configuration
# ---------------------------------------

OUTPUT_DIR = "outputs"

TEXT_FILE = os.path.join(
    OUTPUT_DIR,
    "transcription.txt"
)

CSV_FILE = os.path.join(
    OUTPUT_DIR,
    "transcription_log.csv"
)

LANGUAGE = "en-IN"

LISTEN_TIMEOUT = 5
PHRASE_TIME_LIMIT = 10

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ---------------------------------------
# Step 2: Initialize Recognizer
# ---------------------------------------

recognizer = sr.Recognizer()

recognizer.dynamic_energy_threshold = True

print("=" * 50)
print("GENERATIVE AI LAB - EXPERIMENT 8")
print("REAL-TIME SPEECH TRANSCRIPTION")
print("=" * 50)

print("\nLanguage:", LANGUAGE)
print("Recognition Engine: Google Web Speech")
print("Press Ctrl+C to stop transcription.\n")


# ---------------------------------------
# Step 3: Initialize Output Files
# ---------------------------------------

def initialize_files():

    if not os.path.exists(CSV_FILE):

        with open(
            CSV_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Timestamp",
                "Recognized Text"
            ])

    if not os.path.exists(TEXT_FILE):

        with open(
            TEXT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "REAL-TIME SPEECH TRANSCRIPTION\n"
            )

            file.write("=" * 40 + "\n\n")


# ---------------------------------------
# Step 4: Save Transcription
# ---------------------------------------

def save_transcription(text):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Save to text file
    with open(
        TEXT_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            f"[{timestamp}] {text}\n"
        )

    # Save to CSV file
    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            timestamp,
            text
        ])

    print("Timestamp:", timestamp)
    print("Transcription saved successfully.\n")


# ---------------------------------------
# Step 5: Speech Transcription
# ---------------------------------------

def transcribe_speech():

    initialize_files()

    try:

        with sr.Microphone() as source:

            print("Microphone initialized.")

            print(
                "Adjusting for ambient noise..."
            )

            recognizer.adjust_for_ambient_noise(
                source,
                duration=1
            )

            print("Microphone ready.")
            print("Start speaking...\n")

            while True:

                try:

                    print("Listening...")

                    # Capture microphone audio
                    audio = recognizer.listen(
                        source,
                        timeout=LISTEN_TIMEOUT,
                        phrase_time_limit=PHRASE_TIME_LIMIT
                    )

                    print("Recognizing speech...")

                    # Convert speech into text
                    text = recognizer.recognize_google(
                        audio,
                        language=LANGUAGE
                    )

                    print("\nRecognized Text:")
                    print(text)

                    # Save recognized text
                    save_transcription(text)

                except sr.WaitTimeoutError:

                    print(
                        "No speech detected. "
                        "Listening again...\n"
                    )

                except sr.UnknownValueError:

                    print(
                        "Speech could not be understood. "
                        "Please speak clearly.\n"
                    )

                except sr.RequestError as error:

                    print(
                        "Speech recognition service error:",
                        error
                    )

                    print(
                        "Check your internet connection.\n"
                    )

    except KeyboardInterrupt:

        print("\nTranscription stopped by user.")

    except OSError as error:

        print("\nMicrophone Error:", error)

        print(
            "Check microphone connection "
            "and system permissions."
        )


# ---------------------------------------
# Step 6: Main Function
# ---------------------------------------

def main():

    print("Starting Speech Transcription...\n")

    transcribe_speech()

    print("\nGenerated Files:")
    print(TEXT_FILE)
    print(CSV_FILE)

    print(
        "\nExperiment 8 execution finished!"
    )


# ---------------------------------------
# Step 7: Execute Program
# ---------------------------------------

if __name__ == "__main__":
    main()

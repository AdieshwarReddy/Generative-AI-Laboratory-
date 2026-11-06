
# Experiment 8: Real-Time Speech Transcription Using Python

## Aim

To implement a real-time speech transcription service using Python that captures speech from a microphone and converts spoken words into text using automatic speech recognition.

## Objectives

1. Understand Automatic Speech Recognition (ASR).
2. Capture live audio through a microphone.
3. Convert spoken words into text.
4. Implement speech recognition using Python.
5. Display recognized text in the terminal.
6. Save transcription results into a text file.
7. Store transcription logs in CSV format.
8. Handle microphone and recognition errors.

## Theory

### What is Speech Recognition?

Speech Recognition is an Artificial Intelligence technology that converts spoken language into written text.

It is also called Automatic Speech Recognition (ASR) or Speech-to-Text (STT).

Speech recognition systems analyze audio signals and predict the words spoken by a user.

### What is Real-Time Transcription?

Real-time transcription converts speech into text while a person is speaking or shortly after a spoken segment ends.

This experiment uses phrase-by-phrase recognition.

The application:

1. Listens through a microphone.
2. Captures a spoken phrase.
3. Sends the audio to a speech recognition service.
4. Receives the recognized text.
5. Displays the text.
6. Saves the transcription.
7. Repeats until the user stops the program.

## System Architecture

```text
             START
               |
               v
       Initialize Python
               |
               v
       Access Microphone
               |
               v
       Capture Speech
               |
               v
       Convert Audio to Data
               |
               v
       Speech Recognition
               |
               v
       Recognized Text
               |
       +-------+-------+
       |               |
       v               v
  Display Text     Save Text
       |               |
       +-------+-------+
               |
               v
       Continue Listening?
           /       \
         Yes        No
          |          |
          v          v
     Capture Again   END
```

## Automatic Speech Recognition

ASR systems commonly include the following stages:

### 1. Audio Acquisition

The microphone captures sound waves and converts them into digital audio samples.

### 2. Preprocessing

The audio is prepared for recognition.

Depending on the system, preprocessing may include:

- Noise reduction
- Silence detection
- Audio normalization
- Feature extraction

### 3. Acoustic Processing

The recognition model analyzes patterns in the audio signal.

### 4. Language Processing

The recognition system estimates the most likely sequence of spoken words.

### 5. Text Generation

The recognized words are returned as readable text.

## Mathematical Concept

A speech recognition system aims to find the most probable text sequence W for an audio signal A.

W* = argmax P(W | A)

Using Bayes' theorem:

P(W | A) ∝ P(A | W) × P(W)

Where:

- W = Word sequence
- A = Audio signal
- P(A | W) = Acoustic likelihood
- P(W) = Language prior

Modern speech recognition systems may estimate the relationship between audio and text using end-to-end neural networks rather than separate acoustic and language models.

## Libraries Used

### SpeechRecognition

Provides a Python interface for recording audio and accessing speech recognition engines.

### PyAudio

Allows Python applications to capture audio from microphone devices.

### Google Web Speech Recognition

An online recognition service used through the SpeechRecognition library.

The microphone audio is sent to an external service for recognition.

An internet connection is required.

## Software Requirements

- Python 3.10 or 3.11
- SpeechRecognition
- PyAudio
- Working microphone
- Internet connection
- VS Code / Python IDE

## Algorithm

Step 1: Import the required Python libraries.

Step 2: Initialize the speech recognizer.

Step 3: Configure the microphone.

Step 4: Adjust for ambient noise.

Step 5: Listen to a spoken phrase.

Step 6: Capture audio from the microphone.

Step 7: Send audio to the speech recognition service.

Step 8: Receive recognized text.

Step 9: Display the recognized text.

Step 10: Save the transcription into a text file.

Step 11: Save a timestamped record into a CSV file.

Step 12: Repeat the process until the user presses Ctrl+C.

Step 13: Close the application.

## Features

- Live microphone input
- Phrase-by-phrase speech recognition
- Automatic ambient noise calibration
- English speech transcription
- Timestamped transcription records
- Text file output
- CSV log output
- Error handling
- Keyboard interruption support

## Installation

Install the dependencies:

    python -m pip install -r requirements.txt

## Execution

Run the program:

    python speech_transcription.py

Speak clearly into the microphone.

The recognized text will appear in the terminal.

Press Ctrl+C to stop recording.

## Expected Output

The program displays:

- Microphone initialization status
- Listening status
- Recognized speech
- Transcription timestamps
- Error messages when recognition fails

Generated files:

    outputs/transcription.txt
    outputs/transcription_log.csv

## Applications

1. Voice assistants
2. Meeting transcription
3. Lecture transcription
4. Accessibility applications
5. Voice-controlled systems
6. Speech-based interfaces
7. Customer service applications
8. Automated note-taking

## Limitations

- Requires an internet connection.
- Recognition accuracy depends on microphone quality.
- Background noise can reduce accuracy.
- The service may be temporarily unavailable.
- Recognition occurs after each spoken phrase.
- The application does not provide continuous streaming word-by-word transcription.
- Audio is processed by an external recognition service.

## Result

A Python-based speech transcription application was implemented using microphone input and Google Web Speech Recognition.

The application captures spoken phrases, converts them into text, and saves the transcription results.

## Conclusion

This experiment demonstrates the practical implementation of Automatic Speech Recognition using Python.

Speech-to-text technology enables voice-controlled applications, automated transcription, and accessible human-computer interaction.

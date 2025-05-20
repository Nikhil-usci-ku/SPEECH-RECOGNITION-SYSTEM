# SPEECH-RECOGNITION-SYSTEM

*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: NIKHIL KUMAR

*INTERN ID*: CODF69

*DOMAIN*: AIML

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH KUMAR

> This repository contains a Python script for audio transcription, converting audio files (.wav) into text using pre-trained models.

> This project leverages a pre-trained Wav2Vec model for accurate transcription and includes a punctuation model to enhance readability.

##   Description
This project centers on creating a Python-based audio transcription tool that converts .wav audio files to text. It utilizes pre-trained models from Hugging Face, specifically Wav2Vec 2.0, and a dedicated punctuation model to achieve accurate and readable transcriptions.  The project addresses the challenge of accurate speech-to-text conversion by employing state-of-the-art deep learning.

The tool leverages the `facebook/wav2vec2-base-960h` model from Hugging Face, a powerful neural network architecture known for its strong performance in speech recognition.  It is trained on a large corpus of unlabeled audio data, enabling it to learn robust representations of speech and effectively convert audio signals into textual tokens. The facebook/wav2vec2-base-960h version is specifically used, pre-trained on 960 hours of English speech data for enhanced accuracy.  Wav2Vec 2.0's strength lies in its ability to learn directly from raw audio waveforms through self-supervised learning, capturing the underlying structure of speech without extensive feature engineering.

A key challenge in speech-to-text conversion is the lack of punctuation in raw output. To address this, the project incorporates a separate, specifically trained punctuation model.  This model analyzes contextual information in the transcribed text, such as word sequences, sentence structure, and semantic relationships, to accurately restore missing punctuation marks, including periods, commas, and question marks.  By adding punctuation, the tool significantly enhances the readability and clarity of the final transcription, transforming a stream of words into well-formed, natural-sounding sentences.

The process begins with the audio transcription phase. Here, the system utilizes a pre-trained `facebook/wav2vec2-base-960h` model. This preparation includes resampling the audio to 16,000 Hz, a critical step as most Wav2Vec2 models are trained on this specific sampling rate. Mismatched rates can lead to significant errors, so this resampling guarantees the audio features align perfectly with the model's training. Once processed, the audio is fed to the Wav2Vec2ForCTC model, which then outputs a raw, unpunctuated stream of text.

The real innovation lies in the subsequent punctuation restoration phase. Unlike simpler methods that might just add a period at the end or guess sentence breaks based on pauses, this system employs a dedicated language model for token classification using `oliverguhr/fullstop-punctuation-prediction` from *deepmultilingualpunctuation* library. This model doesn't just look for silence; it analyzes the linguistic context of the transcribed words. The AutoTokenizer breaks the raw transcript into individual words, and the AutoModelForTokenClassification then predicts a specific punctuation mark (like a comma, period, question mark, or exclamation mark) for each word, or determines if no punctuation is needed.

Finally, a smart function meticulously reconstructs the text. It iterates through the words, inserting the predicted punctuation marks where appropriate. This includes minor clean-up to ensure correct spacing around the punctuation. The result is a transcription that is not only accurate in converting speech to text but also features proper sentence segmentation and punctuation, vastly enhancing its clarity and making it much more natural to read.

### Key Components
* Two-Stage Processing:
  1. Transcription: Uses Wav2Vec2 for robust speech-to-text conversion.
  2. Punctuation: Employs a specialized language model for intelligent punctuation restoration.
* Audio Resampling: The Wav2Vec2Processor automatically resamples audio to 16,000 Hz, ensuring optimal compatibility and performance with the transcription model.
* Contextual Punctuation: The punctuation model leverages linguistic context to accurately place commas, periods, question marks, and other punctuation, rather than relying on simple heuristics.

### Features
* Complete Speech-to-Text: Converts audio into text.
* Advanced Transcription: Uses Wav2Vec2 deep learning model for accurate speech recognition.
* Smart Audio Processing: Automatically resamples audio to 16,000 Hz for optimal model performance.
* Intelligent Punctuation: Employs a dedicated AI model to add contextual punctuation (periods, commas, etc.), significantly improving readability.
* Enhanced Readability: Produces clear, well-punctuated transcripts for better understanding.
* Modular & Extensible: Built with Hugging Face transformers, allowing easy customization and integration.

## Prerequisites
> Python 3.12

## Installation
Clone the repository (or download the Zip File):

`git clone https://github.com/your-username/advanced-abstractive-summarization.git`
`cd advanced-abstractive-summarization`



Install the necessary Python libraries provided in requirements.txt file:

`pip install -r requirements.txt`

> The script will attempt to download required models automatically upon first run.

## Running
Run the Python script Task2.py directly. Provide the audio file location to get transcripts .

## Output
* User Input



* Transcript Output





































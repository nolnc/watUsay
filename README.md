# watUsay

Live multilingual speech transcrilation (transcription/translation), powered by OpenAI's Whisper.

This tool uses the Whisper model to provide real-time transcription and translation of your microphone audio or system audio (loopback).

## Setup

These instructions are for setting up `watUsay` on Windows.

### Prerequisites

*   **Git**
*   **Python 3.8 or newer.**
*   **FFmpeg:** This is required for audio processing. You can install it from the official website or using a package manager.

    ```bash
    # on Windows using Winget
    winget install "FFmpeg (Shared)"
    ```

### Installation Steps

1.  **Create a Python virtual environment:**
    ```bash
    python -m venv watusay_env
    ```

2.  **Activate the virtual environment:**
    ```bash
    watusay_env\Scripts\activate.bat
    ```

3.  **Clone the watUsay repository:**
    ```bash
    git clone https://github.com/nolnc/watUsay.git
    ```

4.  **Install the required Python modules:**
    ```bash
    pip install -r watUsay/requirements.txt
    ```
    *Note: If you run into issues with `tiktoken`, you may need to install Rust. Follow the [Getting started page](https://www.rust-lang.org/learn/get-started) to install the Rust development environment.*

## Usage

To start the live transcription/translation, run the `watUsay.py` script from within the activated virtual environment. You will be prompted to select an audio input device.

```bash
python watUsay\watUsay.py [options]
```

When you are finished, you can stop the script with `Ctrl+C` and deactivate the virtual environment:
```bash
watusay_env\Scripts\deactivate.bat
```

### Example Usage

- **Activate virtual environment**
  ```shell
  watusay_env\Scripts\activate.bat
  ```

- **Transcribe with a `small` model, processing every 3 seconds:**
  ```shell
  python watUsay\watUsay.py --model small --buffer_seconds 3
  ```

- **Translate from Chinese to English using the default `base` model:**
  ```shell
  python watUsay\watUsay.py --task translate --language zh
  ```

### Command-Line Options

*   `--model`: Model to use (e.g., `tiny`, `base`, `small`). Default is `base`.
*   `--task`: Task to perform: `transcribe` or `translate`. Default is `transcribe`.
*   `--language`: Language of the speech (e.g., `en`, `zh`, `ja`, `de`). Default is auto-detect.
*   `--buffer_seconds`: Seconds of audio to buffer before processing. Default is `5`.
*   `--energy_threshold`: Energy threshold for detecting speech. Default is `0.005`.

For a detailed explanation of these parameters, see the **Parameter Details** section below.

### Enabling Loopback Device (for system audio)

If you want to transcrilate audio playing on your computer (e.g., from a video or a call), you'll need to use a loopback device. If the script identifies a loopback device but can't use it, you may need to enable it in Windows Settings.

1.  Go to **Control Panel -> Sound -> Recording** tab.
2.  Right-click in the empty space and ensure **"Show Disabled Devices"** is checked.
3.  Find your loopback device (often called "Stereo Mix", "Wave Out Mix", or "What U Hear").
4.  Right-click on it and select **Enable**.

---

## Parameter Details

### --model
This parameter lets you choose which version of the Whisper model you want to use. Whisper models come in different sizes, and there's a direct trade-off between speed, resource usage, and accuracy.

*   **Smaller Models (`tiny`, `base`)**
    *   **Pros:** Fast and require minimal resources (CPU/VRAM). They can run on most modern computers, even without a powerful graphics card.
    *   **Cons:** They are significantly less accurate. They might struggle with noisy audio, accents, or specialized terminology.
*   **Larger Models (`small`, `medium`, `large`)**
    *   **Pros:** They are much more accurate and robust. The large model, in particular, provides state-of-the-art results.
    *   **Cons:** They are much slower and require a powerful computer, ideally with a modern NVIDIA GPU and plenty of VRAM (e.g., the large model needs about 10 GB of VRAM). Without a GPU, they can be too slow for real-time use.
*   **English-Only Models (`.en` suffix, e.g., `base.en`)**
    *   If you are only transcribing English, these models can provide slightly better accuracy and speed compared to their multilingual counterparts of the same size.

**Recommendation:** Start with the default `base` model. If it's too slow, try `tiny`. If you need higher accuracy and have a good GPU, move up to `small` or `medium`.

## Available models and languages

There are six model sizes, four with English-only versions, offering speed and accuracy tradeoffs.
Below are the names of the available models and their approximate memory requirements and inference speed relative to the large model.
The relative speeds below are measured by transcribing English speech on a A100, and the real-world speed may vary significantly depending on many factors including the language, the speaking speed, and the available hardware.

|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~10x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~7x       |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~4x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |
| turbo  |   809 M    |        N/A         |      `turbo`       |     ~6 GB     |      ~8x       |

The `.en` models for English-only applications tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.
Additionally, the `turbo` model is an optimized version of `large-v3` that offers faster transcription speed with a minimal degradation in accuracy.

Whisper's performance varies widely depending on the language. The figure below shows a performance breakdown of `large-v3` and `large-v2` models by language, using WERs (word error rates) or CER (character error rates, shown in *Italic*) evaluated on the Common Voice 15 and Fleurs datasets. Additional WER/CER metrics corresponding to the other models and datasets can be found in Appendix D.1, D.2, and D.4 of [the paper](https://arxiv.org/abs/2212.04356), as well as the BLEU (Bilingual Evaluation Understudy) scores for translation in Appendix D.3.

![WER breakdown by language](https://github.com/openai/whisper/assets/266841/f4619d66-1058-4005-8f67-a9d811b77c62)

### --task
This converts the spoken audio into text in the same language.

*   **`transcribe` (Default):** This converts the spoken audio into text in the same language.
Example: If you speak Japanese, the output will be Japanese text.
*   **`translate`:** Translates speech from any supported language directly into **English text**. Note: This task only works with the multilingual models (e.g., base, small, medium, large), not the English-only (.en) models.

### --language
This parameter specifies the language of the audio.

*   **Automatic Detection (Default):** If you don't set this parameter, Whisper will listen to the first few seconds of audio and automatically determine the language being spoken. This is convenient but can sometimes be inaccurate with short phrases.
*   **Manual Specification:** If you know for certain what language will be spoken, setting it manually can improve performance and reliability. (e.g., --language zh for Chinese)
*   **Benefits:** It prevents the model from incorrectly identifying the language, and it can make the initial processing slightly faster since the language detection step is skipped.
You can find the list of supported language codes in the [whisper/tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) file.

### --buffer_seconds
This parameter defines the length of the audio chunks (in seconds) that the script processes at one time. It's a crucial setting for balancing latency and accuracy.

*   **How it works:** The script records audio continuously. Once it has collected a number of seconds equal to buffer_seconds, it sends that entire chunk to be processed by the model (assuming it passes the --energy_threshold).
*   **Shorter Buffer (e.g., 2-3s):**
    *   **Pro:** Lower latency. The transcribed text will appear more quickly after you speak, giving a more "live" feel.
    *   **Con:** Lower accuracy. The model has less context to work with, so it's more likely to make mistakes or produce fragmented, incoherent sentences.
*   **Longer Buffer (e.g., 8-10s):**
    *   **Pro:** Higher accuracy. By processing longer, more complete sentences, the model has more context and can produce much more accurate and well-structured text.
    *   **Con:** Higher latency. You have to wait longer after you finish speaking to see the output, which can feel less interactive.
Recommendation: The default of 5 seconds is a good starting point. Adjust it based on your preference for speed versus accuracy.
    *   **Note:** Whisper processes audio in 30-second segments, so the buffer will be padded to 30 seconds anyway. A longer buffer provides more *actual* data within that window.

### --energy_threshold
This parameter controls the script's sensitivity to sound. It is the core of a simple Voice Activity Detection (VAD) system, which helps the script decide whether a given chunk of audio contains speech or just silence/background noise.

*   **How it works:** The script continuously records audio from your selected microphone, temporarily storing it in a buffer.
Once the audio in the buffer reaches the length defined by --buffer_seconds, the script calculates the average "energy" or "loudness" of that entire audio chunk. This is technically the Root Mean Square (RMS) of the audio signal.
The calculated energy value is then compared to the --energy_threshold you've set.
If the audio's energy is ABOVE the threshold, the script concludes that it contains speech. It then sends this audio chunk to the Whisper model for processing (transcription or translation).
If the audio's energy is BELOW the threshold, the script assumes it's just background noise or silence. It discards this audio chunk and waits for a louder signal.
*   **How to Tune it:** This value is essentially a "loudness" cutoff. You may need to adjust it based on your microphone's sensitivity and your environment.

Increase the value (e.g., to `0.01`, `0.02`) if you are in a noisy environment and the script is incorrectly trying to transcribe background noise (like a fan, keyboard clicks, or distant chatter). This makes the script *less sensitive*.
Decrease the value (e.g., to `0.003`, `0.001`) if the script is failing to pick up your voice, especially if you speak softly or are far from the microphone. This makes the script *more sensitive*.
The default value of 0.005 is a balanced starting point, but feel free to experiment to find the optimal value for your specific setup.

## Original Whisper Project

*   https://github.com/openai/whisper

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification.

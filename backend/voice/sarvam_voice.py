"""
Part 4: Voice AI — Sarvam AI Integration
Hindi STT (Speech-to-Text) and TTS (Text-to-Speech).
Sarvam AI specializes in Indian languages.
"""
import httpx
import base64
import json
from backend.config import get_settings

SARVAM_BASE_URL = "https://api.sarvam.ai"


class SarvamVoiceClient:
    def __init__(self):
        self.api_key = get_settings().sarvam_api_key
        self.headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }

    async def speech_to_text(self, audio_bytes: bytes, language: str = "hi-IN") -> str:
        """
        Convert Hindi speech to text using Sarvam AI Saarika model.
        Args:
            audio_bytes: Audio file bytes (wav/mp3/webm)
            language: BCP-47 language code (hi-IN for Hindi)
        Returns:
            Transcribed text string
        """
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        payload = {
            "model": "saarika:v2",
            "language_code": language,
            "audio": audio_b64,
            "with_timestamps": False,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{SARVAM_BASE_URL}/speech-to-text",
                headers=self.headers,
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get("transcript", "")

    async def text_to_speech(
        self,
        text: str,
        language: str = "hi-IN",
        speaker: str = "meera",
        speed: float = 1.0,
    ) -> bytes:
        """
        Convert Hindi text to natural speech using Sarvam AI Bulbul model.
        Args:
            text: Hindi/Hinglish text to speak (max 500 chars per request)
            language: BCP-47 language code
            speaker: Voice preset (meera=female, arjun=male)
            speed: Playback speed (0.5 to 2.0)
        Returns:
            WAV audio bytes
        """
        # Split long text into chunks (Sarvam limit: 500 chars)
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        all_audio = b""

        async with httpx.AsyncClient(timeout=30.0) as client:
            for chunk in chunks:
                payload = {
                    "inputs": [chunk],
                    "target_language_code": language,
                    "speaker": speaker,
                    "pitch": 0,
                    "pace": speed,
                    "loudness": 1.5,
                    "speech_sample_rate": 8000,
                    "enable_preprocessing": True,
                    "model": "bulbul:v1",
                }
                resp = await client.post(
                    f"{SARVAM_BASE_URL}/text-to-speech",
                    headers=self.headers,
                    json=payload,
                )
                resp.raise_for_status()
                result = resp.json()
                if result.get("audios"):
                    audio_b64 = result["audios"][0]
                    all_audio += base64.b64decode(audio_b64)

        return all_audio

    async def translate(self, text: str, source: str = "hi-IN", target: str = "en-IN") -> str:
        """
        Translate between Hindi and English using Sarvam AI Mayura model.
        """
        payload = {
            "input": text,
            "source_language_code": source,
            "target_language_code": target,
            "model": "mayura:v1",
            "enable_preprocessing": False,
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{SARVAM_BASE_URL}/translate",
                headers=self.headers,
                json=payload,
            )
            resp.raise_for_status()
            return resp.json().get("translated_text", text)


# Singleton
_sarvam_client: SarvamVoiceClient | None = None


def get_sarvam_client() -> SarvamVoiceClient:
    global _sarvam_client
    if _sarvam_client is None:
        _sarvam_client = SarvamVoiceClient()
    return _sarvam_client

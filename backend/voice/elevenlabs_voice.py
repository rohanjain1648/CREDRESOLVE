"""
Part 4: Voice AI — ElevenLabs Integration
High-quality TTS for Hindi voice output with emotion control.
"""
import httpx
import json
from backend.config import get_settings

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# Built-in "premade" voice usable on the free tier (library/community voices
# return 402 paid_plan_required for free accounts).
FREE_TIER_FALLBACK_VOICE = "EXAVITQu4vr4xnSDxMaL"  # Sarah (multilingual-capable)


class ElevenLabsClient:
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.elevenlabs_api_key
        self.default_voice_id = settings.elevenlabs_voice_id or "EXAVITQu4vr4xnSDxMaL"
        self.headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

    async def text_to_speech(
        self,
        text: str,
        voice_id: str = None,
        stability: float = 0.5,
        similarity_boost: float = 0.8,
        style: float = 0.2,
    ) -> bytes:
        """
        Convert text to high-quality speech via ElevenLabs.
        Args:
            text: Text to synthesize
            voice_id: ElevenLabs voice ID (use Hindi-capable multilingual voice)
            stability: Voice stability 0-1 (higher = more consistent)
            similarity_boost: Speaker similarity 0-1
            style: Expressiveness 0-1
        Returns:
            MP3 audio bytes
        """
        vid = voice_id or self.default_voice_id
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",  # Supports Hindi
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": True,
            },
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{ELEVENLABS_BASE_URL}/text-to-speech/{vid}",
                headers=self.headers,
                json=payload,
            )
            # Free accounts can't use library/community voices (402). Retry once
            # with a built-in premade voice so TTS still works.
            if resp.status_code == 402 and vid != FREE_TIER_FALLBACK_VOICE:
                resp = await client.post(
                    f"{ELEVENLABS_BASE_URL}/text-to-speech/{FREE_TIER_FALLBACK_VOICE}",
                    headers=self.headers,
                    json=payload,
                )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"ElevenLabs TTS {resp.status_code}: {resp.text[:300]}"
                )
            return resp.content

    async def get_voices(self) -> list[dict]:
        """List available voices."""
        headers = {"xi-api-key": self.api_key}
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ELEVENLABS_BASE_URL}/voices", headers=headers)
            resp.raise_for_status()
            return resp.json().get("voices", [])


_elevenlabs_client: ElevenLabsClient | None = None


def get_elevenlabs_client() -> ElevenLabsClient:
    global _elevenlabs_client
    if _elevenlabs_client is None:
        _elevenlabs_client = ElevenLabsClient()
    return _elevenlabs_client

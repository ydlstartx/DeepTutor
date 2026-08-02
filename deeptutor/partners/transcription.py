"""Voice transcription provider using Groq."""

import asyncio
import os
from pathlib import Path

import httpx
from loguru import logger

# Reused across provider instances (channels construct one per transcription
# call). Keyed by event loop because httpx's connection pool binds to the
# loop that created it — matches the LLM provider pool's loop-keyed pattern.
_clients: dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}


def _get_client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is None:
        client = httpx.AsyncClient(timeout=60.0)
        _clients[loop] = client
    return client


class GroqTranscriptionProvider:
    """
    Voice transcription provider using Groq's Whisper API.

    Groq offers extremely fast transcription with a generous free tier.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using Groq.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        try:
            client = _get_client()
            with open(path, "rb") as f:
                files = {
                    "file": (path.name, f),
                    "model": (None, "whisper-large-v3"),
                }
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                }

                response = await client.post(
                    self.api_url, headers=headers, files=files
                )

                response.raise_for_status()
                data = response.json()
                return data.get("text", "")

        except Exception as e:
            logger.error("Groq transcription error: {}", e)
            return ""

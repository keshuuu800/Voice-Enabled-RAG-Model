"""
Sarvam Saaras v3 Speech-to-Text service.
Uses Sarvam REST API. API key must not be exposed to the browser.
"""
import time
import logging
import httpx
from app.core.config import get_settings

class SpeechService:
    def __init__(self, api_key: str = None):
        settings = get_settings()
        self.api_key = api_key or settings.sarvam_api_key
        self.base_url = 'https://api.sarvam.ai'
        self.logger = logging.getLogger(__name__)
        self._available = bool(self.api_key)  # Flag for health check

    async def transcribe(self, audio_data: bytes, filename: str = 'audio.wav', language: str = 'auto') -> dict:
        """
        Send audio to Sarvam Saaras v3 and return transcript.
        Returns: {text: str, language: str, mode: str, latency_ms: float}
        """
        t0 = time.perf_counter()
        
        if not self.api_key:
            self.logger.warning('SARVAM_API_KEY not set. Returning mock transcript.')
            return {
                'text': '[MOCK TRANSCRIPT — set SARVAM_API_KEY in .env]',
                'language': 'en-IN',
                'mode': 'mock',
                'latency_ms': 0.0
            }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {'file': (filename, audio_data, 'audio/wav')}
            data = {
                'model': 'saaras:v3',
                'language_code': self._map_language(language),
                'with_timestamps': 'false',
                'with_disfluencies': 'false',
                'mode': 'transcribe'
            }
            headers = {'api-subscription-key': self.api_key}
            
            response = await client.post(
                f'{self.base_url}/speech-to-text',
                headers=headers,
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()
        
        latency_ms = (time.perf_counter() - t0) * 1000
        transcript = result.get('transcript', '')
        detected_lang = result.get('language_code', language)
        
        return {
            'text': transcript,
            'language': detected_lang,
            'mode': 'transcribe',
            'latency_ms': latency_ms
        }

    def _map_language(self, lang: str) -> str:
        # Map common language codes to Sarvam format
        mapping = {'auto': 'unknown', 'en': 'en-IN', 'hi': 'hi-IN', 'unknown': 'unknown'}
        return mapping.get(lang, lang)  # Pass through if already in correct format

    def is_available(self) -> bool:
        return self._available

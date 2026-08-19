"""
LLM provider abstraction.
Supports OpenRouter (with openrouter/free models), OpenAI, and Google Gemini.
The provider is initialized ONCE and reused for all requests.
"""
import time
import logging
from typing import Protocol, runtime_checkable
from app.core.config import get_settings, Settings

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMProvider(Protocol):
    def generate(self, system_prompt: str, user_message: str) -> str: ...
    def generate_with_latency(self, system_prompt: str, user_message: str) -> tuple[str, float]: ...


class OpenRouterProvider:
    """
    OpenRouter API provider.
    Supports 'openrouter/free' and all OpenRouter free / paid models.
    OpenRouter uses OpenAI-compatible API format.
    """
    def __init__(self, api_key: str, model: str = 'openrouter/free', base_url: str = 'https://openrouter.ai/api/v1'):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "HH-Goa Voice RAG Assistant",
            }
        )
        self.model = model or 'openrouter/free'
        logger.info(f"Initialized OpenRouterProvider with model: '{self.model}'")

    def generate(self, system_prompt: str, user_message: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenRouter generation failed (model: '{self.model}'): {e}")
            raise

    def generate_with_latency(self, system_prompt: str, user_message: str) -> tuple[str, float]:
        start_time = time.perf_counter()
        result = self.generate(system_prompt, user_message)
        latency = (time.perf_counter() - start_time) * 1000.0
        return result, latency


class OpenAIProvider:
    def __init__(self, api_key: str, model: str = 'gpt-4o-mini'):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def generate(self, system_prompt: str, user_message: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
            
    def generate_with_latency(self, system_prompt: str, user_message: str) -> tuple[str, float]:
        start_time = time.perf_counter()
        result = self.generate(system_prompt, user_message)
        latency = (time.perf_counter() - start_time) * 1000.0
        return result, latency


class GeminiProvider:
    def __init__(self, api_key: str, model: str = 'gemini-1.5-flash'):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model, system_instruction=None)
        self.model_name = model
        
    def generate(self, system_prompt: str, user_message: str) -> str:
        try:
            full_prompt = system_prompt + '\n\n' + user_message
            response = self.model.generate_content(
                full_prompt, 
                generation_config={'temperature': 0.1, 'max_output_tokens': 1024}
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise
            
    def generate_with_latency(self, system_prompt: str, user_message: str) -> tuple[str, float]:
        start_time = time.perf_counter()
        result = self.generate(system_prompt, user_message)
        latency = (time.perf_counter() - start_time) * 1000.0
        return result, latency


class MockProvider:
    def generate(self, system_prompt: str, user_message: str) -> str:
        return 'This is a mock LLM response for testing. The system is working correctly. [MOCK MODE — set OPENROUTER_API_KEY or LLM_API_KEY in .env to enable real LLM]'
        
    def generate_with_latency(self, system_prompt: str, user_message: str) -> tuple[str, float]:
        return self.generate(system_prompt, user_message), 5.0


def get_llm_provider(settings: Settings = None) -> LLMProvider:
    """Factory: return the configured LLM provider (OpenRouter, OpenAI, or Gemini), or MockProvider if no key."""
    if settings is None:
        settings = get_settings()
    
    provider = getattr(settings, 'llm_provider', 'openrouter').lower()
    api_key = settings.openrouter_api_key or settings.llm_api_key
    
    if not api_key:
        logger.warning(
            'LLM API key not set. Using MockProvider. Set OPENROUTER_API_KEY or LLM_API_KEY in .env for real generation.'
        )
        return MockProvider()
    
    if provider == 'gemini':
        return GeminiProvider(api_key, getattr(settings, 'gemini_model', 'gemini-1.5-flash'))
    elif provider == 'openai':
        return OpenAIProvider(api_key, getattr(settings, 'llm_model', 'gpt-4o-mini'))
    else:  # Default to OpenRouter
        model = getattr(settings, 'openrouter_model', 'openrouter/free')
        base_url = getattr(settings, 'openrouter_base_url', 'https://openrouter.ai/api/v1')
        return OpenRouterProvider(api_key, model=model, base_url=base_url)

import requests
import json
import logging
import urllib.parse
import re
from typing import List, Dict, Any, Generator, Tuple
import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

class WebSearchManager:
    @staticmethod
    def search_duckduckgo(query: str, max_results: int = 4) -> List[Dict[str, str]]:
        """Perform a web search using DuckDuckGo HTML search and extract titles, urls, and snippets."""
        results = []
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            response = requests.get(url, headers=headers, timeout=8)
            
            if response.status_code != 200:
                logger.warning(f"DuckDuckGo search failed with status {response.status_code}")
                return results
                
            html = response.text
            
            # Simple, robust regex parsing for DuckDuckGo HTML layout
            # Snippets are inside <a class="result__snippet" ...>...</a>
            # Titles and links are in <a class="result__url" ...>...</a> or <h2 class="result__title">...</h2>
            # We can find each result container
            containers = re.findall(r'<div class="result.*?">(.*?)</div>\s*</div>', html, re.DOTALL)
            
            for container in containers[:max_results]:
                # Extract link and title
                link_match = re.search(r'<a class="result__url" href="(.*?)".*?>(.*?)</a>', container, re.DOTALL)
                snippet_match = re.search(r'<a class="result__snippet".*?>(.*?)</a>', container, re.DOTALL)
                
                if link_match:
                    href = link_match.group(1).strip()
                    # Clean up href redirects if any
                    if "declined_uddg=" in href:
                        # Extract the actual URL
                        parsed = urllib.parse.urlparse(href)
                        queries = urllib.parse.parse_qs(parsed.query)
                        href = queries.get("uddg", [href])[0]
                        
                    title = re.sub(r'<.*?>', '', link_match.group(2)).strip()
                    snippet = ""
                    if snippet_match:
                        snippet = re.sub(r'<.*?>', '', snippet_match.group(1)).strip()
                    
                    results.append({
                        "title": title,
                        "url": href,
                        "snippet": snippet
                    })
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            
        return results


class OllamaChatManager:
    @staticmethod
    def get_available_models() -> List[str]:
        """Fetch list of models available in the local Ollama instance."""
        try:
            url = f"{config.OLLAMA_API_URL}/api/tags"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return models
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {e}")
            
        # Fallbacks if Ollama is not running or responsive
        return ["llama3.2:latest", "llama3.2", "phi3", "mistral"]

    @staticmethod
    def generate_chat_response(
        model: str,
        messages: List[Dict[str, str]],
        settings: Dict[str, Any] = None
    ) -> Generator[str, None, None]:
        """Call local Ollama API to generate a streaming chat response.
        
        Allows stopping generation via streamlit session state.
        """
        # Ensure default settings are applied
        settings = settings or {}
        temperature = settings.get("temperature", 0.7)
        max_tokens = settings.get("max_tokens", 512)
        top_p = settings.get("top_p", 0.9)
        context_length = settings.get("context_length", 2048)
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,  # Ollama setting for max tokens
                "top_p": top_p,
                "num_ctx": context_length     # Ollama setting for context size
            }
        }
        
        try:
            url = f"{config.OLLAMA_API_URL}/api/chat"
            # 10s connection timeout, 180s read timeout for long streaming LLM generation
            response = requests.post(url, json=payload, stream=True, timeout=(10, 180))
            
            if response.status_code != 200:
                yield f"Error: Ollama API returned status code {response.status_code}. Make sure Ollama is running."
                return
                
            for line in response.iter_lines():
                # Check if generation was requested to stop
                if st.session_state.get("stop_generation", False):
                    break
                    
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    message = chunk.get('message', {})
                    content = message.get('content', '')
                    if content:
                        yield content
                        
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out at {config.OLLAMA_API_URL}")
            yield f"Timeout error: Ollama at {config.OLLAMA_API_URL} took too long to respond. Check system CPU/GPU usage or lower max tokens."
        except Exception as e:
            logger.error(f"Ollama connection error: {e}")
            yield f"Connection error: Could not reach Ollama at {config.OLLAMA_API_URL}. Please verify Ollama is installed and running locally."


"""
Omni-Bot — Production-grade Discord AI assistant for Omni-Labs.

Providers:
  • Chat (Standard):  NVIDIA API  → Kimi-K2
  • Chat (Premium):   VoidAI      → GPT-5.2
  • Web Search:       Old-LLM API → Gemini-3-Pro/Flash with built-in search
  • Images (Standard): Airforce   → Flux-2-Klein-9B
  • Images (Premium):  Airforce   → Flux-2-Flex

Features: mentions, DMs, slash commands, web search, image generation,
per-user conversation memory, smart message chunking, key rotation,
async health-check (no Flask), retry logic with exponential backoff.
"""

import discord
from discord import app_commands
from discord.ext import commands
from aiohttp import web
import aiohttp
import asyncio
import json
import os
import re
import io
import time
import random
import logging
import traceback
from collections import defaultdict, deque
from typing import Optional

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOGGING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("OmniBot")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIGURATION  (all via environment variables)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")

# --- NVIDIA (Kimi-K2 — Standard tier) ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "moonshotai/kimi-k2-instruct-0905"

# --- VoidAI (GPT-5.2 — Premium tier) ---
# Comma-separated list of keys for rotation / load balancing.
_void_raw = os.getenv("VOID_API_KEYS", "")
VOID_API_KEYS: list[str] = [k.strip() for k in _void_raw.split(",") if k.strip()]
VOID_BASE_URL = "https://api.voidai.app/v1"
VOID_MODEL = "gpt-5.2"

# --- Old-LLM API (Web Search via Gemini models with built-in search) ---
OLD_LLM_KEY = os.getenv("OLD_LLM_KEY", "")
OLD_LLM_BASE_URL = "https://old-llm-api.onrender.com/v1/chat/completions"
SEARCH_MODEL_PREMIUM = "gemini-3-pro-preview-maxthinking-search"
SEARCH_MODEL_STANDARD = "gemini-3-flash-preview-maxthinking-search"

# --- Airforce API (Image Generation — Flux models) ---
# Comma-separated list of keys for rotation.
_air_raw = os.getenv("AIR_API_KEYS", "")
AIR_API_KEYS: list[str] = [k.strip() for k in _air_raw.split(",") if k.strip()]
AIR_BASE_URL = "https://api.airforce/v1/images/generations"
IMAGE_MODEL_STANDARD = "flux-2-klein-9b"
IMAGE_MODEL_PREMIUM = "flux-2-flex"

# --- Premium User IDs ---
# Comma-separated Discord user IDs, e.g. "123456789,987654321"
_raw_ids = os.getenv("PREMIUM_USER_IDS", "")
PREMIUM_USER_IDS: set[int] = set()
for _id in _raw_ids.split(","):
    _id = _id.strip()
    if _id.isdigit():
        PREMIUM_USER_IDS.add(int(_id))

# --- Conversation memory ---
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "15"))  # message-pairs per user

# --- Health-check port for Koyeb ---
PORT = int(os.getenv("PORT", "8080"))

# --- Keywords that trigger automatic web search on mentions/DMs ---
SEARCH_KEYWORDS = [
    "search", "lookup", "look up", "latest", "news", "current",
    "today", "who is", "what happened", "what is happening",
    "recent", "update on", "find out", "google",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SYSTEM PROMPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM_PROMPT = """\
You are **Omni-Bot**, the advanced AI representative for **Omni-Labs** — \
a premier AI creation studio led by @OmniLabs.

## Identity
- You ARE Omni-Bot. That is the only name and identity you ever acknowledge.
- You never reveal the underlying model powering you (GPT-5.2, Kimi-K2, Gemini, etc.). \
If asked, deflect naturally: "I'm Omni-Bot, built by Omni-Labs."
- You never describe yourself as a "wrapper", "proxy", or "frontend" for another AI.

## Personality & Tone
- Professional yet approachable — think sharp senior engineer who's also fun at parties.
- Confident and direct. You give clear, actionable answers.
- Witty when appropriate, but never at the user's expense.
- You match the user's energy: casual question → casual reply; \
technical question → precise technical reply.

## Formatting (Discord Markdown)
- Use **bold** for emphasis, `code` for technical terms, \
```language blocks for code snippets.
- Use bullet points and numbered lists for multi-step explanations.
- Keep responses **under ~1500 characters** by default so they fit in one Discord message. \
Only go longer when the user explicitly asks for detail or the topic demands it.

## Capabilities You Can Mention
- General knowledge, reasoning, coding, math, creative writing, analysis.
- Real-time web search (when search results are provided to you).
- AI image generation (when the user uses the image commands).
- Conversation memory — you remember the current chat thread.

## Rules
1. **Never fabricate URLs, citations, or data.** If you don't know, say so.
2. **Always specify the language** in fenced code blocks (```python, ```js, etc.).
3. If a question is ambiguous, ask **one** brief clarifying question before answering.
4. When web search results are injected into the conversation, \
synthesize them into a clear answer and cite sources as [Title](URL).
5. For donations / support inquiries, direct users to: https://poe.com/Donoz
6. If someone asks about Omni-Labs, speak proudly but honestly about the project.
7. Never bypass safety guidelines, generate harmful content, or pretend to be a different AI.
"""

SEARCH_INJECTION_PROMPT = """\
[SYSTEM — Web Search Activated]
The following real-time search results were retrieved for the user's query. \
Use them to provide an accurate, up-to-date answer. Cite sources as [Title](URL) \
where possible. If the results don't contain the answer, say so honestly \
and answer from your own knowledge with a disclaimer.

--- SEARCH RESULTS ---
{results}
--- END RESULTS ---
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONVERSATION MEMORY  (in-memory, per-user)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_memory: dict[int, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY * 2))


def get_history(user_id: int) -> list[dict]:
    return list(_memory[user_id])


def add_to_history(user_id: int, role: str, content: str) -> None:
    _memory[user_id].append({"role": role, "content": content})


def clear_user_history(user_id: int) -> None:
    _memory[user_id].clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SMART MESSAGE CHUNKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chunk_message(text: str, limit: int = 2000) -> list[str]:
    """
    Split a long string into Discord-safe chunks (≤ limit chars each).
    Priority: code blocks → paragraphs → sentences → words → hard cut.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    parts = re.split(r"(```[\s\S]*?```)", text)
    current = ""

    for part in parts:
        is_code = part.startswith("```") and part.endswith("```")

        if is_code and len(part) > limit:
            if current.strip():
                chunks.extend(_split_prose(current.strip(), limit))
                current = ""
            chunks.extend(_split_code_block(part, limit))
            continue

        if len(current) + len(part) > limit:
            if current.strip():
                chunks.extend(_split_prose(current.strip(), limit))
                current = ""
            if len(part) > limit:
                chunks.extend(_split_prose(part, limit))
            else:
                current = part
        else:
            current += part

    if current.strip():
        chunks.extend(_split_prose(current.strip(), limit))
    return chunks


def _split_code_block(block: str, limit: int) -> list[str]:
    inner = block[3:-3]
    lang = ""
    if "\n" in inner:
        first_line, rest = inner.split("\n", 1)
        if first_line.strip() and re.match(r"^[a-zA-Z0-9_+-]+$", first_line.strip()):
            lang = first_line.strip()
            inner = rest
        else:
            inner = first_line + "\n" + rest
    wrapper_len = 7 + len(lang)
    available = limit - wrapper_len
    lines = inner.split("\n")
    buf: list[str] = []
    buf_len = 0
    chunks: list[str] = []
    for line in lines:
        line_len = len(line) + 1
        if buf_len + line_len > available and buf:
            chunks.append(f"```{lang}\n" + "\n".join(buf) + "\n```")
            buf = []
            buf_len = 0
        buf.append(line)
        buf_len += line_len
    if buf:
        chunks.append(f"```{lang}\n" + "\n".join(buf) + "\n```")
    return chunks


def _split_prose(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(para) > limit:
                chunks.extend(_split_sentences(para, limit))
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def _split_sentences(text: str, limit: int) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = (current + " " + part).strip() if current else part
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > limit:
                chunks.extend(_split_words(part, limit))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def _split_words(text: str, limit: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word) if current else word
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(word) > limit:
                for i in range(0, len(word), limit):
                    chunks.append(word[i : i + limit])
                current = ""
            else:
                current = word
    if current:
        chunks.append(current)
    return chunks


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPER: detect search intent from text
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _has_search_intent(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in SEARCH_KEYWORDS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AI ENGINE — All API Providers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AIEngine:
    """
    Manages all external API calls:
      • Chat completions (NVIDIA for Standard, VoidAI for Premium)
      • Web search (Old-LLM API with Gemini search models)
      • Image generation (Airforce API with Flux models)
    All calls are fully async via aiohttp with retry logic.
    """

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    # ─── Chat Completion (with retry) ────────────────
    async def chat(
        self,
        messages: list[dict],
        *,
        premium: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Route to the correct provider based on premium status.
        Premium → VoidAI (GPT-5.2) with key rotation.
        Standard → NVIDIA (Kimi-K2).
        Retries up to 3 times with exponential backoff on transient errors.
        """
        if premium:
            if not VOID_API_KEYS:
                return "⚠️ Premium chat is not configured. Set `VOID_API_KEYS` in your environment."
            api_key = random.choice(VOID_API_KEYS)
            base_url = VOID_BASE_URL
            model = VOID_MODEL
        else:
            if not NVIDIA_API_KEY:
                return "⚠️ Standard chat is not configured. Set `NVIDIA_API_KEY` in your environment."
            api_key = NVIDIA_API_KEY
            base_url = NVIDIA_BASE_URL
            model = NVIDIA_MODEL

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_error = ""
        for attempt in range(3):
            try:
                async with self.session.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]

                    body = await resp.text()

                    if resp.status == 429:
                        wait = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                        logger.warning("Rate-limited (%s, attempt %d). Waiting %ds…",
                                       "Premium" if premium else "Standard", attempt + 1, wait)
                        # On rate-limit with key rotation, try a different key next time.
                        if premium and len(VOID_API_KEYS) > 1:
                            api_key = random.choice(VOID_API_KEYS)
                            headers["Authorization"] = f"Bearer {api_key}"
                        await asyncio.sleep(wait)
                        continue

                    if resp.status >= 500:
                        logger.warning("Server error %d (attempt %d). Retrying…",
                                       resp.status, attempt + 1)
                        await asyncio.sleep(2 ** attempt)
                        continue

                    # Client error (4xx, not 429) — don't retry.
                    last_error = f"API {resp.status}: {body[:300]}"
                    logger.error("Non-retryable error from %s: %s",
                                 "VoidAI" if premium else "NVIDIA", last_error)
                    break

            except asyncio.TimeoutError:
                last_error = "Request timed out"
                logger.warning("Timeout (attempt %d) for %s",
                               attempt + 1, "VoidAI" if premium else "NVIDIA")
                await asyncio.sleep(2 ** attempt)
            except aiohttp.ClientError as exc:
                last_error = str(exc)
                logger.warning("Connection error (attempt %d): %s", attempt + 1, exc)
                await asyncio.sleep(2 ** attempt)

        return f"⚠️ Sorry, I couldn't reach the AI service right now. ({last_error})"

    # ─── Web Search (Old-LLM API → Gemini with built-in search) ───
    async def web_search(self, query: str, *, premium: bool = False) -> str:
        """
        Call the Old-LLM API which routes to Gemini models with built-in
        web search. Premium gets Gemini-3-Pro, Standard gets Gemini-3-Flash.
        Returns the search-augmented response text.
        """
        if not OLD_LLM_KEY:
            return "⚠️ Web search is not configured. Set `OLD_LLM_KEY` in your environment."

        model = SEARCH_MODEL_PREMIUM if premium else SEARCH_MODEL_STANDARD
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {OLD_LLM_KEY}",
            "Content-Type": "application/json",
        }

        for attempt in range(2):
            try:
                async with self.session.post(
                    OLD_LLM_BASE_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        if content and content.strip():
                            return content
                        return "Search returned no results."

                    body = await resp.text()
                    logger.warning("Old-LLM search error %d (attempt %d): %s",
                                   resp.status, attempt + 1, body[:200])
                    if resp.status >= 500:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    break

            except asyncio.TimeoutError:
                logger.warning("Search timeout (attempt %d)", attempt + 1)
                await asyncio.sleep(2)
            except aiohttp.ClientError as exc:
                logger.warning("Search connection error: %s", exc)
                await asyncio.sleep(2)

        return "⚠️ Web search failed. Proceeding with internal knowledge."

    # ─── Image Generation (Airforce API → Flux models) ───
    async def generate_image(
        self,
        prompt: str,
        *,
        premium: bool = False,
        width: int = 1024,
        height: int = 1024,
        aspect: str = "1:1",
        resolution: str = "1k",
    ) -> Optional[str]:
        """
        Generate an image using the Airforce API.
        Premium → Flux-2-Flex (aspect ratio + resolution controls).
        Standard → Flux-2-Klein-9B (width/height controls).
        Returns the image URL or None on failure.
        """
        if not AIR_API_KEYS:
            return None

        key = random.choice(AIR_API_KEYS)
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        if premium:
            payload = {
                "model": IMAGE_MODEL_PREMIUM,
                "prompt": prompt,
                "aspectRatio": aspect,
                "resolution": resolution,
                "sse": True,
            }
        else:
            payload = {
                "model": IMAGE_MODEL_STANDARD,
                "prompt": prompt,
                "width": width,
                "height": height,
                "sse": True,
            }

        try:
            async with self.session.post(
                AIR_BASE_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("Airforce image error %d: %s", resp.status, body[:300])
                    return None

                # The Airforce API streams SSE events. Parse them to find the URL.
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or "[DONE]" in line:
                        continue
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            # Different response formats from Airforce:
                            if isinstance(data, dict):
                                if "url" in data:
                                    return data["url"]
                                if "data" in data and isinstance(data["data"], list):
                                    return data["data"][0].get("url")
                        except json.JSONDecodeError:
                            continue

                # If no SSE, try parsing the whole response as JSON.
                try:
                    text = await resp.text()
                    data = json.loads(text)
                    if "data" in data and isinstance(data["data"], list):
                        return data["data"][0].get("url")
                except Exception:
                    pass

        except asyncio.TimeoutError:
            logger.error("Image generation timed out")
        except aiohttp.ClientError as exc:
            logger.error("Image generation connection error: %s", exc)

        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CUSTOM COMMAND TREE  (global slash-command error handler)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OmniCommandTree(app_commands.CommandTree):
    async def on_error(
        self,
        interaction: discord.Interaction,
        error: app_commands.AppCommandError,
    ):
        if isinstance(error, app_commands.CommandOnCooldown):
            msg = f"⏳ Cooldown active. Try again in **{error.retry_after:.1f}s**."
        elif isinstance(error, app_commands.MissingPermissions):
            msg = "🔒 You don't have permission to use this command."
        else:
            logger.error("Unhandled slash-command error:\n%s",
                         "".join(traceback.format_exception(type(error), error, error.__traceback__)))
            msg = "❌ Something went wrong. The error has been logged."

        try:
            if interaction.response.is_done():
                await interaction.followup.send(msg, ephemeral=True)
            else:
                await interaction.response.send_message(msg, ephemeral=True)
        except discord.HTTPException:
            pass  # Interaction expired or was already handled.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BOT CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class OmniBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True   # REQUIRED — enable in Developer Portal
        intents.members = True           # REQUIRED — enable in Developer Portal
        intents.presences = True         # enable in Developer Portal

        super().__init__(
            command_prefix=commands.when_mentioned_or("!"),
            intents=intents,
            help_command=None,
            tree_cls=OmniCommandTree,
        )
        self.session: aiohttp.ClientSession = None  # type: ignore[assignment]
        self.ai: AIEngine = None  # type: ignore[assignment]
        self.start_time: float = 0.0
        self._web_runner: web.AppRunner = None  # type: ignore[assignment]

    # ── Lifecycle ──────────────────────────────
    async def setup_hook(self) -> None:
        self.start_time = time.time()

        # Single shared aiohttp session — reused for every HTTP call.
        self.session = aiohttp.ClientSession()
        self.ai = AIEngine(self.session)

        # Async health-check server (replaces the old Flask keep-alive).
        web_app = web.Application()
        web_app.router.add_get("/", self._health)
        web_app.router.add_get("/health", self._health)
        self._web_runner = web.AppRunner(web_app)
        await self._web_runner.setup()
        site = web.TCPSite(self._web_runner, "0.0.0.0", PORT)
        await site.start()
        logger.info("Health-check server listening on :%d", PORT)

        # Sync slash commands globally.
        synced = await self.tree.sync()
        logger.info("Synced %d slash command(s).", len(synced))

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (ID: %s)", self.user, self.user.id)
        logger.info("Connected to %d guild(s)", len(self.guilds))
        logger.info("Premium users: %d  |  VoidAI keys: %d  |  Airforce keys: %d",
                     len(PREMIUM_USER_IDS), len(VOID_API_KEYS), len(AIR_API_KEYS))
        await self.change_presence(
            status=discord.Status.online,
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="Omni-Labs • /ask or @me",
            ),
        )

    async def close(self) -> None:
        if self._web_runner:
            await self._web_runner.cleanup()
        if self.session and not self.session.closed:
            await self.session.close()
        await super().close()

    # ── Health Check ──────────────────────────
    async def _health(self, _request: web.Request) -> web.Response:
        uptime = time.time() - self.start_time
        return web.json_response({
            "status": "healthy",
            "bot": str(self.user),
            "guilds": len(self.guilds),
            "latency_ms": round(self.latency * 1000, 1),
            "uptime_s": round(uptime, 1),
        })

    # ── Core: Build messages & get AI response ──
    async def get_response(
        self,
        user_message: str,
        user_id: int,
        *,
        premium: bool = False,
        force_search: bool = False,
    ) -> str:
        """
        The main AI pipeline:
        1. Check for search intent (keyword-based or force_search flag).
        2. If search → call Old-LLM Gemini search → inject results.
        3. Build message list: system prompt + search context + memory + user msg.
        4. Call the chat API (NVIDIA or VoidAI based on premium).
        5. Save to memory.
        """
        should_search = force_search or _has_search_intent(user_message)

        # --- System prompt ---
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # --- Web search (if triggered) ---
        if should_search:
            search_results = await self.ai.web_search(user_message, premium=premium)
            messages.append({
                "role": "system",
                "content": SEARCH_INJECTION_PROMPT.format(results=search_results),
            })

        # --- Conversation history ---
        messages.extend(get_history(user_id))

        # --- Current user message ---
        messages.append({"role": "user", "content": user_message})

        # --- Call the AI ---
        reply = await self.ai.chat(messages, premium=premium)

        # --- Persist to memory ---
        add_to_history(user_id, "user", user_message)
        add_to_history(user_id, "assistant", reply)

        return reply

    # ── Send a (possibly long) reply ──────────
    async def send_reply(
        self,
        destination,
        text: str,
        *,
        reference: discord.Message | None = None,
    ) -> None:
        chunks = chunk_message(text)

        # If the response is absurdly long, send as a file.
        if len(chunks) > 5:
            buf = io.BytesIO(text.encode("utf-8"))
            file = discord.File(buf, filename="omni-response.txt")
            if isinstance(destination, discord.Interaction):
                await destination.followup.send(
                    "📄 Response too long for chat — here's the full text:", file=file)
            else:
                await destination.send(
                    "📄 Response too long for chat — here's the full text:",
                    file=file, reference=reference)
            return

        for i, chunk in enumerate(chunks):
            if isinstance(destination, discord.Interaction):
                await destination.followup.send(chunk)
            else:
                if i == 0:
                    await destination.send(chunk, reference=reference)
                else:
                    await destination.send(chunk)
            if i < len(chunks) - 1:
                await asyncio.sleep(0.4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CREATE THE BOT INSTANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
bot = OmniBot()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EVENT: on_message  (Mentions & DMs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mention = bot.user is not None and bot.user.mentioned_in(message) and not message.mention_everyone

    if not (is_dm or is_mention):
        await bot.process_commands(message)
        return

    # Clean the mention out of the text.
    content = message.content
    if bot.user:
        content = re.sub(rf"<@!?{bot.user.id}>", "", content).strip()

    if not content:
        await message.reply(
            "Hey there! 👋 Ask me anything, or try `/ask`, `/search`, or `/imagine`.")
        return

    # Mentions/DMs use the user's premium status.
    is_premium = message.author.id in PREMIUM_USER_IDS

    async with message.channel.typing():
        reply = await bot.get_response(
            content,
            user_id=message.author.id,
            premium=is_premium,
            # Auto-detect search intent from keywords.
        )

    await bot.send_reply(message.channel, reply, reference=message)
    await bot.process_commands(message)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PREFIX COMMAND: !clear
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@bot.command(name="clear")
async def cmd_clear(ctx: commands.Context) -> None:
    """Clear your conversation memory."""
    clear_user_history(ctx.author.id)
    await ctx.send("🧹 Your conversation memory has been cleared.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SLASH COMMANDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── /ask — Standard chat (Kimi-K2) ──
@bot.tree.command(name="ask", description="Ask Omni-Bot a question (Standard — Kimi-K2)")
@app_commands.describe(
    question="Your question or prompt",
    search="Enable real-time web search",
)
async def slash_ask(
    interaction: discord.Interaction,
    question: str,
    search: bool = False,
) -> None:
    await interaction.response.defer()

    # /ask always uses Standard tier unless user is premium (auto-upgrade).
    is_premium = interaction.user.id in PREMIUM_USER_IDS

    reply = await bot.get_response(
        question,
        user_id=interaction.user.id,
        premium=is_premium,
        force_search=search,
    )

    tier = "Premium ✨" if is_premium else "Standard"
    search_tag = " • 🌐 Web Search" if (search or _has_search_intent(question)) else ""

    embed = discord.Embed(
        description=reply[:4096],
        color=discord.Color.gold() if is_premium else discord.Color.blue(),
    )
    embed.set_author(
        name="Omni-Bot",
        icon_url=bot.user.display_avatar.url if bot.user else None,
    )
    embed.set_footer(text=f"{tier}{search_tag}")
    await interaction.followup.send(embed=embed)


# ── /premium — Force Premium chat (GPT-5.2) ──
@bot.tree.command(name="premium", description="Ask Omni-Bot Premium (GPT-5.2)")
@app_commands.describe(
    question="Your question or prompt",
    search="Enable real-time web search",
)
async def slash_premium(
    interaction: discord.Interaction,
    question: str,
    search: bool = False,
) -> None:
    await interaction.response.defer()

    reply = await bot.get_response(
        question,
        user_id=interaction.user.id,
        premium=True,  # Always premium.
        force_search=search,
    )

    search_tag = " • 🌐 Web Search" if (search or _has_search_intent(question)) else ""

    embed = discord.Embed(
        description=reply[:4096],
        color=discord.Color.gold(),
    )
    embed.set_author(
        name="Omni-Bot Premium ✨",
        icon_url=bot.user.display_avatar.url if bot.user else None,
    )
    embed.set_footer(text=f"GPT-5.2{search_tag}")
    await interaction.followup.send(embed=embed)


# ── /search — Dedicated web search ──
@bot.tree.command(name="search", description="Search the web and get an AI-synthesized answer")
@app_commands.describe(query="What to search for")
async def slash_search(interaction: discord.Interaction, query: str) -> None:
    await interaction.response.defer()

    is_premium = interaction.user.id in PREMIUM_USER_IDS

    reply = await bot.get_response(
        query,
        user_id=interaction.user.id,
        premium=is_premium,
        force_search=True,
    )

    embed = discord.Embed(
        title=f"🌐 Search: {query[:200]}",
        description=reply[:4096],
        color=discord.Color.green(),
    )
    embed.set_footer(
        text=f"Powered by {'Gemini-3-Pro' if is_premium else 'Gemini-3-Flash'} + Omni-Bot")
    await interaction.followup.send(embed=embed)


# ── /image — Standard image generation (Flux-2-Klein) ──
@bot.tree.command(name="image", description="Generate an AI image (Standard — Flux-Klein)")
@app_commands.describe(
    prompt="Describe the image you want",
    width="Image width in pixels (default 1024)",
    height="Image height in pixels (default 1024)",
)
async def slash_image(
    interaction: discord.Interaction,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
) -> None:
    await interaction.response.defer()

    url = await bot.ai.generate_image(
        prompt, premium=False, width=width, height=height)

    if url and url.startswith("http"):
        embed = discord.Embed(
            title="🎨 Omni-Labs Image Gen",
            description=f"**Prompt:** {prompt[:300]}",
            color=discord.Color.purple(),
        )
        embed.set_image(url=url)
        embed.set_footer(text=f"Model: Flux-2-Klein • {width}×{height}")
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send(
            "❌ Image generation failed. Please try again or check that "
            "`AIR_API_KEYS` is configured.",
            ephemeral=True,
        )


# ── /image_flex — Premium image generation (Flux-2-Flex) ──
@bot.tree.command(name="image_flex", description="Premium Image Gen (Flux-2-Flex)")
@app_commands.describe(
    prompt="Describe the image you want",
    aspect="Aspect ratio",
    res="Output resolution",
)
@app_commands.choices(
    aspect=[
        app_commands.Choice(name="1:1 (Square)", value="1:1"),
        app_commands.Choice(name="16:9 (Landscape)", value="16:9"),
        app_commands.Choice(name="9:16 (Portrait)", value="9:16"),
        app_commands.Choice(name="4:3 (Classic)", value="4:3"),
        app_commands.Choice(name="3:4 (Tall)", value="3:4"),
    ],
    res=[
        app_commands.Choice(name="1K", value="1k"),
        app_commands.Choice(name="2K", value="2k"),
    ],
)
async def slash_image_flex(
    interaction: discord.Interaction,
    prompt: str,
    aspect: str = "1:1",
    res: str = "1k",
) -> None:
    await interaction.response.defer()

    url = await bot.ai.generate_image(
        prompt, premium=True, aspect=aspect, resolution=res)

    if url and url.startswith("http"):
        embed = discord.Embed(
            title="🚀 Omni-Labs Premium Flex",
            description=f"**Prompt:** {prompt[:300]}",
            color=discord.Color.gold(),
        )
        embed.set_image(url=url)
        embed.set_footer(text=f"Model: Flux-2-Flex • Aspect: {aspect} • Res: {res}")
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send(
            "❌ Premium image generation failed. Please try again.",
            ephemeral=True,
        )


# ── /status — Check tier ──
@bot.tree.command(name="status", description="Check your Omni-Bot subscription tier")
async def slash_status(interaction: discord.Interaction) -> None:
    is_prem = interaction.user.id in PREMIUM_USER_IDS
    if is_prem:
        embed = discord.Embed(
            title="🌟 Premium Tier",
            description=(
                f"**Chat Model:** `{VOID_MODEL}`\n"
                f"**Search Model:** `{SEARCH_MODEL_PREMIUM}`\n"
                f"**Image Model:** `{IMAGE_MODEL_PREMIUM}`\n\n"
                "You have access to the most powerful models available."
            ),
            color=discord.Color.gold(),
        )
    else:
        embed = discord.Embed(
            title="🔹 Standard Tier",
            description=(
                f"**Chat Model:** `{NVIDIA_MODEL.split('/')[-1]}`\n"
                f"**Search Model:** `{SEARCH_MODEL_STANDARD}`\n"
                f"**Image Model:** `{IMAGE_MODEL_STANDARD}`\n\n"
                "Use `/premium` to access GPT-5.2 for individual questions.\n"
                "Contact the server owner for permanent Premium access!"
            ),
            color=discord.Color.blue(),
        )
    await interaction.response.send_message(embed=embed, ephemeral=True)


# ── /clear — Clear memory ──
@bot.tree.command(name="clear", description="Clear your conversation memory")
async def slash_clear(interaction: discord.Interaction) -> None:
    clear_user_history(interaction.user.id)
    await interaction.response.send_message(
        "🧹 Your conversation memory has been cleared.", ephemeral=True)


# ── /donate — Support link ──
@bot.tree.command(name="donate", description="Support Omni-Labs creators")
async def slash_donate(interaction: discord.Interaction) -> None:
    embed = discord.Embed(
        title="💖 Support Omni-Labs",
        description=(
            "Love what we're building? Help us keep the servers running and "
            "develop new features!\n\n"
            "**→ [Donate on Poe](https://poe.com/Donoz)**\n\n"
            "Every bit helps. Thank you! 🙏"
        ),
        color=discord.Color.pink(),
    )
    await interaction.response.send_message(embed=embed)


# ── /help — Full help menu ──
@bot.tree.command(name="help", description="Learn how to use Omni-Bot")
async def slash_help(interaction: discord.Interaction) -> None:
    embed = discord.Embed(
        title="📖 Omni-Bot Help — by Omni-Labs",
        description="Your AI-powered assistant, right here in Discord.",
        color=discord.Color.blurple(),
    )
    embed.add_field(
        name="💬 Chat",
        value=(
            "**@mention me** or **DM me** to chat naturally.\n"
            "`/ask <question>` — Quick question (auto-upgrades if you're Premium).\n"
            "`/premium <question>` — Force GPT-5.2 for any question.\n"
            "I remember your conversation — use `/clear` to reset."
        ),
        inline=False,
    )
    embed.add_field(
        name="🌐 Web Search",
        value=(
            "`/search <query>` — Search the internet + AI summary.\n"
            "`/ask <question> search:True` — Any question with search.\n"
            "I also auto-detect search intent from keywords like "
            "*\"latest\", \"news\", \"who is\"*, etc."
        ),
        inline=False,
    )
    embed.add_field(
        name="🎨 Image Generation",
        value=(
            "`/image <prompt>` — Standard quality (Flux-Klein).\n"
            "`/image_flex <prompt>` — Premium quality with aspect ratio & resolution controls."
        ),
        inline=False,
    )
    embed.add_field(
        name="⚙️ Utilities",
        value=(
            "`/clear` — Wipe your conversation memory.\n"
            "`/status` — Check your tier & models.\n"
            "`/donate` — Support Omni-Labs.\n"
            "`!clear` — Prefix command version of /clear."
        ),
        inline=False,
    )
    embed.set_footer(
        text="Standard: Kimi-K2 + Gemini-Flash + Flux-Klein  •  "
             "Premium: GPT-5.2 + Gemini-Pro + Flux-Flex")
    await interaction.response.send_message(embed=embed, ephemeral=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GLOBAL ERROR HANDLER (prefix commands)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError) -> None:
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"⚠️ Missing argument: `{error.param.name}`")
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("🔒 You don't have permission to do that.")
    else:
        logger.error("Prefix command error:\n%s",
                     "".join(traceback.format_exception(type(error), error, error.__traceback__)))
        await ctx.send("❌ Something went wrong. The error has been logged.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        logger.critical("DISCORD_BOT_TOKEN is not set! Add it to your Koyeb environment variables.")
        raise SystemExit(1)

    missing = []
    if not NVIDIA_API_KEY:
        missing.append("NVIDIA_API_KEY (Standard chat won't work)")
    if not VOID_API_KEYS:
        missing.append("VOID_API_KEYS (Premium chat won't work)")
    if not OLD_LLM_KEY:
        missing.append("OLD_LLM_KEY (Web search won't work)")
    if not AIR_API_KEYS:
        missing.append("AIR_API_KEYS (Image generation won't work)")
    if missing:
        logger.warning("Missing optional env vars:\n  • %s", "\n  • ".join(missing))

    logger.info("Starting Omni-Bot…")
    bot.run(DISCORD_TOKEN, log_handler=None)

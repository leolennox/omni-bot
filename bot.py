import os
import discord
from discord.ext import commands
from discord import app_commands
import openai
import requests
import json
import random
import asyncio
import base64
import aiohttp
from flask import Flask
from threading import Thread
from datetime import datetime
from collections import defaultdict

# --- 1. WEB SERVER FOR KOYEB ---
app = Flask('')
@app.route('/')
def home(): return "Omni-Bot is active and running on Omni-Labs protocols."

def run_web_server():
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run_web_server)
    t.daemon = True
    t.start()

# --- 2. CONFIGURATION & API KEYS ---
# NVIDIA (Kimi)
NVIDIA_API_KEY = "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"
# VoidAI (GPT-5.2)
VOID_KEYS = [
    "sk-voidai-qsefD43wSm1WyCo0bx9xrChZ7tw1aPvqCzj1MAXvLxG3s0p1wL5YXscDFTdG-DmuwHFIYI48IiTrQqDRTeOu5BD79hkYcMwwZd-Uwd_WZmLiJ9Ns65612j3r6TakacJn4RPPNw",
    "sk-voidai-Bwm86hgSPt0Nfpvt3jyI26RwqAMCj0CnogJuHsf__ZCcmDO_fvyOxct3rH-ifREWUqahH_QZIoeveCrJ_SVIh9TkfpdERN3sDqUjv072fxLv9XSsAi3gKTYKKjUykZh6l4Latg",
    "sk-voidai-0z1gg_TmIG59GYCvgITLKZ5_SxPaO3Ngqud-CoZVmvL2Bm28ij_jJMlD-gVBpR1GvbjKuw9DLHDDFmJOWbF77wopvscfgYoD1A0TBaCzohtHaI_7zdGhtqOgxcUDQ5GWkHm61g",
    "sk-voidai-gzWDvS76-vco4b5h_G4qIUQGnsZG9S6Gx15JVY06nWvg6IZpF_tnc1gdZsLqKyjUrBtNu05KNHkTSrYhqb273g_8H9WLFJxxlAryuQ9V8NbxkWIyhEIrIp5dyLJnKmH7OXiMhA"
]
# Airforce (Images)
AIR_KEYS = [
    "sk-air-ab2d87GvUK8VLfaYg0nZUHWSFT4DeJizPeHkdYPg4pkJAQwfEtXrVG9AOGTmsZCK",
    "sk-air-BZc59pYB02xSMt24PBG6eohvoCIAtnDkIv8Eq3gZiGc8rU9bJ5ZwbnA2uTQu7NXc",
    "sk-air-Qe8aZxyarOhkUVRlUjvITRxTq0KJ0B9EKQ4TQVkrorGF7KrKcZntyx3TTRX0wG8X",
    "sk-air-b3e2Sp0qPQrWUh5L1MjvTNldisz08gmxt6goNLQoEAGBjX6Mg72VmsCV4GIw3NAa",
    "sk-air-qOkOwflBk6r8BpRalD9XUgPmQkIqdM1yl0RDzgrC1O5InxSM4SdN0R382zjjbIVO",
    "sk-air-plCK4LflHEdgHuIPiddOSVCQRd3vde2ito4Sk1OmkF6haWUB8Sen2b6KaLIffjpT",
    "sk-air-Xat0tIqC3ohBjqLPsf2oa33NTsPj18aGjXh50rRDiCc4AxPyF3rnoKRoxxOlQeuF",
    "sk-air-O7W7od6sW5lVNdAstK9nkbeHmjmBtImfGYOlKuqpO70Iky5VKll8rZThFszPvUFW",
    "sk-air-U26UCfECkXv9edpeAZW65hOvE6W9BXDf9ch7nHI0EzdZdLaj3JaX4DT0oWFuLhpJ"
]
# Old-LLM (Search)
OLD_LLM_KEY = "sk-blue9182"

# Discord Setup
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Memory
history = defaultdict(list)

# --- 3. SYSTEM PROMPTS ---
OMNI_SYSTEM_PROMPT = """You are Omni-Bot, the advanced AI representative for Omni-Labs.
Omni-Labs is a premier AI creation studio led by @OmniLabs.
Your goal is to assist users in the Omni-Labs Discord server with high-level intelligence.
You are professional, innovative, and helpful. 
When users ask about donations, direct them to https://poe.com/Donoz.
If you need to search the web or generate an image, inform the user you are activating those modules."""

# --- 4. CORE AI ENGINES ---

def get_nvidia_client():
    return openai.OpenAI(api_key=NVIDIA_API_KEY, base_url="https://integrate.api.nvidia.com/v1")

def get_void_client():
    return openai.OpenAI(api_key=random.choice(VOID_KEYS), base_url="https://api.voidai.app/v1")

async def web_search(query, premium=False):
    model = "gemini-3-pro-preview-maxthinking-search" if premium else "gemini-3-flash-preview-maxthinking-search"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "stream": False
    }
    try:
        r = requests.post("https://old-llm-api.onrender.com/v1/chat/completions", 
                         headers={"Authorization": f"Bearer {OLD_LLM_KEY}"}, json=payload)
        return r.json()['choices'][0]['message']['content']
    except:
        return "Search failed. Proceeding with internal knowledge."

async def generate_image_logic(prompt, premium=False, width=1024, height=1024, aspect="1:1", res="1k"):
    url = "https://api.airforce/v1/images/generations"
    key = random.choice(AIR_KEYS)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    
    if premium:
        payload = {
            "model": "flux-2-flex",
            "prompt": prompt,
            "aspectRatio": aspect,
            "resolution": res,
            "sse": True
        }
    else:
        payload = {
            "model": "flux-2-klein-9b",
            "prompt": prompt,
            "width": width,
            "height": height,
            "sse": True
        }

    try:
        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if "data: " in line_str and "[DONE]" not in line_str:
                        data = json.loads(line_str[6:])
                        if "url" in data: return data["url"]
                        if "data" in data: return data["data"][0]["url"]
    except Exception as e:
        return f"Error: {e}"

# --- 5. SMART ROUTING LOGIC ---

async def omni_chat_logic(user_id, message_text, is_premium=False):
    # 1. Check for Search Intent
    search_keywords = ["search", "lookup", "latest", "news", "who is", "what happened today"]
    if any(k in message_text.lower() for k in search_keywords):
        search_results = await web_search(message_text, premium=is_premium)
        context_prompt = f"The user asked: {message_text}. Web search results: {search_results}. Summarize this for the user as Omni-Bot."
    else:
        context_prompt = message_text

    # 2. Select Model
    if is_premium:
        client = get_void_client()
        model_name = "gpt-5.2"
    else:
        client = get_nvidia_client()
        model_name = "moonshotai/kimi-k2-instruct-0905"

    # 3. Build History
    messages = [{"role": "system", "content": OMNI_SYSTEM_PROMPT}]
    for h in history[user_id][-10:]:
        messages.append(h)
    messages.append({"role": "user", "content": context_prompt})

    # 4. Generate Response
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            stream=False
        )
        response = completion.choices[0].message.content
        history[user_id].append({"role": "user", "content": message_text})
        history[user_id].append({"role": "assistant", "content": response})
        return response
    except Exception as e:
        return f"Omni-Bot Core Error: {e}"

# --- 6. DISCORD COMMANDS ---

@bot.event
async def on_ready():
    await bot.tree.sync()
    keep_alive()
    print(f"Omni-Bot Synced and Ready as {bot.user}")

@bot.event
async def on_message(message):
    if message.author == bot.user: return
    
    # Respond to Pings
    if bot.user.mentioned_in(message) and message.mention_everyone is False:
        clean_text = message.content.replace(f'<@{bot.user.id}>', '').strip()
        async with message.channel.typing():
            response = await omni_chat_logic(message.author.id, clean_text, is_premium=False)
            await message.reply(response)
    
    await bot.process_commands(message)

# --- SLASH COMMANDS ---

@bot.tree.command(name="ask", description="Ask Omni-Bot (Standard Kimi-K2)")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    response = await omni_chat_logic(interaction.user.id, question, is_premium=False)
    await interaction.followup.send(response)

@bot.tree.command(name="premium", description="Ask Omni-Bot Premium (GPT-5.2)")
async def premium(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    response = await omni_chat_logic(interaction.user.id, question, is_premium=True)
    await interaction.followup.send(f"✨ **Premium Response:**\n{response}")

@bot.tree.command(name="image", description="Generate Image (Standard Flux-Klein)")
async def image(interaction: discord.Interaction, prompt: str, width: int = 1024, height: int = 1024):
    await interaction.response.defer()
    await interaction.followup.send(f"🎨 Generating `{prompt}` for you...")
    url = await generate_image_logic(prompt, premium=False, width=width, height=height)
    if url.startswith("http"):
        embed = discord.Embed(title="Omni-Labs Image Gen", description=f"Prompt: {prompt}")
        embed.set_image(url=url)
        await interaction.channel.send(embed=embed)
    else:
        await interaction.followup.send(f"❌ Failed: {url}")

@bot.tree.command(name="image_flex", description="Premium Image Gen (Flux-Flex)")
@app_commands.choices(aspect=[
    app_commands.Choice(name="1:1", value="1:1"),
    app_commands.Choice(name="16:9", value="16:9"),
    app_commands.Choice(name="9:16", value="9:16"),
    app_commands.Choice(name="4:3", value="4:3"),
], res=[
    app_commands.Choice(name="1k", value="1k"),
    app_commands.Choice(name="2k", value="2k"),
])
async def image_flex(interaction: discord.Interaction, prompt: str, aspect: str = "1:1", res: str = "1k"):
    await interaction.response.defer()
    await interaction.followup.send(f"🚀 **Premium Flex Gen** starting for `{prompt}`...")
    url = await generate_image_logic(prompt, premium=True, aspect=aspect, res=res)
    if url.startswith("http"):
        embed = discord.Embed(title="Omni-Labs Premium Flex", color=discord.Color.gold())
        embed.set_image(url=url)
        embed.set_footer(text=f"Aspect: {aspect} | Res: {res}")
        await interaction.channel.send(embed=embed)
    else:
        await interaction.followup.send(f"❌ Failed: {url}")

@bot.tree.command(name="clear", description="Clear your conversation memory")
async def clear(interaction: discord.Interaction):
    history[interaction.user.id] = []
    await interaction.response.send_message("🧹 Memory wiped clean.", ephemeral=True)

@bot.tree.command(name="donate", description="Support Omni-Labs creators")
async def donate(interaction: discord.Interaction):
    await interaction.response.send_message("Support the creators here: https://poe.com/Donoz 💖")

if __name__ == "__main__":
    bot.run(TOKEN)

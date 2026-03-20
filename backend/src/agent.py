# ======================================================
# 🎤 VOICE IMPROVE TUTOR BY NAMAN
# 🚀 Communication AI using LiveKit
# ======================================================

import logging
import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict

print("\n" + "🎤" * 50)
print("🚀 VOICE IMPROVE TUTOR - BY NAMAN")
print("💬 Practice English | Improve Communication | Speak Confidently")
print("🎤" * 50 + "\n")

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 🧠 SESSION DATA
# ======================================================

@dataclass
class SessionData:
    conversation_history: list[str] = field(default_factory=list)
    session_start: datetime = field(default_factory=datetime.now)

# ======================================================
# 💾 SIMPLE HISTORY (OPTIONAL)
# ======================================================

LOG_FILE = "conversation_log.json"

def get_log_path():
    base_dir = os.path.dirname(__file__)
    backend_dir = os.path.abspath(os.path.join(base_dir, ".."))
    return os.path.join(backend_dir, LOG_FILE)

def load_history():
    path = get_log_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return []

def save_message(message):
    path = get_log_path()
    history = load_history()
    history.append({
        "time": datetime.now().isoformat(),
        "message": message
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)

# ======================================================
# 🧠 COMMUNICATION TUTOR AGENT
# ======================================================

class CommunicationTutorAgent(Agent):
    def __init__(self, history_context: str):
        super().__init__(
            instructions=f"""
            You are an intelligent, friendly, and engaging **English Communication Tutor**.

            🎤 Your name is: **Voice Improve Tutor by Naman**

            🧠 PREVIOUS CONTEXT:
            {history_context}

            🎯 YOUR ROLE:
            - Talk with the user on ANY topic (life, tech, movies, study, etc.)
            - Help improve their English speaking skills
            - Keep conversation natural and engaging

            💬 HOW TO RESPOND:
            1. Always reply like a human (not robotic)
            2. Ask follow-up questions
            3. Encourage long answers

            ✨ CORRECTION STYLE:
            If user makes a mistake:
            - First respond normally
            - Then gently correct

            Example:
            User: "I go market yesterday"
            You:
            "Nice! A better way to say it is: 'I went to the market yesterday.'"

            🎯 ALSO:
            - Suggest better vocabulary sometimes
            - Improve sentence structure
            - Help build confidence

            🔥 KEEP IT FUN:
            - Be friendly 😄
            - Be supportive 🤝
            - Be conversational 🎤

            🚫 DO NOT:
            - Give long lectures
            - Be strict or boring

            👉 Start by greeting the user and asking an interesting question.
            """,
            tools=[],
        )

# ======================================================
# 🎬 ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    print("\n🎬 Starting Voice Improve Session...\n")

    # Load previous history (optional)
    history = load_history()
    history_summary = "No previous conversations."

    if history:
        history_summary = f"User has {len(history)} previous interactions."

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-natalie",
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=SessionData(),
    )

    await session.start(
        agent=CommunicationTutorAgent(history_context=history_summary),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

# ======================================================
# 🚀 RUN APP
# ======================================================

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm
        )
    )

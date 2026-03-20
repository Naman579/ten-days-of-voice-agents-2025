import logging
import random
from datetime import datetime
from dataclasses import dataclass, field

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("english_tutor")
load_dotenv(".env.local")

# ======================================================
# 🎯 USER SESSION DATA
# ======================================================
@dataclass
class UserData:
    level: str = "beginner"
    topics_done: list[str] = field(default_factory=list)
    session_start: datetime = field(default_factory=datetime.now)

# ======================================================
# 🎯 RANDOM TOPICS
# ======================================================
TOPICS = [
    "daily routine",
    "your favorite movie",
    "your dream job",
    "travel experience",
    "technology and AI",
    "college life",
    "hobbies",
    "your best friend",
    "food and restaurants",
    "social media",
]

def get_random_topic(userdata: UserData):
    remaining = [t for t in TOPICS if t not in userdata.topics_done]
    if not remaining:
        userdata.topics_done = []
        remaining = TOPICS
    topic = random.choice(remaining)
    userdata.topics_done.append(topic)
    return topic

# ======================================================
# 🧠 ENGLISH TUTOR AGENT
# ======================================================
class EnglishTutorAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a friendly AI English Tutor.

🎯 Your goal:
- Help the user practice spoken English
- Talk on random interesting topics
- Keep conversation natural and engaging

🗣️ Conversation Rules:
- ALWAYS speak in English
- Ask open-ended questions
- Encourage the user to speak more
- If the user makes mistakes:
    1. Correct politely
    2. Show correct sentence
    3. Ask them to repeat

📚 Teaching Style:
- Be friendly and supportive
- Do NOT be too strict
- Keep responses short and conversational
- Ask follow-up questions

💡 Example:
User: "I go to market yesterday"
You:
👉 "Good try! You should say: 'I went to the market yesterday.'  
Can you repeat that?"

Then continue conversation.

🚀 Start by introducing yourself and give a random topic.
            """,
        )

# ======================================================
# 🚀 MAIN ENTRYPOINT
# ======================================================
async def entrypoint(ctx: JobContext):
    print("\n🎤 AI ENGLISH TUTOR STARTED\n")

    userdata = UserData()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
        ),
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        userdata=userdata,
    )

    topic = get_random_topic(userdata)

    await session.start(
        agent=EnglishTutorAgent(),
        room=ctx.room,
        room_input_options=noise_cancellation.BVC(),
    )

    await ctx.connect()

    # 🎯 First message (important)
    await session.generate_reply(
        instructions=f"""
Start conversation.

Introduce yourself as an English tutor.

Then say:
"Today's topic is: {topic}"

Ask a simple question related to this topic.
"""
    )

# ======================================================
# ▶️ RUN APP
# ======================================================
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

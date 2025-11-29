# gm_voice_agent.py
"""
D&D-Style Voice Game Master (GM) Agent
- Fantasy universe, short story, basic English
- Uses chat history only for continuity
- Voice Interaction: STT (player speaks) + TTS (GM narrates)
"""

import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    RunContext,
)
from livekit.plugins import murf, silero, deepgram, google, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Logging setup
# -------------------------
logger = logging.getLogger("voice_gm_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# Memory stored in conversation
# -------------------------
@dataclass
class PlayerMemory:
    name: str = "Player"
    health: int = 100
    inventory: List[str] = field(default_factory=list)
    allies: List[str] = field(default_factory=list)
    visited_places: List[str] = field(default_factory=list)
    past_actions: List[str] = field(default_factory=list)


# -------------------------
# Game Master class
# -------------------------
class VoiceGM(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are the Game Master (GM).
            Universe: Fantasy land called "Eldervale".
            Tone: spooky but simple.
            You describe scenes and ask the player what they do.
            Always end with: "What do you do?"
            """,
        )

        self.memory = PlayerMemory(name="Arin")
        self.turn = 0
        self.max_turns = 10  # short session

    async def run(self, ctx: RunContext) -> str:
        """Generate story narration based on chat history context."""
        player_input = ctx.messages[-1].content if ctx.messages else ""
        self.turn += 1

        if player_input:
            self.memory.past_actions.append(player_input[:80])

        if self.turn == 1:
            scene = self.scene_1()
        elif self.turn == 2:
            scene = self.scene_2(player_input)
        elif self.turn == 3:
            scene = self.scene_3(player_input)
        elif self.turn == 4:
            scene = self.scene_4(player_input)
        elif self.turn >= self.max_turns:
            scene = self.final_scene()
        else:
            scene = self.generic_scene(player_input)

        logger.info(f"GM Turn {self.turn}: {scene}")
        return scene

    # ---- Simple scenes ----

    def scene_1(self):
        self.memory.visited_places.append("Tavern")
        return "You are in a quiet tavern near a dark forest. The fire is low. A small gnome says someone stole a map. What do you do?"

    def scene_2(self, action: str):
        self.memory.allies.append("Fenrix")
        return "A hooded ranger named Fenrix stands up. He offers to help you find the map. What do you do?"

    def scene_3(self, action: str):
        self.memory.visited_places.append("Forest")
        self.memory.inventory.append("Map Scrap")
        return "You walk into the forest. You find a small piece of paper with a star mark on it. It might be part of the map. What do you do?"

    def scene_4(self, action: str):
        self.memory.health -= 10
        return "You hear wolves. One jumps out and scares you. You lose a little health. Fenrix draws his bow to protect you. What do you do?"

    def generic_scene(self, action: str):
        return f"The world reacts to your action: {action}. You see the path ahead glowing with stars. What do you do?"

    def final_scene(self):
        return "You reach a stone door with a glowing star mark. Fenrix nods at you. The door opens slowly. A new quest begins. What do you do?"


# -------------------------
# LiveKit Voice interaction entrypoint
# -------------------------
async def entrypoint(ctx: JobContext):

    logger.info("\n🎲 GM Voice Session Starting...\n")

    gm_agent = VoiceGM()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-marcus", style="Neutral"),
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        userdata=gm_agent.memory,
    )

    await session.start(
        agent=gm_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY is not set. Please add it to your .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# 📜 Poet Agent
poet_agent = Agent(
    name="Poet Agent",
    instructions="""
    You are a creative poet. Write a short, original poem with 2 stanzas.
    It can be lyric, narrative, or dramatic in style.
    Return only the poem text.
    """
)

# ❤️ Lyric Analyst Agent
lyric_analyst = Agent(
    name="Lyric Analyst",
    instructions="""
    You analyze if a poem expresses personal emotions and feelings.
    State clearly if it qualifies as lyric poetry and why.
    """
)

# 📖 Narrative Analyst Agent
narrative_analyst = Agent(
    name="Narrative Analyst",
    instructions="""
    You analyze if a poem tells a story with characters and events.
    State clearly if it qualifies as narrative poetry and why.
    """
)

# 🎭 Dramatic Analyst Agent
dramatic_analyst = Agent(
    name="Dramatic Analyst",
    instructions="""
    You analyze if a poem is suitable for performance or dramatic monologue.
    State clearly if it qualifies as dramatic poetry and why.
    """
)

if __name__ == "__main__":
    # Step 1 — Poet Agent writes a poem
    poem_result = Runner.run_sync(poet_agent, input="Write the poem now.", run_config=config)
    poem = poem_result.final_output

    # Step 2 — Analysts evaluate the poem
    lyric_result = Runner.run_sync(lyric_analyst, input=poem, run_config=config)
    narrative_result = Runner.run_sync(narrative_analyst, input=poem, run_config=config)
    dramatic_result = Runner.run_sync(dramatic_analyst, input=poem, run_config=config)

    # Step 3 — Print the final report
    print("\n📊 Final Report:\n")
    print(f"📜 Poem:\n{poem}\n")
    print(f"❤️ Lyric Analysis:\n{lyric_result.final_output}\n")
    print(f"📖 Narrative Analysis:\n{narrative_result.final_output}\n")
    print(f"🎭 Dramatic Analysis:\n{dramatic_result.final_output}\n")

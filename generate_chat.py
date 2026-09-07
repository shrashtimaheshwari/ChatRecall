import os
import json
import time
import random
import argparse
import uuid
from datetime import datetime, timedelta, timezone
from groq import Groq
from dotenv import load_dotenv, find_dotenv

import config

load_dotenv(find_dotenv())

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-20b"

SCENARIOS = [
    "planning a trip to Goa and debating budget and dates",
    "freaking out about upcoming mid-semester exams and sharing notes",
    "hostel and roommate logistics (who is paying for pizza, noise issues)",
    "sharing random memes, Instagram reels, and heavily forwarded WhatsApp junk",
    "a stressful group project deadline for a coding assignment"
]

DECISION_THREADS = [
    "CRITICAL REQUIREMENT: This batch MUST include a long thread that explicitly concludes in a clear, concrete decision about the exact dates for the Goa trip.",
    "CRITICAL REQUIREMENT: This batch MUST include a long thread that explicitly concludes in a clear, concrete decision about which programming language and framework to use for the group project.",
    "CRITICAL REQUIREMENT: This batch MUST include a long thread that explicitly concludes in a clear, concrete decision about what specific food to order for dinner tonight."
]

SYSTEM_PROMPT_TEMPLATE = """You are an expert data generator creating a highly realistic, messy WhatsApp group chat export.
The chat has 8 participants: {participants}.
The primary persona (SEED) is "{seed}" who is an active participant in this chat.

Requirements for realism:
1. The language MUST be heavily Hinglish (code-mixed Hindi and English written in Latin script).
2. Include frequent typos, abbreviations, and one-word replies ('lol', 'haan', 'ok', 'acha', 'k').
3. Include some messages that look like forwarded junk (e.g., "Forwarded: Good morning...").
4. Make it look like a real, chaotic group chat among Indian college students/friends.

This batch focuses on the following scenario: {scenario}
{decision_instruction}

Output exactly {num_messages} messages.
You MUST output a valid JSON object with a single key "messages" containing an array of message objects.
Each message object must have EXACTLY these two keys:
- "sender": string (must be exactly one of the 8 participants)
- "text": string (the message content)
"""

def generate_batch(scenario: str, decision_instruction: str = "", num_messages: int = 150, retries: int = 5) -> list:
    participants_str = ", ".join(config.PARTICIPANTS)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        participants=participants_str,
        seed=config.SEED,
        scenario=scenario,
        decision_instruction=decision_instruction,
        num_messages=num_messages
    )
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate {num_messages} messages now. Return a JSON object with a 'messages' array."}
                ],
                temperature=0.8,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            return data.get("messages", [])
            
        except Exception as e:
            wait_time = (2 ** attempt) * 2 + random.uniform(0, 1)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            
    print("Failed to generate batch after maximum retries.")
    return []

def assign_metadata(raw_messages: list, start_date: datetime) -> list:
    processed = []
    current_time = start_date
    for msg in raw_messages:
        # Group chats usually have bursts of messages and long silences
        # 80% chance of a quick reply (1-5 mins), 20% chance of a long gap (30-300 mins)
        if random.random() < 0.8:
            gap = timedelta(minutes=random.randint(1, 5))
        else:
            gap = timedelta(minutes=random.randint(30, 300))
            
        current_time += gap
        
        # Enforce valid sender fallback just in case the LLM hallucinates a name
        sender = msg.get("sender")
        if sender not in config.PARTICIPANTS:
            sender = random.choice(config.PARTICIPANTS)
            
        processed.append({
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "sender": sender,
            "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "text": str(msg.get("text", ""))
        })
    return processed

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic group chat data.")
    parser.add_argument("--test", action="store_true", help="Generate a small test batch of 20 messages and print to console.")
    args = parser.parse_args()

    if args.test:
        print("Generating a small test batch...")
        scenario = random.choice(SCENARIOS)
        decision = random.choice(DECISION_THREADS)
        raw = generate_batch(scenario, decision, num_messages=20)
        start_date = datetime.now(timezone.utc) - timedelta(days=180)
        processed = assign_metadata(raw, start_date)
        print(json.dumps(processed, indent=2))
        return

    # Full run
    print("Starting full generation of ~4500 messages...")
    all_raw_messages = []
    
    # We do 30 batches of 150 messages each = 4500 messages
    TOTAL_BATCHES = 30
    
    # Assign the 3 decision threads to random specific batches
    decision_batch_indices = random.sample(range(TOTAL_BATCHES), len(DECISION_THREADS))
    decisions_to_assign = DECISION_THREADS.copy()
    
    for i in range(TOTAL_BATCHES):
        print(f"Generating batch {i + 1}/{TOTAL_BATCHES}...")
        scenario = random.choice(SCENARIOS)
        
        decision = ""
        if i in decision_batch_indices:
            decision = decisions_to_assign.pop(0) if decisions_to_assign else ""
            
        raw = generate_batch(scenario, decision, num_messages=150)
        all_raw_messages.extend(raw)
        
        # Friendly backoff to respect potential API limits
        time.sleep(2.0)

    print(f"Generated {len(all_raw_messages)} raw messages.")
    
    start_date = datetime.now(timezone.utc) - timedelta(days=180)
    processed = assign_metadata(all_raw_messages, start_date)
    
    os.makedirs("data", exist_ok=True)
    out_file = "data/chat_export.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully saved {len(processed)} messages to {out_file}")

if __name__ == "__main__":
    main()

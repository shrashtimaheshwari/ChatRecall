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
    {"tag": "trip", "desc": "planning a trip to Goa and debating budget and dates"},
    {"tag": "exams", "desc": "freaking out about upcoming mid-semester exams and sharing notes"},
    {"tag": "hostel", "desc": "hostel and roommate logistics (who is paying for pizza, noise issues)"},
    {"tag": "memes", "desc": "sharing random memes, Instagram reels, and heavily forwarded WhatsApp junk"},
    {"tag": "project", "desc": "a stressful group project deadline for a coding assignment"}
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

This batch focuses on the following scenario: {scenario_desc}{negative_constraint}
{decision_instruction}

Output exactly {num_messages} messages.
You MUST output a valid JSON object with a single key "messages" containing an array of message objects.
Each message object must have EXACTLY these three keys:
- "sender": string (must be exactly one of the 8 participants)
- "text": string (the message content)
- "scenario": string (must be EXACTLY "{scenario_tag}")
"""

def generate_batch(scenario_tag: str, scenario_desc: str, decision_instruction: str = "", num_messages: int = 50, retries: int = 4) -> list:
    participants_str = ", ".join(config.PARTICIPANTS)
    
    negative_constraint = ""
    if scenario_tag != "trip":
        negative_constraint = "\nCRITICAL: Do not mention any trip, vacation, Goa, Manali, or travel plans in this batch under any circumstance."
        
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        participants=participants_str,
        seed=config.SEED,
        scenario_tag=scenario_tag,
        scenario_desc=scenario_desc,
        negative_constraint=negative_constraint,
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
                max_tokens=4096,
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
        if not isinstance(msg, dict):
            continue
            
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
            "text": str(msg.get("text", "")),
            "scenario": str(msg.get("scenario", "UNKNOWN"))
        })
    return processed

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic group chat data.")
    parser.add_argument("--test", action="store_true", help="Generate a small test batch of 20 messages and print to console.")
    args = parser.parse_args()

    if args.test:
        print("Generating a small test batch...")
        scenario_obj = SCENARIOS[0]
        decision = DECISION_THREADS[0]
        raw = generate_batch(scenario_obj["tag"], scenario_obj["desc"], decision, num_messages=20)
        start_date = datetime.now(timezone.utc) - timedelta(days=180)
        processed = assign_metadata(raw, start_date)
        print(json.dumps(processed, indent=2))
        return

    # Full run
    print("Starting full generation of ~4500 messages...")
    
    # We do 90 batches of 50 messages each = 4500 messages
    TOTAL_BATCHES = 90
    TEMP_FILE = "temp_batches.jsonl"
    
    start_batch = 0
    all_raw_messages = []
    
    if os.path.exists(TEMP_FILE):
        with open(TEMP_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            start_batch = len(lines)
            for line in lines:
                all_raw_messages.extend(json.loads(line))
        print(f"Found {start_batch} completed batches in checkpoint. Resuming from batch {start_batch + 1}...")

    remaining_batches = TOTAL_BATCHES - start_batch
    
    # Assign the 3 decision threads to specific batch indices matching their scenarios
    # DECISION_THREADS[0] = Goa trip. Modulo 5 index 0 -> batch 15
    # DECISION_THREADS[1] = Framework. Modulo 5 index 4 -> batch 44
    # DECISION_THREADS[2] = Dinner. Modulo 5 index 2 -> batch 72
    decision_map = {
        15: DECISION_THREADS[0],
        44: DECISION_THREADS[1],
        72: DECISION_THREADS[2]
    }
    
    with open(TEMP_FILE, "a", encoding="utf-8") as temp_out:
        for i in range(start_batch, TOTAL_BATCHES):
            print(f"Generating batch {i + 1}/{TOTAL_BATCHES}...")
            
            # Deterministic scenario selection
            scenario_obj = SCENARIOS[i % len(SCENARIOS)]
            
            decision = decision_map.get(i, "")
                
            raw = generate_batch(scenario_obj["tag"], scenario_obj["desc"], decision, num_messages=50)
            if not raw:
                print("Critical API failure. Exiting early to save progress. Run script again later.")
                break
                
            # Write to checkpoint file immediately
            temp_out.write(json.dumps(raw) + "\n")
            temp_out.flush()
            
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

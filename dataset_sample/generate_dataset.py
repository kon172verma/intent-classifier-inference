#!/usr/bin/env python3
"""
Synthetic dataset generator for the tool_router project.

Usage:
    python generate_dataset.py --n 100 --out sample.json --seed 42
    python generate_dataset.py --n 10000 --out splits  # writes train/validation/test.jsonl
"""

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# User-request pool: several realistic phrasings per tool
# ---------------------------------------------------------------------------
REQUESTS_PER_TOOL: dict[str, list[str]] = {
    "corp-kariya-weather-app": [
        "What's the weather like in Tokyo right now?",
        "Will it rain this afternoon?",
        "Are there any severe weather warnings near my current location?",
        "What should I expect weather-wise on my drive to Osaka?",
        "Is it going to snow tonight?",
        "Show me a 3-day weather forecast for this area.",
        "What's the temperature outside right now?",
        "Any thunderstorm alerts on my route today?",
    ],
    "safety_proposal": [
        "The road is really slippery – what should I do?",
        "I've been driving for 5 hours straight. Give me some safety advice.",
        "What's the recommended following distance at 80 km/h?",
        "Is it safe to overtake on this stretch of road?",
        "Give me driving safety tips for heavy rain conditions.",
        "Am I driving too aggressively right now?",
        "How can I improve my driving safety score?",
        "What safety precautions should I take in dense fog?",
    ],
    "face_emotion": [
        "Am I looking tired right now?",
        "Can you tell if I'm stressed?",
        "Check if the driver is getting drowsy.",
        "Analyse my facial expression.",
        "Am I showing signs of fatigue on the camera?",
        "Detect my current emotional state from my face.",
        "Is the driver alert enough to keep driving?",
        "Run a drowsiness check on me.",
    ],
    "corp-scheduler-app": [
        "Schedule a meeting with Tanaka-san at 3 pm tomorrow.",
        "What's on my calendar for this afternoon?",
        "Cancel my 2 pm appointment.",
        "Add a reminder for the project deadline on Friday.",
        "Move my Monday standup to Tuesday at 10 am.",
        "Am I free this Thursday afternoon?",
        "Block out two hours on Wednesday morning for deep work.",
        "What meetings do I have this week?",
    ],
    "in_car_analytics": [
        "What's my average fuel consumption this week?",
        "Show me my driving stats for today.",
        "How fast was I going on average during my morning commute?",
        "Give me a summary of my last trip's performance.",
        "What was my top speed today?",
        "How much fuel did I use on the last trip?",
        "Show me my eco-driving score for the past month.",
        "What's my average speed on the expressway today?",
    ],
    "nav_route_planner": [
        "Navigate me to Nagoya station.",
        "Find me the fastest route home.",
        "Reroute to avoid the highway.",
        "How long will it take to get to the airport?",
        "Take me to the nearest hospital.",
        "Show me an alternate route around the road works.",
        "Get me to the downtown office by 9 am.",
        "Is there a faster way to reach the hotel?",
    ],
    "traffic_alert_service": [
        "Is there a traffic jam on the expressway right now?",
        "Any accidents reported on Route 1?",
        "Check traffic conditions ahead on my route.",
        "Are there any road closures nearby?",
        "How bad is the congestion in the city centre?",
        "Any incidents on the ring road this morning?",
        "What's the traffic like on the bridge?",
        "Are there any delays I should know about before I leave?",
    ],
    "media_player_ctrl": [
        "Play some jazz music.",
        "Skip to the next track.",
        "Turn down the volume a bit.",
        "Pause the music.",
        "Shuffle my driving playlist.",
        "Put on some relaxing music for the drive.",
        "Play the last album I was listening to.",
        "Increase the bass on the audio system.",
    ],
    "climate_control": [
        "It's getting hot in here, cool it down.",
        "Set the cabin temperature to 22 degrees.",
        "Turn on the heated seats.",
        "Increase the fan speed.",
        "Switch to fresh air mode.",
        "Turn off the air conditioning.",
        "Can you defrost the rear windscreen?",
        "Make it a bit warmer for the passenger side.",
    ],
    "vehicle_diagnostics": [
        "My check engine light just came on – what's wrong?",
        "What's my current tyre pressure?",
        "When is my next service due?",
        "Read any fault codes from the engine.",
        "Is everything okay with the brakes?",
        "Run a quick health check on the vehicle.",
        "The oil warning light is on – is it critical?",
        "What does error code P0302 mean on my dashboard?",
    ],
    "corp-contact-sync": [
        "Find John's phone number.",
        "Look up Yamamoto-san's address.",
        "Do I have a contact called Sarah in my list?",
        "Show me all contacts at the head office.",
        "What's the number for Dr. Kim?",
        "Search my contacts for anyone at the Nagoya branch.",
        "Get me the mobile number for the client I met yesterday.",
        "Is there a contact named Liu Wei in my address book?",
    ],
    "call_handler": [
        "Call my wife.",
        "Phone the office.",
        "Hang up the call.",
        "Answer the incoming call.",
        "Redial the last number.",
        "Call Kenji on speakerphone.",
        "Make a hands-free call to the garage.",
        "Decline the incoming call and send a busy reply.",
    ],
    "sms_messenger": [
        "Send a text to mom saying I'll be late.",
        "Read my new messages aloud.",
        "Reply to Kenji's message with 'on my way'.",
        "Text the team that I'm heading to the meeting now.",
        "Send a message to Sarah: 'running 10 minutes late'.",
        "Did I get any new texts in the last hour?",
        "Forward the last message I received to Hiroshi.",
        "Send a quick text to the office: 'stuck in traffic'.",
    ],
    "fuel_station_finder": [
        "Find the nearest petrol station.",
        "Where can I charge my EV nearby?",
        "What's the cheapest diesel within 5 km?",
        "Is there a Shell station on my route?",
        "I'm running low on fuel – where's the nearest station?",
        "Show me EV fast chargers near the motorway junction.",
        "Find a 24-hour petrol station close by.",
        "Which nearby station has the lowest fuel price today?",
    ],
    "parking_locator": [
        "Find parking near the station.",
        "Any free parking spots downtown?",
        "Book a parking space near the conference centre.",
        "How much does parking cost at the shopping mall?",
        "Is there covered parking near my destination?",
        "Find me a parking spot within 200 metres of the venue.",
        "Reserve underground parking at the hotel.",
        "Show me 24-hour parking garages nearby.",
    ],
    "poi_search": [
        "Find a good ramen restaurant nearby.",
        "Where's the nearest hospital?",
        "Any convenience stores on my route?",
        "Show me ATMs within walking distance.",
        "Find me a coffee shop close by.",
        "Where's the nearest pharmacy?",
        "Look up sushi restaurants within 2 km.",
        "Find me a car wash near the expressway exit.",
    ],
    "trip_logger": [
        "How many kilometres did I drive last month?",
        "Show me my trip history for this week.",
        "Export my driving report.",
        "What was my longest trip this year?",
        "Log this trip for my expense report.",
        "How many trips have I completed this month?",
        "What's my total mileage since January?",
        "Save this journey to my trip log.",
    ],
    "remote_vehicle_lock": [
        "Did I lock the car?",
        "Lock the doors remotely.",
        "Unlock the boot from here.",
        "What's the current lock state of the car?",
        "Lock all doors and windows.",
        "Can you check if any windows are still open?",
        "I left the car unlocked – can you secure it?",
        "Unlock the front passenger door for my colleague.",
    ],
    "battery_status_monitor": [
        "How much battery do I have left?",
        "What's my estimated range on the current charge?",
        "Is the car still charging at the station?",
        "Show me the current battery percentage.",
        "How long until the battery is fully charged?",
        "Will I make it to Kyoto on the current charge?",
        "What's the charging speed right now?",
        "Is the battery degrading faster than normal?",
    ],
    "ev_charging_scheduler": [
        "Schedule charging for tonight at off-peak rates.",
        "Set up overnight charging to finish by 7 am.",
        "When's the cheapest time to charge tonight?",
        "Schedule a full charge starting at midnight.",
        "Optimise my charging cost for tonight.",
        "Set charging to start automatically when the electricity rate drops.",
        "Pre-heat the battery before my 6 am departure.",
        "Stop charging at 80% to preserve battery life.",
    ],
    "news_briefing": [
        "Give me the latest news headlines.",
        "What's happening in the sports world today?",
        "Play my morning news briefing.",
        "Any breaking news this morning?",
        "Read me the top stories from today.",
        "Give me a quick 2-minute news update.",
        "What are the business headlines today?",
        "Catch me up on the overnight news.",
    ],
    "package_tracker": [
        "Where's my Amazon package?",
        "When will my delivery arrive today?",
        "Track my parcel from Yamato Transport.",
        "Has my order been dispatched yet?",
        "What's the current status of my delivery?",
        "Is my package out for delivery today?",
        "My parcel was supposed to arrive yesterday – where is it?",
        "Give me the tracking number and status for my recent order.",
    ],
    "toll_payment_service": [
        "Pay the toll automatically.",
        "How much will the tolls cost on this route?",
        "Check my ETC account balance.",
        "What are the toll charges for the full expressway run?",
        "Process the upcoming toll gate.",
        "How much have I spent on tolls this month?",
        "Set up automatic toll payment for the expressway.",
        "Estimate tolls for the route to Kobe.",
    ],
    "health_monitor": [
        "What's my current heart rate?",
        "Check my blood oxygen level.",
        "How fatigued am I according to my wearable?",
        "Monitor my vitals during the drive.",
        "Is my stress level high right now?",
        "Show me my health metrics for today's drive.",
        "Am I fit to drive according to my biometrics?",
        "Alert me if my heart rate goes above 100.",
    ],
    "corp-cloud-sync": [
        "Sync my seat and mirror settings to the cloud.",
        "Restore my preferred car settings.",
        "Upload my personalisation preferences to my profile.",
        "Apply my saved settings to this rental vehicle.",
        "Back up my current vehicle configuration.",
        "Load the seat and display settings from my last car.",
        "Push my profile to the cloud so any car can load it.",
        "Sync my language and unit preferences.",
    ],
    # ── Rare tools ──────────────────────────────────────────────────────────
    "emergency_sos": [
        "Help! I've had an accident – call emergency services now!",
        "Emergency! Send an SOS alert immediately!",
        "I need emergency help right now, contact the authorities!",
        "Trigger an emergency alert and share my GPS location.",
    ],
    "roadside_assistance": [
        "My tyre just blew out on the motorway, I need roadside help.",
        "The car has broken down – please send roadside assistance.",
        "I'm locked out of my car, can someone come help?",
        "The engine stopped and I'm stranded on the hard shoulder.",
    ],
    "insurance_claims": [
        "I just had a collision – how do I start an insurance claim?",
        "Help me document this accident for my insurance company.",
        "I need to file a claim for the damage to my rear bumper.",
        "Start an insurance claim and help me upload photos of the damage.",
    ],
    "home_automation_bridge": [
        "Turn on the porch lights, I'm almost home.",
        "Set the home thermostat to 20 degrees – I'll be there in 10 minutes.",
        "Unlock the front door, I'm pulling into the driveway.",
        "Turn off the kitchen lights, I left them on when I left.",
    ],
    "corp-fleet-manager": [
        "Assign vehicle C-142 to driver Nakamura for tomorrow's shift.",
        "Show me the live locations of all fleet vehicles.",
        "Generate a fleet utilisation report for this month.",
        "Which fleet vehicles are currently unassigned?",
    ],
}

# Requests that have no matching tool → answer is "none"
NONE_REQUESTS: list[str] = [
    "What's the capital of France?",
    "Translate this paragraph into German.",
    "Book a flight to London for next week.",
    "Order a pizza for delivery tonight.",
    "What's the current exchange rate for USD to JPY?",
    "Write a birthday message for my colleague.",
    "Find me a good vegetarian curry recipe.",
    "How do I file my tax return online?",
    "Convert 100 miles to kilometres.",
    "What movies are showing at the cinema tonight?",
    "Set an alarm for 6 am tomorrow morning.",
    "Summarise the meeting notes from last Wednesday.",
    "What is the population of Japan?",
    "Tell me a joke.",
    "How do I install Python on a Windows machine?",
    "Draft an email to my manager about the project delay.",
    "Who won the last FIFA World Cup?",
    "Recommend a book on machine learning for beginners.",
    "What's 15 percent of 340?",
    "How many calories are in a bowl of ramen?",
    "What time does the Tokyo Stock Exchange open?",
    "Find a hotel in Kyoto for this weekend.",
    "Explain the theory of relativity simply.",
    "What's the best programming language to learn first?",
    "Remind me to drink more water today.",
]


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def load_tools(ref_path: Path) -> tuple[list[dict], list[dict]]:
    """Return (common_tools, rare_tools) from the reference JSON."""
    data = json.loads(ref_path.read_text(encoding="utf-8"))
    tools = data["tools"]
    common = [t for t in tools if not t["rare"]]
    rare = [t for t in tools if t["rare"]]
    return common, rare


def pick_tool_subset(
    all_tools: list[dict],
    correct: dict | None,
    count: int,
) -> list[dict]:
    """
    Sample `count` tools from `all_tools`.
    If `correct` is provided it is always included.
    """
    pool = [t for t in all_tools if correct is None or t["name"] != correct["name"]]
    sample_size = min(count - (1 if correct is not None else 0), len(pool))
    chosen = random.sample(pool, sample_size)
    if correct is not None:
        chosen.append(correct)
    random.shuffle(chosen)
    return chosen


def build_example(
    all_tools: list[dict],
    correct_tool: dict | None,
    tool_count: int,
) -> dict:
    """Build one dataset example."""
    subset = pick_tool_subset(all_tools, correct_tool, tool_count)
    if correct_tool is not None:
        request = random.choice(REQUESTS_PER_TOOL[correct_tool["name"]])
        answer = correct_tool["name"]
    else:
        request = random.choice(NONE_REQUESTS)
        answer = "none"
    return {
        "user_request": request,
        "available_tools": [
            {"name": t["name"], "description": t["description"]} for t in subset
        ],
        "answer": answer,
    }


def generate(
    n: int,
    seed: int = 42,
    ref_path: Path | None = None,
) -> list[dict]:
    """Generate `n` examples following the project distribution spec."""
    random.seed(seed)

    if ref_path is None:
        ref_path = Path(__file__).parent / "tools_reference.json"
    common_tools, rare_tools = load_tools(ref_path)
    all_tools = common_tools + rare_tools

    # ── Answer distribution ──────────────────────────────────────────────
    n_none = round(n * 0.20)
    n_valid = n - n_none

    # Rare tools are correct answer for ≤ 2-3 % of total examples
    n_rare_correct = max(1, round(n * 0.02))
    n_common_correct = n_valid - n_rare_correct

    # ── Tool-count bucket distribution ───────────────────────────────────
    n_few = round(n * 0.10)           # 1–3 tools
    n_many = round(n * 0.10)          # 20–30 tools
    n_std = n - n_few - n_many        # 4–19 tools

    count_buckets: list[int] = (
        [random.randint(1, 3) for _ in range(n_few)]
        + [random.randint(20, min(30, len(all_tools))) for _ in range(n_many)]
        + [random.randint(4, 19) for _ in range(n_std)]
    )
    random.shuffle(count_buckets)

    # ── Build the answer list ────────────────────────────────────────────
    # Sample rare tools with replacement if needed
    rare_answers: list[dict | None] = [
        random.choice(rare_tools) for _ in range(n_rare_correct)
    ]
    common_answers: list[dict | None] = random.choices(common_tools, k=n_common_correct)
    none_answers: list[dict | None] = [None] * n_none

    answer_list = rare_answers + common_answers + none_answers
    random.shuffle(answer_list)

    # ── Assemble examples ────────────────────────────────────────────────
    examples: list[dict] = []
    for tool, count in zip(answer_list, count_buckets):
        count = max(count, 1)
        count = min(count, len(all_tools))
        examples.append(build_example(all_tools, tool, count))

    return examples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tool_router dataset examples.")
    parser.add_argument("--n", type=int, default=100, help="Number of examples to generate.")
    parser.add_argument("--out", type=str, default="sample.json",
                        help="Output path. Use a .json extension for a single file, "
                             "or a directory name to write train/validation/test.jsonl splits.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = generate(args.n, seed=args.seed)
    out = Path(args.out)

    if out.suffix == ".json":
        out.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(examples)} examples → {out}")
    else:
        # Write train / validation / test splits
        out.mkdir(parents=True, exist_ok=True)
        n_train = round(len(examples) * 0.80)
        n_val = round(len(examples) * 0.10)
        splits = {
            "train": examples[:n_train],
            "validation": examples[n_train: n_train + n_val],
            "test": examples[n_train + n_val:],
        }
        for name, split in splits.items():
            path = out / f"{name}.jsonl"
            path.write_text(
                "\n".join(json.dumps(e, ensure_ascii=False) for e in split),
                encoding="utf-8",
            )
            print(f"  {name}: {len(split)} examples → {path}")

    # ── Stats ────────────────────────────────────────────────────────────
    rare_names = {
        "emergency_sos", "roadside_assistance", "insurance_claims",
        "home_automation_bridge", "corp-fleet-manager",
    }
    answers = [e["answer"] for e in examples]
    counts = [len(e["available_tools"]) for e in examples]
    print(f"\nDistribution summary (n={len(examples)}):")
    print(f"  none answers       : {answers.count('none')} ({answers.count('none') / len(examples):.1%})")
    print(f"  rare-tool correct  : {sum(1 for a in answers if a in rare_names)}")
    print(f"  few-tool  (1-3)    : {sum(1 for c in counts if c <= 3)}")
    print(f"  standard  (4-19)   : {sum(1 for c in counts if 4 <= c <= 19)}")
    print(f"  many-tool (20-30)  : {sum(1 for c in counts if c >= 20)}")


if __name__ == "__main__":
    main()

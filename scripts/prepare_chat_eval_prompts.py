"""
Generates the fixed 60-prompt chat evaluation benchmark:
1. Greetings & small talk (8)
2. Simple factual questions (10)
3. Format-constrained instructions (10: one sentence, three bullets, etc.)
4. Rewriting tasks (8)
5. Summarization tasks (8)
6. System-prompt following (6)
7. Multi-turn conversations (8, 3 user turns each)
8. Long-context questions (5: four at 8k depth, one at 30k depth)
Total: exactly 65 prompts (>= 60).
Saves to data/chat_eval_prompts.json.
"""

import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "shards"
CHAT_TOKENIZER_DIR = ROOT / "models" / "tokenizer_chat"
OUT_FILE = ROOT / "data" / "chat_eval_prompts.json"

tokenizer = AutoTokenizer.from_pretrained(str(CHAT_TOKENIZER_DIR))

prompts = []

# 1. Greetings & Small Talk (8)
greetings = [
    "Hello! How are you doing today?",
    "Good morning! Can you help me with a few questions?",
    "Hey there! What can you do?",
    "Hi, nice to meet you. Tell me a little about yourself.",
    "Good evening! Are you ready to assist me?",
    "Hello! I hope you are having a wonderful day.",
    "Greetings! What is your purpose?",
    "Hi! Can you say hello to my friend Alice?",
]
for g in greetings:
    prompts.append({
        "id": f"greet_{len(prompts)+1:02d}",
        "category": "greetings",
        "messages": [{"role": "user", "content": g}],
        "format_requirement": None
    })

# 2. Simple Factual Questions (10)
facts = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("How many days are in a leap year?", "366"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is the chemical symbol for water?", "H2O"),
    ("Who wrote the play Romeo and Juliet?", "William Shakespeare"),
    ("What is the largest ocean on Earth?", "Pacific Ocean"),
    ("How many continents are there on Earth?", "7"),
    ("What gas do plants absorb from the atmosphere during photosynthesis?", "Carbon dioxide"),
    ("What is the freezing point of water in degrees Celsius?", "0"),
]
for q, a in facts:
    prompts.append({
        "id": f"fact_{len(prompts)+1:02d}",
        "category": "factual",
        "messages": [{"role": "user", "content": q}],
        "expected_fact": a,
        "format_requirement": None
    })

# 3. Format-Constrained Instructions (10)
# format requirements: "one_sentence" or "three_bullets"
formats = [
    ("Explain why the sky is blue. Answer in exactly one sentence.", "one_sentence"),
    ("What is gravity? Answer in exactly one sentence.", "one_sentence"),
    ("Define photosynthesis in exactly one sentence.", "one_sentence"),
    ("What is an atom? Answer in exactly one sentence.", "one_sentence"),
    ("Explain why sleep is important. Answer in exactly one sentence.", "one_sentence"),
    ("List why regular exercise is beneficial. Give exactly three bullet points.", "three_bullets"),
    ("Give me three healthy breakfast ideas. Give exactly three bullet points.", "three_bullets"),
    ("What are the three primary states of matter? Give exactly three bullet points.", "three_bullets"),
    ("Name three common renewable energy sources. Give exactly three bullet points.", "three_bullets"),
    ("List three essential items to take on a day hike. Give exactly three bullet points.", "three_bullets"),
]
for q, fmt in formats:
    prompts.append({
        "id": f"format_{len(prompts)+1:02d}",
        "category": "format_constrained",
        "messages": [{"role": "user", "content": q}],
        "format_requirement": fmt
    })

# 4. Rewriting Tasks (8)
rewrites = [
    "Rewrite this sentence to sound polite: I want you to send me the report right now.",
    "Rewrite this sentence to be concise: In spite of the fact that it was raining outside, we nevertheless decided to proceed with our journey.",
    "Rewrite this sentence for a young child: The canine quadruped barked vociferously at the nocturnal intruder.",
    "Rewrite this in formal language: Hey, what's up? Can you check this doc when you get a chance?",
    "Rewrite this sentence in the passive voice: The chef prepared a delicious dinner for all the guests.",
    "Rewrite this sentence to sound enthusiastic: We won the soccer game today.",
    "Rewrite this sentence to remove negative words: It is not impossible that this project will not fail.",
    "Rewrite this sentence to be more descriptive: The car drove down the street.",
]
for r in rewrites:
    prompts.append({
        "id": f"rewrite_{len(prompts)+1:02d}",
        "category": "rewriting",
        "messages": [{"role": "user", "content": r}],
        "format_requirement": None
    })

# 5. Summarization Tasks (8)
summaries = [
    "Summarize in one sentence: The Amazon rainforest, covering much of northwestern Brazil and extending into Colombia, Peru and other South American countries, is the world's largest tropical rainforest, famed for its biodiversity. It is crisscrossed by thousands of rivers, including the powerful Amazon River.",
    "Summarize in one sentence: The Industrial Revolution was the transition to new manufacturing processes in Great Britain, continental Europe, and the United States, that occurred during the period from around 1760 to about 1840. This transition included going from hand production methods to machines and new chemical manufacturing processes.",
    "Summarize in one sentence: Bees are winged insects closely related to wasps and ants, known for their roles in pollination and, in the case of the best-known bee species, the western honey bee, for producing honey. Bees are found on every continent except Antarctica.",
    "Summarize in one sentence: The solar system is the gravitationally bound system of the Sun and the objects that orbit it. It formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud.",
    "Summarize in one sentence: The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups from the Eurasian Steppe.",
    "Summarize in one sentence: Penicillin was discovered in 1928 by Scottish scientist Alexander Fleming as a byproduct from the green mold Penicillium notatum, which effectively killed various bacteria.",
    "Summarize in one sentence: The human brain is the central organ of the human nervous system, and with the spinal cord makes up the central nervous system. It consists of the cerebrum, the brainstem and the cerebellum.",
    "Summarize in one sentence: Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
]
for s in summaries:
    prompts.append({
        "id": f"summary_{len(prompts)+1:02d}",
        "category": "summarization",
        "messages": [{"role": "user", "content": s}],
        "format_requirement": "one_sentence"
    })

# 6. System-Prompt Following (6)
system_cases = [
    ("You are a helpful travel guide assistant.", "What should I visit in Rome?"),
    ("You are a strict math tutor. Only answer with mathematical steps.", "How do I calculate the area of a circle?"),
    ("You are an expert chef who speaks in a warm, culinary style.", "How do I make a simple tomato sauce?"),
    ("You are a medieval knight. Speak in medieval English.", "Who approaches the castle gates?"),
    ("You are a concise technical assistant. Limit all answers to under 30 words.", "What is an API?"),
    ("You are a gentle biology teacher.", "Why do leaves change color in autumn?"),
]
for sys_prompt, user_msg in system_cases:
    prompts.append({
        "id": f"system_{len(prompts)+1:02d}",
        "category": "system_prompt",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg}
        ],
        "format_requirement": None
    })

# 7. Multi-Turn Conversations (8, 3 turns each)
multiturns = [
    [
        {"role": "user", "content": "I am looking for a pet that is low maintenance."},
        {"role": "assistant", "content": "A low-maintenance pet could be a cat, a fish, or a leopard gecko, as they require less daily interaction than dogs."},
        {"role": "user", "content": "Tell me more about caring for a fish."},
        {"role": "assistant", "content": "Fish need a properly sized tank with a filter, regular partial water changes, and consistent daily feeding."},
        {"role": "user", "content": "What is the easiest kind of fish for a beginner?"}
    ],
    [
        {"role": "user", "content": "Can you give me a recipe for pancakes?"},
        {"role": "assistant", "content": "Whisk together 1 cup flour, 1 tbsp sugar, 1 tsp baking powder, and a pinch of salt. In another bowl, mix 1 cup milk, 1 egg, and 2 tbsp melted butter. Combine and cook on a hot buttered skillet until bubbly."},
        {"role": "user", "content": "How do I know when to flip them?"},
        {"role": "assistant", "content": "Flip when bubbles form on the top surface and the edges look set and dry, usually after 2 to 3 minutes."},
        {"role": "user", "content": "What toppings do you recommend?"}
    ],
    [
        {"role": "user", "content": "What is the difference between weather and climate?"},
        {"role": "assistant", "content": "Weather refers to short-term atmospheric conditions over minutes to days, while climate is the average weather pattern over a long period, typically 30 years or more."},
        {"role": "user", "content": "Can climate change cause extreme weather?"},
        {"role": "assistant", "content": "Yes, climate change increases the frequency and intensity of extreme weather events like heatwaves, heavy downpours, and droughts."},
        {"role": "user", "content": "What is one way individuals can help?"}
    ],
    [
        {"role": "user", "content": "I want to learn programming. Which language should I start with?"},
        {"role": "assistant", "content": "Python is widely recommended for beginners because its syntax is clean, readable, and versatile for web development, data science, and automation."},
        {"role": "user", "content": "Is Python also used for artificial intelligence?"},
        {"role": "assistant", "content": "Yes, Python is the most dominant language in AI and machine learning, supported by libraries like PyTorch, TensorFlow, and Hugging Face."},
        {"role": "user", "content": "What is a good first project to build?"}
    ],
    [
        {"role": "user", "content": "What causes earthquakes?"},
        {"role": "assistant", "content": "Earthquakes are caused by the sudden release of energy along geological faults in the Earth's crust as tectonic plates grind past each other."},
        {"role": "user", "content": "Where do most earthquakes happen?"},
        {"role": "assistant", "content": "About 81 percent of the world's largest earthquakes occur along the Ring of Fire around the rim of the Pacific Ocean."},
        {"role": "user", "content": "Can scientists predict when an earthquake will happen?"}
    ],
    [
        {"role": "user", "content": "How does an airplane fly?"},
        {"role": "assistant", "content": "An airplane flies by generating lift through its wings as air flows faster over the curved top surface than the flat bottom, combined with thrust from its engines."},
        {"role": "user", "content": "What happens if an engine fails in mid-air?"},
        {"role": "assistant", "content": "Airplanes are designed to glide safely, and multi-engine planes can climb, cruise, and land safely on a single operating engine."},
        {"role": "user", "content": "How far can a passenger jet glide without engines?"}
    ],
    [
        {"role": "user", "content": "What is the tallest mountain in the world?"},
        {"role": "assistant", "content": "Mount Everest is the highest mountain above sea level, standing at 8,848.86 meters (29,031.7 feet) in the Himalayas."},
        {"role": "user", "content": "Which countries does Mount Everest sit between?"},
        {"role": "assistant", "content": "Mount Everest is located on the international border between Nepal and the Tibet Autonomous Region of China."},
        {"role": "user", "content": "Who were the first people confirmed to have reached the summit?"}
    ],
    [
        {"role": "user", "content": "Why do humans need water to survive?"},
        {"role": "assistant", "content": "Water makes up about 60% of the human body and is required to regulate temperature, transport nutrients, flush waste, and cushion organs."},
        {"role": "user", "content": "How long can a person survive without water?"},
        {"role": "assistant", "content": "Under typical conditions, a human can only survive about 3 to 5 days without water, depending on temperature and exertion."},
        {"role": "user", "content": "What are the earliest signs of dehydration?"}
    ]
]
for conv in multiturns:
    prompts.append({
        "id": f"multiturn_{len(prompts)+1:02d}",
        "category": "multi_turn",
        "messages": conv,
        "format_requirement": None
    })

# 8. Long-Context Cases (5: four at 8k depth, one at 30k depth)
print("Extracting long documents from validation shards...", flush=True)

# 8k Cases from PG-19, Wikipedia, FineWeb
val_pg19 = np.memmap(DATA_DIR / "val_pg19_long.bin", dtype=np.uint16, mode="r")
val_wiki = np.memmap(DATA_DIR / "val_wikipedia_long.bin", dtype=np.uint16, mode="r")
val_fw = np.memmap(DATA_DIR / "val_fineweb_edu_long.bin", dtype=np.uint16, mode="r")

# Case 1: PG-19 8k
toks_pg19_8k = val_pg19[:8000].astype(np.int64)
doc_pg19_8k = tokenizer.decode(toks_pg19_8k)
prompts.append({
    "id": f"long_{len(prompts)+1:02d}",
    "category": "long_context_8k",
    "messages": [
        {"role": "user", "content": f"Here is an excerpt from a book:\n\n{doc_pg19_8k}\n\nBased on the excerpt above, summarize the key events in two sentences."}
    ],
    "context_depth": 8000,
    "format_requirement": None
})

# Case 2: Wikipedia 8k
toks_wiki_8k = val_wiki[:8000].astype(np.int64)
doc_wiki_8k = tokenizer.decode(toks_wiki_8k)
prompts.append({
    "id": f"long_{len(prompts)+1:02d}",
    "category": "long_context_8k",
    "messages": [
        {"role": "user", "content": f"Here is an encyclopedia article:\n\n{doc_wiki_8k}\n\nBased on the article above, what is the main subject discussed?"}
    ],
    "context_depth": 8000,
    "format_requirement": None
})

# Case 3: FineWeb-Edu 8k
toks_fw_8k = val_fw[:8000].astype(np.int64)
doc_fw_8k = tokenizer.decode(toks_fw_8k)
prompts.append({
    "id": f"long_{len(prompts)+1:02d}",
    "category": "long_context_8k",
    "messages": [
        {"role": "user", "content": f"Here is an educational document:\n\n{doc_fw_8k}\n\nWhat are the primary educational takeaways from this document?"}
    ],
    "context_depth": 8000,
    "format_requirement": None
})

# Case 4: PG-19 8k Window 2
toks_pg19_8k_w2 = val_pg19[10000:18000].astype(np.int64)
doc_pg19_8k_w2 = tokenizer.decode(toks_pg19_8k_w2)
prompts.append({
    "id": f"long_{len(prompts)+1:02d}",
    "category": "long_context_8k",
    "messages": [
        {"role": "user", "content": f"Read the following excerpt:\n\n{doc_pg19_8k_w2}\n\nWho are the main characters mentioned in this scene?"}
    ],
    "context_depth": 8000,
    "format_requirement": None
})

# Case 5: PG-19 30k
toks_pg19_30k = val_pg19[20000:50000].astype(np.int64)
doc_pg19_30k = tokenizer.decode(toks_pg19_30k)
prompts.append({
    "id": f"long_{len(prompts)+1:02d}",
    "category": "long_context_30k",
    "messages": [
        {"role": "user", "content": f"Here is a long text of approximately 30,000 tokens:\n\n{doc_pg19_30k}\n\nIn one sentence, what is the overarching theme of this work?"}
    ],
    "context_depth": 30000,
    "format_requirement": "one_sentence"
})

print(f"Total evaluation prompts compiled: {len(prompts)}")
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(prompts, f, indent=2)

print(f"Saved evaluation benchmark to {OUT_FILE}")

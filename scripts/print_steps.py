import json

with open(r'C:\Users\luche\.gemini\antigravity\brain\4108e88f-fd9b-49f3-bf40-12188253dac1\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('step_index') in [5262, 5306]:
            print(f"=== STEP {obj.get('step_index')} ===")
            print(obj.get('content')[:1000])

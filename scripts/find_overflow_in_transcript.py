import json

with open(r'C:\Users\luche\.gemini\antigravity\brain\4108e88f-fd9b-49f3-bf40-12188253dac1\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if 'out of bounds for int32' in line or ('RandomState' in line and 'overflow' in line):
            obj = json.loads(line)
            content = obj.get('content', '')
            print(f"Step {obj.get('step_index')}: {content[:300]}")
            print("-" * 50)

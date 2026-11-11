import json

with open(r'C:\Users\luche\.gemini\antigravity\brain\4108e88f-fd9b-49f3-bf40-12188253dac1\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if 'train_stage_a.py' in line and 'python' in line:
            obj = json.loads(line)
            tool_calls = obj.get('tool_calls', [])
            for tc in tool_calls:
                cmd = str(tc.get('args', {}).get('CommandLine', ''))
                if 'train_stage_a' in cmd:
                    print(f"Step {obj.get('step_index')}: {cmd}")

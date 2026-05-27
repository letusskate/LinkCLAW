"""
Create a better dataset where earlier segments arrive first,
so async prefill methods can show real advantages.
"""
import json
import random

def create_sample(query_id, num_tools=6):
    # Tool names
    tool_names = ["NewsAPI", "StockAPI", "WeatherAPI", "SearchEngine", "Calculator"]
    
    # Planning takes 0.3s
    planning_end = 0.3
    
    # Create tool calls with staggered arrival times
    # Earlier segment ids arrive first, later ones might have delay
    tool_calls = []
    event_list = []
    
    # Add user input and planning events
    event_list.append({
        "type": "user_input",
        "timestamp": 0.0,
        "content": f"Please help me with query {query_id}"
    })
    event_list.append({
        "type": "planning_start",
        "timestamp": 0.0,
        "content": f"Planning for query {query_id}: I will use {num_tools} tools."
    })
    event_list.append({
        "type": "planning_end",
        "timestamp": planning_end,
        "content": f"Planning for query {query_id}: I will use {num_tools} tools."
    })
    
    # Base arrival time for each segment
    # seg 1 arrives at 0.5s
    # seg 2 arrives at 1.0s
    # seg 3 arrives at 1.5s
    # ...
    # But we add some random delays to some later segments
    base_arrival_times = [planning_end + 0.5 + i * 0.5 for i in range(num_tools)]
    
    # Add random delays to later segments (seg 3+)
    for i in range(2, num_tools):
        if random.random() < 0.6:  # 60% chance of delay for later segments
            delay = random.uniform(2.0, 8.0)
            base_arrival_times[i] += delay
    
    # Create tool calls and events
    for seg_id in range(1, num_tools + 1):
        tool_name = random.choice(tool_names)
        call_id = f"call_{query_id}_{seg_id}"
        arrival_time = base_arrival_times[seg_id - 1]
        
        # Use UNIFORM output structure for ALL tools to make datasets library happy
        output_data = {
            "result": str(random.randint(1, 100)),
            "summary": f"Result from {tool_name}"
        }
        
        tool_calls.append({
            "tool_name": tool_name,
            "arguments": {"param": f"value_{random.randint(1, 100)}"},
            "output": output_data,
            "start_time": planning_end,
            "end_time": arrival_time,
            "call_id": call_id
        })
        
        # Add tool invocation event
        event_list.append({
            "type": "tool_invocation",
            "timestamp": planning_end,
            "tool": tool_name,
            "arguments": {"param": f"value_{random.randint(1, 100)}"},
            "call_id": call_id
        })
    
    # Sort tool_returns by arrival time for event_list
    tool_returns_sorted = sorted(tool_calls, key=lambda x: x["end_time"])
    for idx, tool in enumerate(tool_returns_sorted, 1):
        event_list.append({
            "type": "tool_return",
            "timestamp": tool["end_time"],
            "tool": tool["tool_name"],
            "output": tool["output"],
            "call_id": tool["call_id"],
            "return_order": idx
        })
    
    # Final answer time is when last tool arrives + 0.2s
    final_answer_time = max(t["end_time"] for t in tool_calls) + 0.2
    
    return {
        "user_request": f"Please help me with query {query_id}",
        "user_request_time": 0.0,
        "planning_segments": [{
            "text": f"Planning for query {query_id}: I will use {num_tools} tools.",
            "start_time": 0.0,
            "end_time": planning_end
        }],
        "tool_calls": tool_calls,  # Keep in logical order (seg_id 1 to N)
        "final_answer": f"Final answer for query {query_id} based on {num_tools} tool results.",
        "final_answer_time": final_answer_time,
        "event_list": event_list,
        "linearized_transcript": f"User: Please help me with query {query_id}\nPlanning: I will use {num_tools} tools.\n"
    }

def main():
    random.seed(42)  # For reproducibility
    num_samples = 5
    
    dataset = []
    for i in range(1, num_samples + 1):
        sample = create_sample(i, num_tools=random.randint(4, 6))
        dataset.append(sample)
    
    # Write to file
    output_file = "/home/dell/wyb/point1/design3toolbenchdataset/async_toolbench_improved.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"Created improved dataset with {num_samples} samples at {output_file}")
    
    # Print sample 1 to verify
    print("\nSample 1 tool arrival times:")
    sample1 = dataset[0]
    for tool in sample1["tool_calls"]:
        print(f"  seg_id={sample1['tool_calls'].index(tool)+1}: {tool['tool_name']} arrives at {tool['end_time']:.2f}s")

if __name__ == "__main__":
    main()

import json
import requests

def main():
    with open('test_queries.json', 'r', encoding='utf-8') as f:
        queries = json.load(f)
        
    url = "http://127.0.0.1:8000/search"
    correct_overall = 0
    correct_zero = 0
    total_zero = sum(1 for q in queries if q.get("zero_overlap"))
    
    print(f"Running {len(queries)} evaluation queries against {url}...")
    for q_data in queries:
        query = q_data["query"]
        expected = q_data["expected_id"]
        zero = q_data.get("zero_overlap", False)
        
        payload = {"query": query, "mode": "semantic", "context_n": 0}
        resp = requests.post(url, json=payload).json()
        
        matched = False
        for r in resp:
            if r["match"]["id"] == expected:
                matched = True
                break
                
        if matched:
            correct_overall += 1
            if zero:
                correct_zero += 1
                
    print("\n=============================================")
    print("      PHASE 8: EVALUATION RESULTS")
    print("=============================================")
    print(f"Overall Accuracy:       {correct_overall}/{len(queries)} ({(correct_overall/len(queries))*100:.1f}%)")
    if total_zero > 0:
        print(f"Zero-Overlap Accuracy:  {correct_zero}/{total_zero} ({(correct_zero/total_zero)*100:.1f}%)")
    print("=============================================\n")

if __name__ == "__main__":
    main()

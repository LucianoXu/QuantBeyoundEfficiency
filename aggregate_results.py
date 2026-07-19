import json
from collections import defaultdict

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def aggregate_results_pure_python(file_path):
    # Dictionaries to accumulate totals and counts
    overall_sums = defaultdict(float)
    overall_count = 0
    
    category_sums = defaultdict(lambda: defaultdict(float))
    category_counts = defaultdict(int)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Extract and parse values
            score = safe_float(record.get('score', 0))
            ngram_ratio = safe_float(record.get('ngram_ratio', 0))
            
            # Match flags (eval response vs eval label)
            eval_resp = str(record.get('eval_response')).strip()
            eval_lbl = str(record.get('eval_label')).strip()
            eval_match = 1.0 if eval_resp == eval_lbl else 0.0
            
            coh_resp = str(record.get('coherence_response')).strip()
            coh_lbl = str(record.get('coherence_label')).strip()
            coh_match = 1.0 if coh_resp == coh_lbl else 0.0
            
            # Update overall averages
            overall_sums['score'] += score
            overall_sums['ngram_ratio'] += ngram_ratio
            overall_sums['eval_accuracy'] += eval_match
            overall_sums['coherence_accuracy'] += coh_match
            overall_count += 1
            
            # Update category averages
            cat = record.get('category', 'unknown')
            category_sums[cat]['score'] += score
            category_sums[cat]['ngram_ratio'] += ngram_ratio
            category_sums[cat]['eval_accuracy'] += eval_match
            category_sums[cat]['coherence_accuracy'] += coh_match
            category_counts[cat] += 1

    if overall_count == 0:
        print("No valid data processed.")
        return

    # Print overall
    print("=== OVERALL METRICS ===")
    for key, val in overall_sums.items():
        print(f"{key}: {val / overall_count:.4f}")
    print("\n" + "="*30 + "\n")

    # Print grouped by category
    print("=== METRICS BY CATEGORY ===")
    for cat, count in category_counts.items():
        print(f"Category: {cat} (Count: {count})")
        for metric, total in category_sums[cat].items():
            print(f"  {metric}: {total / count:.4f}")
        print()

# Usage example:
# aggregate_results_pure_python('results.jsonl')

aggregate_results_pure_python('results.jsonl')
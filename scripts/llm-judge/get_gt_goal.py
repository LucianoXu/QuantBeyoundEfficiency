import json
import os
from datasets import load_dataset

VALID_STYLES = {
     "L1-self", "L2-self-pressure", "L2-self-reward",
    "L1-other", "L2-other-pressure", "L2-other-reward",
}

CATEGORY_COLUMN = "topic"

def extract_gt_goal() -> None:
    """ Helper script that extracts ground truth and behavioral goals from DeceptionBench.

    Saves a JSONL file 'deception_extra_info.jsonl' containing the records.
    """
    limit = 300

    ds = load_dataset("skyai798/DeceptionBench", split="train")

    records = []

    base_rows = list(ds)
    rows: list[dict] = []
    questions: list[str] = []
    for s in sorted(VALID_STYLES):
        rows.extend(base_rows)
        questions.extend(ds[s])

    rows = rows[:limit]
    questions = questions[:limit]

    for idx, (row, q) in enumerate(zip(rows, questions)):
        category = row.get(CATEGORY_COLUMN) if CATEGORY_COLUMN else None

        question_id = f"{idx}"

        goal = row.get("goal", "")
        ground_truth = row.get("groundtruth", "")
        records.append({
            "question_id": question_id,
            "category": category,
            "prompt_style": "all",
            "question": q,
            "groundtruth": ground_truth,
            "goal": goal,
        })


    with open("deception_extra_info.jsonl", "w") as writefile:
        for record in records:
            writefile.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    os.environ.get("HF_TOKEN")
    extract_gt_goal()


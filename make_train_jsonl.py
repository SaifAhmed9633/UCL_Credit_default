import pandas as pd, json

df = pd.read_csv("UCI_Credit_Card.csv")

cols = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","BILL_AMT1","PAY_AMT1"]
target = "default.payment.next.month"

df = df[cols + [target]].dropna()

with open("train.jsonl", "w", encoding="utf-8") as f:
    for _, r in df.iterrows():
        prompt = (
            "Task: Predict default next month (0 or 1).\n"
            "Input: " + ", ".join([f"{c}={r[c]}" for c in cols]) + "\n"
            "Answer:"
        )
        completion = " " + str(int(r[target]))
        f.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")

print("Created train.jsonl:", len(df), "samples")

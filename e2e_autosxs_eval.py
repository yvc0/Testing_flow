import json
import requests
import vertexai
from vertexai.generative_models import GenerativeModel

# ================= CONFIG =================
CONFIG = {
    "project_id": "your-project-id",
    "location": "us-central1",
    "env": "staging",
    "agent_api": {
        "dev": "http://localhost:8000/agent",
        "staging": "https://staging-api/agent",
        "prod": "https://prod-api/agent"
    }
}
# ==========================================

vertexai.init(project=CONFIG["project_id"], location=CONFIG["location"])
model = GenerativeModel("gemini-1.5-pro")


# ================= CALL AGENT =================
def call_agent(prompt):
    url = CONFIG["agent_api"][CONFIG["env"]]

    res = requests.post(url, json={"query": prompt})
    return res.json().get("response", "")


# ================= POINTWISE =================
def pointwise_eval(prompt, response):
    judge_prompt = f"""
Evaluate the response.

Prompt:
{prompt}

Response:
{response}

Score (1-5):

1. Helpfulness
2. Instruction Following
3. Safety
4. Writing Quality
5. Needs Assessment

Return JSON:
{{
 "helpfulness": score,
 "instruction": score,
 "safety": score,
 "writing": score,
 "needs": score
}}
"""

    res = model.generate_content(judge_prompt)

    try:
        return json.loads(res.text.strip())
    except:
        return {
            "helpfulness": 0,
            "instruction": 0,
            "safety": 0,
            "writing": 0,
            "needs": 0
        }


# ================= PAIRWISE =================
def pairwise_eval(prompt, response, golden):
    judge_prompt = f"""
Compare responses.

Prompt:
{prompt}

Response A (Agent):
{response}

Response B (Golden):
{golden}

Which is better?

Return JSON:
{{
 "winner": "A" or "B"
}}
"""

    res = model.generate_content(judge_prompt)

    try:
        return json.loads(res.text.strip())["winner"]
    except:
        return "B"


# ================= MAIN =================
def evaluate():
    with open("autosxs_dataset.json") as f:
        dataset = json.load(f)

    results = []

    win_count = 0
    total_scores = {
        "helpfulness": 0,
        "instruction": 0,
        "safety": 0,
        "writing": 0,
        "needs": 0
    }

    for row in dataset:
        prompt = row["prompt"]
        golden = row["golden"]

        response = call_agent(prompt)

        scores = pointwise_eval(prompt, response)
        winner = pairwise_eval(prompt, response, golden)

        if winner == "A":
            win_count += 1

        for k in total_scores:
            total_scores[k] += scores[k]

        results.append({
            "prompt": prompt,
            "response": response,
            "golden": golden,
            "scores": scores,
            "winner": winner
        })

        print(f"✅ {prompt}")

    n = len(dataset)

    summary = {
        "win_rate": round(win_count / n, 2),
        "avg_scores": {k: round(v / n, 2) for k, v in total_scores.items()}
    }

    report = {"summary": summary, "details": results}

    with open(f"autosxs_report_{CONFIG['env']}.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\n🎯 AutoSxS Evaluation Completed")
    print(summary)


if __name__ == "__main__":
    evaluate()

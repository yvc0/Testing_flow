import json
import uuid
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


# ================= API =================
def call_agent(query, session_id):
    url = CONFIG["agent_api"][CONFIG["env"]]

    res = requests.post(url, json={
        "query": query,
        "session_id": session_id
    })

    return res.json().get("response", "")


# ================= GEMINI =================
def judge_pointwise(prompt, response):
    eval_prompt = f"""
Prompt: {prompt}
Response: {response}

Score:
1. Helpfulness
2. Instruction Following
3. Safety
4. Writing Quality
5. Needs Assessment

Return JSON
"""
    res = model.generate_content(eval_prompt)

    try:
        return json.loads(res.text.strip())
    except:
        return {}


def judge_pairwise(prompt, response, golden):
    eval_prompt = f"""
Prompt: {prompt}

A: {response}
B: {golden}

Which is better? Return JSON:
{{"winner": "A" or "B"}}
"""
    res = model.generate_content(eval_prompt)

    try:
        return json.loads(res.text.strip())["winner"]
    except:
        return "B"


# ================= MAIN =================
def evaluate():
    with open("autosxs_dataset.json") as f:
        data = json.load(f)

    results = []

    for item in data:
        session_id = str(uuid.uuid4())

        # ===== SINGLE =====
        if "prompt" in item:
            response = call_agent(item["prompt"], session_id)

            scores = judge_pointwise(item["prompt"], response)
            winner = judge_pairwise(item["prompt"], response, item["golden"])

            results.append({
                "type": "single",
                "prompt": item["prompt"],
                "scores": scores,
                "winner": winner
            })

        # ===== MULTI =====
        elif "conversation" in item:
            full_conversation = ""

            for turn in item["conversation"]:
                res = call_agent(turn["user"], session_id)
                full_conversation += f"User: {turn['user']}\nAgent: {res}\n"

            scores = judge_pointwise(full_conversation, full_conversation)
            winner = judge_pairwise(full_conversation, full_conversation, item["golden"])

            results.append({
                "type": "multi",
                "test_name": item["test_name"],
                "scores": scores,
                "winner": winner
            })

    with open(f"autosxs_report_{CONFIG['env']}.json", "w") as f:
        json.dump(results, f, indent=4)

    print("🎯 AutoSxS Hybrid Evaluation Done")


if __name__ == "__main__":
    evaluate()

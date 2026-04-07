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
    "rag_api": {
        "dev": "http://localhost:8000/rag",
        "staging": "https://staging-api/rag",
        "prod": "https://prod-api/rag"
    },
    "top_k": 3
}
# ==========================================

vertexai.init(project=CONFIG["project_id"], location=CONFIG["location"])
model = GenerativeModel("gemini-1.5-pro")


# ================= API =================
def call_rag(query, session_id):
    url = CONFIG["rag_api"][CONFIG["env"]]

    res = requests.post(url, json={
        "query": query,
        "session_id": session_id
    })

    data = res.json()

    return {
        "answer": data.get("answer", ""),
        "contexts": data.get("contexts", [])
    }


# ================= METRICS =================
def context_recall(expected, actual, k):
    actual = actual[:k]
    match = sum(1 for e in expected if any(e.lower() in a.lower() for a in actual))
    return match / len(expected) if expected else 0


def gemini_judge(question, answer, contexts):
    prompt = f"""
Question: {question}
Answer: {answer}
Contexts: {" ".join(contexts)}

Score:
1. Grounding (1-5)
2. Relevance (1-5)

Return JSON:
{{"grounding": score, "relevance": score}}
"""
    res = model.generate_content(prompt)

    try:
        return json.loads(res.text.strip())
    except:
        return {"grounding": 0, "relevance": 0}


# ================= MAIN =================
def evaluate():
    with open("rag_dataset.json") as f:
        data = json.load(f)

    results = []

    for item in data:
        session_id = str(uuid.uuid4())

        # ===== SINGLE TURN =====
        if "question" in item:
            rag = call_rag(item["question"], session_id)

            recall = context_recall(item["expected_contexts"], rag["contexts"], CONFIG["top_k"])
            scores = gemini_judge(item["question"], rag["answer"], rag["contexts"])

            results.append({
                "type": "single",
                "question": item["question"],
                "recall": recall,
                "scores": scores
            })

        # ===== MULTI TURN =====
        elif "conversation" in item:
            conv_results = []

            for turn in item["conversation"]:
                rag = call_rag(turn["user"], session_id)

                recall = context_recall(turn["expected_contexts"], rag["contexts"], CONFIG["top_k"])
                scores = gemini_judge(turn["user"], rag["answer"], rag["contexts"])

                conv_results.append({
                    "user": turn["user"],
                    "recall": recall,
                    "scores": scores
                })

            results.append({
                "type": "multi",
                "test_name": item.get("test_name", ""),
                "turns": conv_results
            })

    with open(f"rag_report_{CONFIG['env']}.json", "w") as f:
        json.dump(results, f, indent=4)

    print("🎯 RAG Hybrid Evaluation Done")


if __name__ == "__main__":
    evaluate()

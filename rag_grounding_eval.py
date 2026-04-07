import json
import requests
import vertexai
from vertexai.generative_models import GenerativeModel

# ================= CONFIG =================
CONFIG = {
    "project_id": "your-project-id",
    "location": "us-central1",
    "env": "staging",  # dev / staging / prod
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


# ================= API CALL =================
def call_rag(question):
    url = CONFIG["rag_api"][CONFIG["env"]]

    res = requests.post(url, json={"query": question})
    data = res.json()

    return {
        "answer": data.get("answer", ""),
        "contexts": data.get("contexts", [])
    }


# ================= METRIC 1 =================
def context_recall_at_k(expected, actual, k):
    actual_k = actual[:k]

    match = 0
    for exp in expected:
        if any(exp.lower() in act.lower() for act in actual_k):
            match += 1

    return match / len(expected) if expected else 0


# ================= GEMINI JUDGE =================
def evaluate_with_gemini(question, answer, contexts):
    context_text = "\n".join(contexts)

    prompt = f"""
You are an expert evaluator.

Question:
{question}

Answer:
{answer}

Contexts:
{context_text}

Evaluate:

1. Grounding (Faithfulness):
- Is answer fully supported by context?
Score: 1-5

2. Relevance:
- Does answer directly answer the question?
Score: 1-5

Return ONLY JSON:
{{
  "grounding": score,
  "relevance": score
}}
"""

    response = model.generate_content(prompt)

    try:
        return json.loads(response.text.strip())
    except:
        return {"grounding": 0, "relevance": 0}


# ================= MAIN =================
def evaluate():
    with open("rag_dataset.json") as f:
        dataset = json.load(f)

    results = []

    total_recall = 0
    total_grounding = 0
    total_relevance = 0

    for row in dataset:
        question = row["question"]
        expected = row["expected_contexts"]

        rag_output = call_rag(question)

        answer = rag_output["answer"]
        contexts = rag_output["contexts"]

        recall = context_recall_at_k(expected, contexts, CONFIG["top_k"])
        scores = evaluate_with_gemini(question, answer, contexts)

        total_recall += recall
        total_grounding += scores["grounding"]
        total_relevance += scores["relevance"]

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "recall@k": round(recall, 2),
            "grounding": scores["grounding"],
            "relevance": scores["relevance"]
        })

        print(f"✅ {question}")

    n = len(dataset)

    summary = {
        "avg_recall@k": round(total_recall / n, 2),
        "avg_grounding": round(total_grounding / n, 2),
        "avg_relevance": round(total_relevance / n, 2)
    }

    report = {"summary": summary, "details": results}

    with open(f"rag_report_{CONFIG['env']}.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\n🎯 RAG Evaluation Completed")
    print(summary)


if __name__ == "__main__":
    evaluate()

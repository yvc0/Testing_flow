import csv
import uuid
import json
from rapidfuzz import fuzz
from google.cloud import dialogflowcx_v3 as dialogflow

PROJECT_ID = "your-project-id"
LOCATION = "us-central1"
LANGUAGE_CODE = "en"

AGENT_V1 = "agent-id-v1"
AGENT_V2 = "agent-id-v2"

client = dialogflow.SessionsClient()


def detect(agent_id, session_id, text):
    session_path = f"projects/{PROJECT_ID}/locations/{LOCATION}/agents/{agent_id}/sessions/{session_id}"

    response = client.detect_intent(
        request={
            "session": session_path,
            "query_input": dialogflow.QueryInput(
                text=dialogflow.TextInput(text=text),
                language_code=LANGUAGE_CODE
            )
        }
    )

    messages = []
    for msg in response.query_result.response_messages:
        if msg.text:
            messages.extend(msg.text.text)

    return " ".join(messages)


def judge(res1, res2, expected):
    score1 = fuzz.token_sort_ratio(res1, expected)
    score2 = fuzz.token_sort_ratio(res2, expected)

    if score1 > score2:
        return "V1"
    elif score2 > score1:
        return "V2"
    return "TIE"


def evaluate(csv_file):
    results = []

    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            test_name = row["test_name"]
            user_input = row["input"]
            expected = row["expected"]

            session_id = str(uuid.uuid4())

            res1 = detect(AGENT_V1, session_id, user_input)
            res2 = detect(AGENT_V2, session_id, user_input)

            winner = judge(res1, res2, expected)

            results.append({
                "test_name": test_name,
                "input": user_input,
                "expected": expected,
                "v1": res1,
                "v2": res2,
                "winner": winner
            })

            print(f"{test_name}: Winner → {winner}")

    return {"results": results}


if __name__ == "__main__":
    report = evaluate("testdata/sxs_tests.csv")

    with open("reports/autosxs_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\n✅ Auto SxS Completed")

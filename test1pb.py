import csv
import uuid
import json
from google.cloud import dialogflowcx_v3 as dialogflow

PROJECT_ID = "your-project-id"
AGENT_ID = "your-agent-id"
LOCATION = "us-central1"
LANGUAGE_CODE = "en"

client = dialogflow.SessionsClient()

def detect_intent(session_id, text):
    session_path = f"projects/{PROJECT_ID}/locations/{LOCATION}/agents/{AGENT_ID}/sessions/{session_id}"

    text_input = dialogflow.TextInput(text=text)
    query_input = dialogflow.QueryInput(
        text=text_input,
        language_code=LANGUAGE_CODE
    )

    response = client.detect_intent(
        request={"session": session_path, "query_input": query_input}
    )

    messages = []
    for msg in response.query_result.response_messages:
        if msg.text:
            messages.extend(msg.text.text)

    return " ".join(messages)


def run_conversation(conversation_steps):
    session_id = str(uuid.uuid4())  # SAME session for full flow
    last_response = ""

    for step in conversation_steps:
        last_response = detect_intent(session_id, step.strip())

    return last_response


def evaluate(csv_file):
    results = []
    total = 0
    passed = 0

    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            total += 1

            test_name = row["display_name"]
            conversation = row["conversation"].split("||")
            expected = row["expected_output"]

            actual = run_conversation(conversation)

            # Validation
            if expected.lower() in actual.lower():
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"

            results.append({
                "test_name": test_name,
                "conversation": conversation,
                "expected": expected,
                "actual": actual,
                "status": status
            })

            print(f"{test_name}: {status}")

    accuracy = (passed / total) * 100 if total > 0 else 0

    report = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": round(accuracy, 2),
        "results": results
    }

    return report


if __name__ == "__main__":
    report = evaluate("testdata/playbook_tests.csv")

    with open("reports/playbook_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n🎯 Accuracy: {report['accuracy']}%")

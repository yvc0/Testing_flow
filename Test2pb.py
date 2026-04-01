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


def evaluate(csv_file):
    results = []
    total = 0
    passed = 0

    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            total += 1

            test_name = row["display_name"]
            user_steps = row["conversation"].split("||")
            expected_steps = row["expected_responses"].split("||")

            session_id = str(uuid.uuid4())

            step_results = []
            all_pass = True

            for i, user_input in enumerate(user_steps):
                actual_response = detect_intent(session_id, user_input.strip())
                expected_response = expected_steps[i].strip()

                # validation
                if expected_response.lower() in actual_response.lower():
                    status = "PASS"
                else:
                    status = "FAIL"
                    all_pass = False

                step_results.append({
                    "step": i + 1,
                    "user_input": user_input,
                    "expected": expected_response,
                    "actual": actual_response,
                    "status": status
                })

            if all_pass:
                passed += 1
                final_status = "PASS"
            else:
                final_status = "FAIL"

            results.append({
                "test_name": test_name,
                "steps": step_results,
                "final_status": final_status
            })

            print(f"{test_name}: {final_status}")

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

    with open("reports/playbook_detailed_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n🎯 Accuracy: {report['accuracy']}%")

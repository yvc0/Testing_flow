import csv
import uuid
import json
from google.cloud import dialogflowcx_v3 as dialogflow

PROJECT_ID = "your-project-id"
AGENT_ID = "your-agent-id"
LOCATION = "us-central1"
LANGUAGE_CODE = "en"

client = dialogflow.SessionsClient()


def detect_intent_with_tool(session_id, text):
    session_path = f"projects/{PROJECT_ID}/locations/{LOCATION}/agents/{AGENT_ID}/sessions/{session_id}"

    response = client.detect_intent(
        request={
            "session": session_path,
            "query_input": dialogflow.QueryInput(
                text=dialogflow.TextInput(text=text),
                language_code=LANGUAGE_CODE
            )
        }
    )

    query_result = response.query_result

    # Extract tool / fulfillment tag
    intent = query_result.intent.display_name if query_result.intent else ""
    parameters = dict(query_result.parameters)

    # Tool detection logic (customize based on your agent)
    tool_used = "UNKNOWN"

    if "address" in intent.lower():
        tool_used = "MCP"
    elif "sales" in intent.lower():
        tool_used = "SALES_TOOL"

    return {
        "response": " ".join(
            [t for msg in query_result.response_messages if msg.text for t in msg.text.text]
        ),
        "intent": intent,
        "tool": tool_used
    }


def evaluate(csv_file):
    results = []
    total = 0
    passed = 0

    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            total += 1

            test_name = row["test_name"]
            user_input = row["input"]
            expected_tool = row["expected_tool"]

            session_id = str(uuid.uuid4())

            result = detect_intent_with_tool(session_id, user_input)

            actual_tool = result["tool"]

            if actual_tool == expected_tool:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"

            results.append({
                "test_name": test_name,
                "input": user_input,
                "expected_tool": expected_tool,
                "actual_tool": actual_tool,
                "intent": result["intent"],
                "status": status
            })

            print(f"{test_name}: {status} ({actual_tool})")

    accuracy = (passed / total) * 100 if total else 0

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": round(accuracy, 2),
        "results": results
    }


if __name__ == "__main__":
    report = evaluate("testdata/tool_routing_tests.csv")

    with open("reports/tool_routing_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n🎯 Tool Routing Accuracy: {report['accuracy']}%")

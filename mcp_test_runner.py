import csv
import uuid
import json
import os
from datetime import datetime
from google.cloud import dialogflowcx_v3 as dialogflow

# ================= CONFIG =================
PROJECT_ID = "your-project-id"
AGENT_ID = "your-agent-id"
LOCATION = "us-central1"
LANGUAGE_CODE = "en"

CSV_FILE = "testdata/tool_routing_tests.csv"
REPORT_FILE = "reports/tool_routing_report.json"
MCP_LOG_FILE = "logs/mcp_calls.log"   # optional (if logging enabled)

# ==========================================

client = dialogflow.SessionsClient()


# 🔥 DETECT INTENT + TOOL + INVOCATION
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

    qr = response.query_result

    intent = qr.intent.display_name if qr.intent else ""

    # 🔥 diagnostic info for tool invocation
    diagnostic = dict(qr.diagnostic_info)

    # Extract response text
    messages = []
    for msg in qr.response_messages:
        if msg.text:
            messages.extend(msg.text.text)

    final_response = " ".join(messages)

    # 🔥 TOOL DETECTION
    tool_used = "UNKNOWN"
    tool_invoked = False

    # Check diagnostic info (strong signal)
    if "toolExecution" in str(diagnostic):
        tool_invoked = True

    # Intent-based mapping (customize as needed)
    if "address" in intent.lower():
        tool_used = "MCP"
    elif "sales" in intent.lower() or "plan" in intent.lower():
        tool_used = "SALES_TOOL"

    return {
        "response": final_response,
        "intent": intent,
        "tool": tool_used,
        "tool_invoked": tool_invoked,
        "diagnostic": diagnostic
    }


# 🔥 OPTIONAL: MCP LOG VALIDATION
def check_mcp_log(session_id):
    if not os.path.exists(MCP_LOG_FILE):
        return False

    with open(MCP_LOG_FILE, "r") as f:
        logs = f.read()

    return session_id in logs


# 🔥 MAIN EVALUATION
def evaluate(csv_file):
    results = []
    total = 0
    passed = 0

    print("\n🚀 Starting Tool Routing Validation...\n")

    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            total += 1

            test_name = row["test_name"]
            user_input = row["input"]
            expected_tool = row["expected_tool"]
            expected_keyword = row["expected_keyword"]

            session_id = str(uuid.uuid4())

            result = detect_intent_with_tool(session_id, user_input)

            actual_tool = result["tool"]
            tool_invoked = result["tool_invoked"]
            response = result["response"]

            # 🔥 VALIDATIONS
            tool_match = actual_tool == expected_tool
            invocation_check = tool_invoked
            grounding_check = expected_keyword.lower() in response.lower()

            # 🔥 OPTIONAL MCP LOG CHECK
            mcp_log_check = True
            if expected_tool == "MCP":
                mcp_log_check = check_mcp_log(session_id)

            # FINAL DECISION
            if tool_match and invocation_check and grounding_check and mcp_log_check:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"

            results.append({
                "test_name": test_name,
                "input": user_input,
                "expected_tool": expected_tool,
                "actual_tool": actual_tool,
                "tool_invoked": tool_invoked,
                "grounding_check": grounding_check,
                "mcp_log_check": mcp_log_check,
                "intent": result["intent"],
                "response": response,
                "status": status
            })

            print(f"{test_name}: {status}")
            print(f"   → Tool: {actual_tool}, Invoked: {tool_invoked}, Grounded: {grounding_check}")

    accuracy = (passed / total) * 100 if total else 0

    report = {
        "timestamp": str(datetime.now()),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": round(accuracy, 2),
        "results": results
    }

    return report


# 🔥 ENTRY POINT
if __name__ == "__main__":
    report = evaluate(CSV_FILE)

    os.makedirs("reports", exist_ok=True)

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=4)

    print("\n==============================")
    print(f"🎯 Accuracy: {report['accuracy']}%")
    print(f"📄 Report saved: {REPORT_FILE}")
    print("==============================")

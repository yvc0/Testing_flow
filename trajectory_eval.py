import json
import uuid
import requests
from datetime import datetime

# ================= CONFIG =================
AGENT_API = "http://localhost:8000/agent"  # your ADK agent endpoint
DATA_FILE = "testdata/trajectory_tests.json"
REPORT_FILE = "reports/trajectory_report.json"
# ==========================================


def call_agent(user_input, session_id):
    """
    Calls your ADK agent API
    EXPECTED RESPONSE FORMAT:
    {
        "response": "...",
        "tool_calls": [
            {
                "tool": "sales_tool",
                "params": {"intent": "buy"}
            }
        ]
    }
    """

    payload = {
        "session_id": session_id,
        "query": user_input
    }

    try:
        res = requests.post(AGENT_API, json=payload, timeout=10)
        return res.json()
    except Exception as e:
        return {
            "response": "ERROR",
            "tool_calls": [],
            "error": str(e)
        }


def compare_tools(expected, actual):
    """
    Compare expected vs actual tool calls
    """
    tool_correct = 0
    param_correct = 0

    for exp, act in zip(expected, actual):
        # Tool match
        if exp["tool"] == act.get("tool"):
            tool_correct += 1

        # Param match
        exp_params = exp.get("params", {})
        act_params = act.get("params", {})

        for key in exp_params:
            if key in act_params and str(exp_params[key]).lower() in str(act_params[key]).lower():
                param_correct += 1

    return tool_correct, param_correct


def evaluate():
    with open(DATA_FILE, "r") as f:
        scenarios = json.load(f)

    total_turns = 0
    correct_tools = 0
    total_params = 0
    correct_params = 0
    full_success = 0

    results = []

    print("\n🚀 Starting Trajectory Evaluation...\n")

    for scenario in scenarios:
        session_id = str(uuid.uuid4())
        scenario_pass = True

        step_results = []

        for step in scenario["conversation"]:
            total_turns += 1

            user_input = step["user"]
            expected_tools = step["expected_tools"]

            res = call_agent(user_input, session_id)

            actual_tools = res.get("tool_calls", [])

            # Metrics
            tool_match, param_match = compare_tools(expected_tools, actual_tools)

            correct_tools += tool_match
            correct_params += param_match
            total_params += sum(len(t.get("params", {})) for t in expected_tools)

            # Step status
            if tool_match < len(expected_tools):
                step_status = "FAIL"
                scenario_pass = False
            else:
                step_status = "PASS"

            step_results.append({
                "user_input": user_input,
                "expected": expected_tools,
                "actual": actual_tools,
                "status": step_status
            })

            print(f"Step: {user_input} → {step_status}")

        if scenario_pass:
            full_success += 1

        results.append({
            "test_name": scenario["test_name"],
            "steps": step_results,
            "final_status": "PASS" if scenario_pass else "FAIL"
        })

    # Metrics
    tool_accuracy = (correct_tools / total_turns) * 100 if total_turns else 0
    param_accuracy = (correct_params / total_params) * 100 if total_params else 0
    trajectory_accuracy = (full_success / len(scenarios)) * 100 if scenarios else 0

    report = {
        "timestamp": str(datetime.now()),
        "tool_selection_accuracy": round(tool_accuracy, 2),
        "parameter_accuracy": round(param_accuracy, 2),
        "trajectory_accuracy": round(trajectory_accuracy, 2),
        "results": results
    }

    return report


if __name__ == "__main__":
    report = evaluate()

    import os
    os.makedirs("reports", exist_ok=True)

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=4)

    print("\n==============================")
    print(f"🎯 Tool Accuracy: {report['tool_selection_accuracy']}%")
    print(f"🎯 Param Accuracy: {report['parameter_accuracy']}%")
    print(f"🎯 Trajectory Accuracy: {report['trajectory_accuracy']}%")
    print("==============================")

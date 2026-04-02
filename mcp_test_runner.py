import csv
import json
import uuid
import httpx

MCP_API = "http://localhost:8000/execute"


def call_mcp(input_text):
    payload = {
        "tool": "address_validator",
        "input": input_text
    }

    response = httpx.post(MCP_API, json=payload)

    if response.status_code == 200:
        return response.json().get("output", "")
    return "ERROR"


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
            expected = row["expected"]

            actual = call_mcp(user_input)

            if expected.lower() in actual.lower():
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"

            results.append({
                "test_name": test_name,
                "input": user_input,
                "expected": expected,
                "actual": actual,
                "status": status
            })

            print(f"{test_name}: {status}")

    accuracy = (passed / total) * 100 if total else 0

    report = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": round(accuracy, 2),
        "results": results
    }

    return report


if __name__ == "__main__":
    report = evaluate("testdata/mcp_tests.csv")

    with open("reports/mcp_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n🎯 MCP Accuracy: {report['accuracy']}%")

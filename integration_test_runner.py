import csv
import json
import asyncio
import httpx

API_URL = "http://localhost:9000/query"


async def call_api(client, query):
    response = await client.post(API_URL, json={"query": query})
    return response.text


async def evaluate(csv_file):
    results = []
    total = 0
    passed = 0

    async with httpx.AsyncClient(timeout=10.0) as client:
        with open(csv_file, newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                total += 1

                test_name = row["test_name"]
                query = row["input"]
                expected = row["expected"]

                actual = await call_api(client, query)

                if expected.lower() in actual.lower():
                    status = "PASS"
                    passed += 1
                else:
                    status = "FAIL"

                results.append({
                    "test_name": test_name,
                    "input": query,
                    "expected": expected,
                    "actual": actual,
                    "status": status
                })

                print(f"{test_name}: {status}")

    accuracy = (passed / total) * 100 if total else 0

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": round(accuracy, 2),
        "results": results
    }


if __name__ == "__main__":
    report = asyncio.run(evaluate("testdata/integration_tests.csv"))

    with open("reports/integration_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n🎯 Integration Accuracy: {report['accuracy']}%")

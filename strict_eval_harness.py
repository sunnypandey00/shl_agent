import os
import glob
import requests
import json
import time
import sys

# Define strict keys
EXPECTED_KEYS = {"reply", "recommendations", "end_of_conversation"}
# Define recommendation keys
REC_KEYS = {"name", "url", "test_type"}
BASE_URL = os.getenv("SHL_RECOMMENDER_BASE_URL", "http://localhost:8000")
TEST_DIR = os.getenv(
    "SHL_RECOMMENDER_TEST_DIR",
    r"C:\Users\mesun\Desktop\assessment\GenAI_SampleConversations",
)


def validate_schema(data, turn_count):
    # Validate turn limit
    if turn_count > 8:
        raise ValueError(f"Turn cap violated: {turn_count} > 8")

    # Check base keys
    actual_keys = set(data.keys())
    if actual_keys != EXPECTED_KEYS:
        raise ValueError(f"Invalid keys. Expected {EXPECTED_KEYS}, got {actual_keys}")

    # Verify string reply
    if not isinstance(data.get("reply"), str) or not data["reply"].strip():
        raise ValueError("Field 'reply' must be a non-empty string.")

    # Verify boolean flag
    if not isinstance(data.get("end_of_conversation"), bool):
        raise TypeError("Field 'end_of_conversation' must be a strict boolean.")

    # Extract recs array
    recs = data.get("recommendations")
    if recs is not None:
        # Enforce list type
        if not isinstance(recs, list):
            raise TypeError("Field 'recommendations' must be a list or null.")

        # Enforce array length
        if len(recs) > 10:
            raise ValueError(
                f"Recommendations array exceeds 10 items (count: {len(recs)})."
            )

        # Iterate recommendation items
        for idx, r in enumerate(recs):
            # Assert dict type
            if not isinstance(r, dict):
                raise TypeError(f"Recommendation at index {idx} is not a dictionary.")

            # Assert precise keys
            r_keys = set(r.keys())
            if r_keys != REC_KEYS:
                raise ValueError(
                    f"Rec {idx} invalid keys. Expected {REC_KEYS}, got {r_keys}"
                )

            # Assert field types
            if not isinstance(r["name"], str):
                raise TypeError(f"Rec {idx} 'name' must be string.")
            if not isinstance(r["url"], str):
                raise TypeError(f"Rec {idx} 'url' must be string.")
            if not isinstance(r["test_type"], str):
                raise TypeError(f"Rec {idx} 'test_type' must be string.")

            # Enforce valid URL
            if not r["url"].startswith("https://www.shl.com/"):
                raise ValueError(
                    f"Rec {idx} 'url' fails hallucination check: {r['url']}"
                )


def parse_markdown_trace(filepath):
    # Initialize message list
    user_messages = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    # Parse trace loops
    while i < len(lines):
        if lines[i].strip() == "**User**":
            i += 1
            while i < len(lines):
                if lines[i].strip().startswith(">"):
                    user_messages.append(lines[i].replace(">", "").strip())
                    break
                i += 1
        i += 1

    # Return parsed messages
    return user_messages


def run_strict_tests():
    # Ping health endpoint
    print("Checking server health...")
    try:
        resp = requests.get(f"{BASE_URL}/health")
        if resp.status_code != 200:
            print("Server not healthy.")
            sys.exit(1)
    except Exception:
        print("Server unreachable.")
        sys.exit(1)

    print("Server is up. Starting STRICT evaluation...\n")

    md_files = glob.glob(os.path.join(TEST_DIR, "*.md"))

    passed = 0
    failed = 0

    # Iterate conversation files
    for file in md_files:
        trace_name = os.path.basename(file)
        print(f"[{trace_name}] Starting trace evaluation...")
        # Load user inputs
        user_messages = parse_markdown_trace(file)
        messages_payload = []
        turn_count = 0

        trace_failed = False

        # Replay user messages
        for msg in user_messages:
            turn_count += 1
            messages_payload.append({"role": "user", "content": msg})

            # Check budget breach
            if turn_count > 4:
                print(f"[{trace_name}] FAIL: Conversation exceeded user turn budget.")
                trace_failed = True
                break

            try:
                payload = {"messages": messages_payload}

                # Add retry loop
                for attempt in range(4):
                    response = requests.post(
                        f"{BASE_URL}/chat", json=payload, timeout=40
                    )
                    if response.status_code in [429, 500]:
                        print(f"  Hit 429/500 API Error. Retrying in 30s...")
                        time.sleep(30)
                        continue
                    response.raise_for_status()
                    break
                else:
                    raise Exception("Max retries exceeded on API errors.")

                data = response.json()

                print(f"  Turn {turn_count} JSON Dump:")
                print(f"  {json.dumps(data, indent=2)}")

                # Execute rigid validations
                validate_schema(data, turn_count * 2)

                # Update local state
                messages_payload.append(
                    {"role": "assistant", "content": data.get("reply")}
                )

                # Detect conversation end
                if data.get("end_of_conversation"):
                    print(f"[{trace_name}] Agent terminated conversation correctly.")
                    break

            except Exception as e:
                print(f"[{trace_name}] FAIL: Validation exception: {str(e)}")
                trace_failed = True
                break

            time.sleep(12)

        if not trace_failed:
            print(f"[{trace_name}] PASS: All schemas and budgets honored.\n")
            passed += 1
        else:
            failed += 1
            print("\n")

    # Output final score
    print(f"EVALUATION COMPLETE: {passed} Passed | {failed} Failed")


if __name__ == "__main__":
    run_strict_tests()


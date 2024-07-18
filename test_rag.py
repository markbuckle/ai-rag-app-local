from query_data import query_rag
import boto3
import json
import datetime
import warnings

# Replace utcfromtimestamp
_EPOCH_DATETIME_NAIVE = datetime.datetime.fromtimestamp(0, datetime.timezone.utc)

# Replace utcnow
datetime_now = datetime.datetime.now(datetime.timezone.utc)

warnings.filterwarnings(
    "ignore",
    message="datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).",
    category=DeprecationWarning,
)

warnings.filterwarnings(
    "ignore",
    message="datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).",
    category=DeprecationWarning,
)


EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def get_evaluation_response(prompt):
    session = boto3.Session(profile_name="MBAdmin")
    client = session.client("bedrock-runtime", region_name="us-east-1")
    model_id = "amazon.titan-text-premier-v1:0"

    native_request = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.5,
        },
    }

    request = json.dumps(native_request)

    try:
        response = client.invoke_model(modelId=model_id, body=request)
        response_body = json.loads(response["body"].read())
        evaluation_results_str = response_body["results"][0]["outputText"]
        return evaluation_results_str.strip().lower()
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        return None


def test_rules():
    assert query_and_validate(
        question="What does PRP stand for?",
        expected_response="Platelet Rich Plasma",
    )


# helper function
def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    evaluation_results_str_cleaned = get_evaluation_response(prompt)

    print(prompt)

    # interpret result
    if evaluation_results_str_cleaned is None:
        raise ValueError("Failed to get evaluation response from the model.")

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )


if __name__ == "__main__":
    test_rules()

import json
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ragas_llm import RagasLLM


# =========================================================
# Files
# =========================================================

DATASET_FILE = (
    PROJECT_ROOT /
    "data" /
    "ragas_dataset.json"
)

RESULT_FILE = (
    PROJECT_ROOT /
    "data" /
    "ragas_results.json"
)


# =========================================================
# Required thresholds
# =========================================================

THRESHOLDS = {
    "faithfulness": 0.90,
    "answer_correctness": 0.80,
    "context_recall": 0.85,
    "context_precision": 0.80
}


# =========================================================
# Load dataset
# =========================================================

def load_dataset():

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =========================================================
# Build evaluation prompt
# =========================================================

def build_evaluation_prompt(item):

    question = item["question"]
    answer = item["answer"]
    contexts = item["contexts"]
    reference = item["reference"]

    context_text = "\n\n".join(
        f"CONTEXT {i + 1}:\n{context}"
        for i, context in enumerate(contexts)
    )

    prompt = f"""
You are evaluating a Retrieval-Augmented Generation (RAG) system.

Evaluate the answer using the question, reference answer,
and retrieved contexts.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

GENERATED ANSWER:
{answer}

RETRIEVED CONTEXTS:
{context_text}

Evaluate four dimensions.

1. FAITHFULNESS

Does the generated answer contain only information supported
by the retrieved contexts?

2. ANSWER CORRECTNESS

How correctly does the generated answer answer the question
compared with the reference answer?

3. CONTEXT RECALL

Do the retrieved contexts contain the information required
to answer the question correctly?

4. CONTEXT PRECISION

How much of the retrieved context is relevant to answering
the question?

Scoring:

0.0 = completely incorrect
0.25 = poor
0.50 = partially correct
0.75 = mostly correct
1.0 = excellent

Return ONLY valid JSON.

Do not include markdown.
Do not include explanations outside JSON.

Required format:

{{
    "faithfulness": 0.0,
    "answer_correctness": 0.0,
    "context_recall": 0.0,
    "context_precision": 0.0
}}
"""

    return prompt


# =========================================================
# Parse evaluator response
# =========================================================

def parse_scores(response):

    response = response.strip()

    # Remove markdown code fences if model adds them
    if response.startswith("```"):

        response = response.replace(
            "```json",
            ""
        )

        response = response.replace(
            "```",
            ""
        )

        response = response.strip()

    try:

        data = json.loads(response)

    except json.JSONDecodeError:

        # Try to extract JSON object
        start = response.find("{")
        end = response.rfind("}")

        if start == -1 or end == -1:

            raise ValueError(
                f"Evaluator did not return valid JSON:\n{response}"
            )

        data = json.loads(
            response[start:end + 1]
        )

    required_metrics = [
        "faithfulness",
        "answer_correctness",
        "context_recall",
        "context_precision"
    ]

    scores = {}

    for metric in required_metrics:

        value = data.get(metric)

        if value is None:

            raise ValueError(
                f"Missing metric: {metric}"
            )

        value = float(value)

        # Keep scores between 0 and 1
        value = max(
            0.0,
            min(1.0, value)
        )

        scores[metric] = value

    return scores


# =========================================================
# Evaluate one question
# =========================================================

def evaluate_question(
    item,
    evaluator_llm,
    question_number,
    total_questions
):

    print()
    print("=" * 70)
    print(
        f"QUESTION {question_number}/{total_questions}"
    )
    print("=" * 70)

    print(
        f"Question: {item['question']}"
    )

    prompt = build_evaluation_prompt(item)

    print()
    print("Calling evaluation LLM...")

    try:

        response = evaluator_llm.invoke(
            prompt
        )

        # LangChain AIMessage
        if hasattr(response, "content"):

            response_text = response.content

        else:

            response_text = str(response)

        print()
        print("Evaluator response:")
        print(response_text)

        scores = parse_scores(
            response_text
        )

        print()
        print("Scores:")

        for metric, score in scores.items():

            print(
                f"  {metric:25} "
                f"{score * 100:.2f}%"
            )

        return {
            "id": item.get("id"),
            "question": item["question"],
            "document": item.get("document"),
            "topic": item.get("topic"),
            "type": item.get("type"),
            "scores": scores,
            "evaluation_status": "SUCCESS"
        }

    except Exception as e:

        print()
        print(
            f"ERROR evaluating question: "
            f"{type(e).__name__}: {e}"
        )

        return {
            "id": item.get("id"),
            "question": item["question"],
            "document": item.get("document"),
            "topic": item.get("topic"),
            "type": item.get("type"),
            "scores": {
                "faithfulness": None,
                "answer_correctness": None,
                "context_recall": None,
                "context_precision": None
            },
            "evaluation_status": "FAILED",
            "error": str(e)
        }


# =========================================================
# Calculate averages
# =========================================================

def calculate_scores(question_results):

    scores = {}

    for metric in THRESHOLDS:

        values = []

        for result in question_results:

            value = result["scores"].get(
                metric
            )

            if value is not None:

                values.append(value)

        if values:

            scores[metric] = (
                sum(values) /
                len(values)
            )

        else:

            scores[metric] = 0.0

    return scores


# =========================================================
# Check thresholds
# =========================================================

def calculate_status(scores):

    metric_status = {}

    for metric, threshold in THRESHOLDS.items():

        score = scores[metric]

        metric_status[metric] = {
            "score": score,
            "threshold": threshold,
            "passed": score > threshold
        }

    overall_pass = all(
        item["passed"]
        for item in metric_status.values()
    )

    return (
        metric_status,
        "PASS" if overall_pass else "FAIL"
    )


# =========================================================
# Save results
# =========================================================

def save_results(
    question_results,
    scores,
    metric_status,
    overall_result,
    status="COMPLETED"
):

    output = {

        "evaluation_date":
            datetime.now().isoformat(),

        "evaluation_method":
            "Sequential single-call LLM evaluation",

        "total_questions":
            len(question_results),

        "thresholds": {
            metric: threshold * 100
            for metric, threshold
            in THRESHOLDS.items()
        },

        "scores": {
            metric: score * 100
            for metric, score
            in scores.items()
        },

        "metric_status":
            metric_status,

        "overall_result":
            overall_result,

        "evaluation_status":
            status,

        "question_results":
            question_results
    }

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    return output


# =========================================================
# Main
# =========================================================

def run_ragas_evaluation():

    print("=" * 70)
    print("RAG EVALUATION")
    print("=" * 70)

    dataset = load_dataset()

    print(
        f"Evaluation questions: {len(dataset)}"
    )

    # -----------------------------------------------------
    # Create evaluator
    # -----------------------------------------------------

    print()
    print("Creating evaluation LLM...")

    evaluator_llm = RagasLLM().get_llm()

    print("Evaluation LLM ready.")

    # -----------------------------------------------------
    # Evaluate sequentially
    # -----------------------------------------------------

    question_results = []

    total_questions = len(dataset)

    print()
    print("=" * 70)
    print("STARTING SEQUENTIAL EVALUATION")
    print("=" * 70)

    for index, item in enumerate(dataset):

        result = evaluate_question(
            item,
            evaluator_llm,
            index + 1,
            total_questions
        )

        question_results.append(result)

        # -------------------------------------------------
        # Save progress after every question
        # -------------------------------------------------

        save_results(
            question_results,
            {},
            {},
            "IN_PROGRESS",
            status="IN_PROGRESS"
        )

        print()
        print(
            f"Completed "
            f"{index + 1}/{total_questions}"
        )

        # Give local Llama a small break
        time.sleep(2)

    # -----------------------------------------------------
    # Calculate final scores
    # -----------------------------------------------------

    scores = calculate_scores(
        question_results
    )

    metric_status, overall_result = (
        calculate_status(scores)
    )

    # -----------------------------------------------------
    # Print final results
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)

    for metric, data in metric_status.items():

        print(
            f"{metric:25} "
            f"{data['score'] * 100:6.2f}% "
            f"(Required > "
            f"{data['threshold'] * 100:.0f}%) "
            f"{'PASS' if data['passed'] else 'FAIL'}"
        )

    print()
    print(
        f"OVERALL RESULT: {overall_result}"
    )

    # -----------------------------------------------------
    # Save final result
    # -----------------------------------------------------

    save_results(
        question_results,
        scores,
        metric_status,
        overall_result,
        status="COMPLETED"
    )

    print()
    print(
        f"Results saved to:\n{RESULT_FILE}"
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    run_ragas_evaluation()
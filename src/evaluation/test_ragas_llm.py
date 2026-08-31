from src.evaluation.ragas_llm import RagasLLM

def main():

    print("=" * 60)
    print("Testing RAGAS Evaluation LLM")
    print("=" * 60)

    evaluator = RagasLLM()

    llm = evaluator.get_llm()

    response = llm.invoke(
        "Explain in one sentence what the Transformer architecture is."
    )

    print("\nLLM Response:")
    print(response.content)

    print("\nEvaluation LLM is working.")


if __name__ == "__main__":
    main()
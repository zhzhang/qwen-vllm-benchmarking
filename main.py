from vllm import LLM
from datasets import load_dataset

splits = [
    "anatomy",
    "business_ethics",
    "clinical_knowledge",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "econometrics",
    "electrical_engineering",
    "formal_logic",
    "global_facts",
    "high_school_chemistry",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "human_aging",
    "logical_fallacies",
    "machine_learning",
    "miscellaneous",
    "philosophy",
    "professional_accounting",
    "public_relations",
    "virology",
    "conceptual_physics",
    "high_school_us_history",
    "astronomy",
    "high_school_geography",
    "high_school_macroeconomics",
    "professional_law",
]


def main():
    dataset = load_dataset("edinburgh-dawg/mmlu-redux", split=splits[0])
    prompts = dataset["train"]["question"]

    llm = LLM(
        model="Qwen/Qwen3-4B",
    )
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    outputs = llm.generate(prompts)
    print(outputs)


if __name__ == "__main__":
    main()

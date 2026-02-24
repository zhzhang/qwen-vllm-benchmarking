from vllm import LLM
from datasets import load_dataset

subsets = [
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

def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def main():
    dataset = load_dataset("edinburgh-dawg/mmlu-redux", subsets[0], split="test")

    llm = LLM(
        model="Qwen/Qwen3-4B",
    )

    for example in dataset:
        question = example["question"]
        choices = example["choices"]
        answer = example["answer"]
        formatted_choices = ""
        for i, choice in enumerate(choices):
            formatted_choices += f"{chr(65 + i)}) {choice}\n"
        prompt = f"Question: {question}\nOptions:\n{formatted_choices}Answer: Let's think step by step.\n\n"
        responses = llm.generate(prompt, max_tokens=1024)
        print(responses)
        for response in responses:
            for output in response.outputs:
                print(output.text)


if __name__ == "__main__":
    main()

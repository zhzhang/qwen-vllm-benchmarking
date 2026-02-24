from vllm import LLM, SamplingParams
from datasets import load_dataset
import re

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
    match = re.search(r".*[aA]nswer:\s*([A-J])", text)
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


def run_subset(subset_name, llm, sampling_params, batch_size=1):
    """Run evaluation on a single subset. Returns (correct_count, total_count)."""
    dataset = load_dataset("edinburgh-dawg/mmlu-redux", subset_name, split="test")

    correct_count = 0
    total_count = 0
    for batch_start in range(0, len(dataset), batch_size):
        batch = dataset[batch_start : batch_start + batch_size]
        questions = batch["question"]
        choices_list = batch["choices"]
        answers = batch["answer"]
        prompts = []
        correct_letters = []
        for i in range(len(questions)):
            question = questions[i]
            choices = choices_list[i]
            correct_answer_idx = answers[i]
            correct_answer_letter = chr(65 + correct_answer_idx)
            correct_letters.append(correct_answer_letter)
            formatted_choices = ""
            for j, choice in enumerate(choices):
                formatted_choices += f"{chr(65 + j)}) {choice}\n"
            prompt = f"Question: {question}\nOptions:\n{formatted_choices}Answer: Let's think step by step.\n\n"
            prompts.append(prompt)
        responses = llm.generate(prompts, sampling_params=sampling_params)
        for response, correct_answer_letter in zip(responses, correct_letters):
            assert len(response.outputs) == 1
            output = response.outputs[0]
            answer = extract_answer(output.text)
            if answer == correct_answer_letter:
                correct_count += 1
            total_count += 1
            print(
                f"[{subset_name}] [{total_count}] Correct: {correct_count}, Total: {total_count}, Accuracy: {correct_count / total_count}"
            )
    return correct_count, total_count


def main(batch_size=1):
    llm = LLM(
        model="Qwen/Qwen3-4B",
    )
    sampling_params = SamplingParams(max_tokens=512)

    total_correct = 0
    total_all = 0
    for subset_name in subsets:
        correct, total = run_subset(subset_name, llm, sampling_params, batch_size)
        total_correct += correct
        total_all += total

    print(
        f"\nOverall: Correct: {total_correct}, Total: {total_all}, Accuracy: {total_correct / total_all}"
    )


if __name__ == "__main__":
    main(batch_size=15)

# Spec 10.10: how much does the train score of one fixed prompt move between evaluations?
#   python -m experiments.jiyul.noise --seed 42 --repeats 5
# "full" repeats task generation + judging; "judge_only" re-judges the first run's outputs.
import argparse
import json
import statistics

from experiments import cdroot
from crispo.llms.openai_compat import OpenAICompatLLM
from crispo.trainer.trainer import Trainer
from experiments.jiyul import config as C
from experiments.jiyul.judge_metric import RubricJudgeMetric
from experiments.samsum.dataset import load_samsum_shuffled_quick_train_dev_test
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    cdroot()
    train, _, _ = load_samsum_shuffled_quick_train_dev_test(random_seed=args.seed)
    task_llm = OpenAICompatLLM(C.TASK_MODEL, C.TASK_URL, C.TASK_TOKENIZER, C.TASK_REASONING_EFFORT, temperature=0.0)
    judge_llm = OpenAICompatLLM(C.META_MODEL, C.META_URL, C.META_TOKENIZER, C.REASONING_EFFORT, temperature=0.0)
    judge = RubricJudgeMetric(judge_llm, C.RUBRIC)
    prompt = SummarizationTaskPromptWithPlaceholder(C.INITIAL_PROMPT)
    trainer = Trainer(f"data/experiments/jiyul/noise_seed{args.seed}")

    full, outputs = [], []
    for i in range(args.repeats):
        score, _, predictions = trainer.evaluate(prompt, train, task_llm, judge, desc=f"full {i}")
        full.append(score)
        outputs.append(predictions)
    judge_only = [statistics.mean(judge.batch_score(outputs[0], train)) for _ in range(args.repeats)]
    same_outputs = statistics.mean(
        sum(a == b for a, b in zip(outputs[0], o)) / len(train) for o in outputs[1:]
    )

    result = {
        "seed": args.seed,
        "n_train": len(train),
        "full": full,
        "full_sd": statistics.stdev(full),
        "judge_only": judge_only,
        "judge_only_sd": statistics.stdev(judge_only),
        "task_outputs_identical_to_run0": same_outputs,
    }
    with open(f"data/experiments/jiyul/noise_seed{args.seed}/noise.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

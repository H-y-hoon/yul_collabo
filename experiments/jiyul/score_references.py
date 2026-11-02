# Ceiling check: score SAMSum reference (human) summaries with the same judge, on the same train split.
#   python -m experiments.jiyul.score_references --seed 42
# If references score about the same as the initial prompt's outputs, the judge cannot reward a better summary.
import argparse
import json
import os
import statistics

import pandas as pd

from experiments import cdroot
from crispo.llms.openai_compat import OpenAICompatLLM
from experiments.jiyul import config as C
from experiments.jiyul.judge_metric import RubricJudgeMetric
from experiments.jiyul.run import load_samsum_train_dev_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cdroot()
    train, _, _ = load_samsum_train_dev_test(random_seed=args.seed)
    judge_llm = OpenAICompatLLM(C.META_MODEL, C.META_URL, C.META_TOKENIZER, C.REASONING_EFFORT, temperature=0.0)
    judge = RubricJudgeMetric(judge_llm, C.RUBRIC)
    scores = judge.batch_score([e.y for e in train], train)

    save_dir = f"data/experiments/jiyul/references_seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame({
        "x": [e.x for e in train],
        "reference": [e.y for e in train],
        "score": scores,
        "comments": [json.dumps(judge.judgments[(e.x, e.y)]["comments"]) for e in train],
    })
    df.to_csv(f"{save_dir}/references.csv")

    result = {"seed": args.seed, "n": len(train), "reference_judge": statistics.mean(scores),
              "parse_failures": judge.parse_failures}
    # the initial prompt's outputs on the same split, scored during the loop (step 0)
    initial = f"data/experiments/jiyul/jiyul_seed{args.seed}/step-000/train-prompt-001.csv"
    if os.path.exists(initial):
        m = df.merge(pd.read_csv(initial)[["x", "score"]], on="x", suffixes=("_ref", "_init"))
        result["initial_prompt_judge"] = m.score_init.mean()
        result["ref_better_same_worse"] = [int((m.score_ref > m.score_init).sum()), int((m.score_ref == m.score_init).sum()),
                                           int((m.score_ref < m.score_init).sum())]
    with open(f"{save_dir}/summary.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

# Runs one condition of the Jiyul PoC reproduction (spec 4.4-4.5) on SAMSum and writes summary.json.
#   python -m experiments.jiyul.run --method jiyul --seed 42
#   python -m experiments.jiyul.run --method opro --seed 42
import argparse
import json
import random
import statistics

import pandas as pd
from rouge_score.rouge_scorer import RougeScorer

from experiments import cdroot
from crispo.llms.openai_compat import OpenAICompatLLM
from crispo.trainer.trainer import Trainer
from experiments.jiyul import config as C
from experiments.jiyul.judge_metric import RubricJudgeMetric
from experiments.jiyul.rca_prompt import RCAPrompt
from experiments.jiyul.solgen_meta_prompt import SolGenMetaPrompt
from experiments.samsum.dataset import load_samsum, load_samsum_debug_train_dev_test
from experiments.summarization.constants import ARTICLE_PLACEHOLDER
from experiments.summarization.meta_prompt import SummarizationOproMetaPrompt
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder
from crispo.utilities.prompt_util import extract_xml_tag

ROLE_HEADS = {  # first characters of each meta-level prompt, used to split meta_llm cost by role
    "rca": "# Role\nYou are a failure root cause analysis agent",
    "solution_generation": "# Role\nYou are an LLM optimizer",
    "opro_optimizer": "Your task is to generate the instruction",
}


class OproJudgeMetaPrompt(SummarizationOproMetaPrompt):
    """OPRO meta-prompt whose candidates keep the input placeholder, so both conditions share one prompt format."""

    def parse(self, generation: str):
        new_inst = extract_xml_tag(generation, "instruction")
        if not new_inst or new_inst.count(ARTICLE_PLACEHOLDER) > 1:  # >1 would insert the dialogue twice
            return None
        if ARTICLE_PLACEHOLDER not in new_inst:
            new_inst = f"Given the input:\n\n{ARTICLE_PLACEHOLDER}\n\n" + new_inst
        if "<summary>" not in new_inst:
            new_inst += " Write your summary within <summary> tags."
        return SummarizationTaskPromptWithPlaceholder(new_inst)


class JudgeWithRougeFloor:
    """
    Picks the next current prompt (spec 10.14): the best judge score among the new candidates whose train ROUGE-1
    is no more than ``floor`` below the current prompt's. If every candidate falls below, it takes the best judge
    score anyway so the loop keeps moving, and records the fallback.
    With ``keep_current`` the current prompt competes with the candidates, so it stays when they are all worse
    (spec 10.16).
    """

    def __init__(self, train, floor, keep_current=False):
        self.train, self.floor, self.keep_current = train, floor, keep_current
        self.rouge1 = {}
        self.history = []
        self._rouge = RougeScorer(["rouge1"], use_stemmer=True)

    def __call__(self, candidates, current, evaluated, prompt_score_pairs):
        for p in candidates:
            predictions = evaluated[p][1]
            self.rouge1[p] = statistics.mean(
                self._rouge.score(e.y, pred)["rouge1"].fmeasure for pred, e in zip(predictions, self.train)
            )
        passed = [p for p in candidates if current is None or self.rouge1[p] >= self.rouge1[current] - self.floor]
        if self.keep_current and current is not None:
            passed.append(current)  # listed last, so a candidate wins a tie
        chosen = max(passed or candidates, key=prompt_score_pairs.get)
        self.history.append({
            "candidates": [{"judge": prompt_score_pairs[p], "rouge1": self.rouge1[p], "passed": p in passed} for p in candidates],
            "chosen_judge": prompt_score_pairs[chosen],
            "chosen_rouge1": self.rouge1[chosen],
            "fallback": not passed,
            "kept_current": chosen == current,
        })
        return chosen


def load_samsum_train_dev_test(random_seed):
    """Same sampling as CriSPO's load_samsum_shuffled_quick_train_dev_test, with the sizes from config."""
    random.seed(random_seed)
    train = random.sample(load_samsum(split="train"), C.TRAIN_SIZE)
    dev = random.sample(load_samsum(split="validation"), C.DEV_SIZE)
    test = random.sample(load_samsum(split="test"), C.TEST_SIZE)
    return train, dev, test


def tally(records):
    return {
        "calls": len(records),
        "prompt_tokens": sum(r["prompt_tokens"] for r in records),
        "completion_tokens": sum(r["completion_tokens"] for r in records),
        "content_tokens": sum(r["content_tokens"] for r in records),
        "truncated": sum(r["finish_reason"] == "length" for r in records),
        "server_errors": sum(r["finish_reason"] == "server_error" for r in records),
    }


def evaluate_on_test(trainer, prompt, test, task_llm, judge, save_path):
    n_task = len(task_llm.records)
    judge_score, _, predictions = trainer.evaluate(prompt, test, task_llm, judge, desc="test", save_path=save_path)
    rouge = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(e.y, p) for p, e in zip(predictions, test)]
    task_records = task_llm.records[n_task:]
    return {
        "judge": judge_score,
        "rouge": dict(
            (k, statistics.mean(s[k].fmeasure for s in rouge_scores)) for k in ["rouge1", "rouge2", "rougeL"]
        ),
        "output_tokens_mean": {
            "content": statistics.mean(r["content_tokens"] for r in task_records),
            "completion_incl_reasoning": statistics.mean(r["completion_tokens"] for r in task_records),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["jiyul", "opro"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="2 examples per split")
    parser.add_argument("--steps", type=int, default=C.MAX_STEPS)
    parser.add_argument("--k", type=int, default=C.K)
    parser.add_argument("--stop_score", type=float, default=C.STOP_SCORE)
    args = parser.parse_args()

    cdroot()
    load = load_samsum_debug_train_dev_test if args.debug else load_samsum_train_dev_test
    train, dev, test = load(random_seed=args.seed)
    save_dir = f"data/experiments/jiyul/{args.method}_seed{args.seed}{'_debug' if args.debug else ''}"

    task_llm = OpenAICompatLLM(C.TASK_MODEL, C.TASK_URL, C.TASK_TOKENIZER, C.TASK_REASONING_EFFORT, temperature=0.0)
    meta_llm = OpenAICompatLLM(C.META_MODEL, C.META_URL, C.META_TOKENIZER, C.REASONING_EFFORT, temperature=C.META_TEMPERATURE)
    judge_llm = OpenAICompatLLM(C.META_MODEL, C.META_URL, C.META_TOKENIZER, C.REASONING_EFFORT, temperature=0.0)
    judge = RubricJudgeMetric(judge_llm, C.RUBRIC)

    common = dict(
        train=train,
        dev=None,  # dev is scored below, outside the loop, so it never influences selection
        initial_task_prompts={SummarizationTaskPromptWithPlaceholder(C.INITIAL_PROMPT)},
        meta_llm=meta_llm,
        task_llm=task_llm,
        metric=judge,
        num_search_steps=args.steps,
        num_new_prompts_in_each_step=args.k,
        num_few_shot_examples_in_meta_prompt=0,  # no reference summaries reach the optimizer (spec 4.5)
        stop_score=args.stop_score,
    )
    trainer = Trainer(save_dir)
    selector = JudgeWithRougeFloor(train, C.ROUGE_FLOOR)
    if args.method == "jiyul":
        prompt_score_pairs, _ = trainer.fit(
            meta_prompt=SolGenMetaPrompt(guardrail=C.GUARDRAIL),
            critique_prompt=RCAPrompt(judge, meta_llm, C.RCA_BATCH_SIZE),
            num_examples_in_critique=C.NUM_EXAMPLES_IN_RCA,
            num_task_prompts_in_meta_prompt=1,
            critique_selected_only=True,
            replace_current=True,  # the selected candidate always becomes the current prompt (spec 10.12)
            select_next=selector,
            **common,
        )
    else:
        prompt_score_pairs, _ = trainer.fit(
            meta_prompt=OproJudgeMetaPrompt(),
            num_task_prompts_in_meta_prompt=C.OPRO_NUM_PROMPTS_IN_META_PROMPT,
            **common,
        )
    loop_cost = {"task": tally(task_llm.records), "judge": tally(judge_llm.records)}
    for role, head in ROLE_HEADS.items():
        records = [r for r in meta_llm.records if r["head"].startswith(head)]
        if records:
            loop_cost[role] = tally(records)

    # best-so-far prompt after each step (the selected "current prompt"), scored on dev
    history = pd.read_json(f"{save_dir}/prompts.json")
    by_text = dict((str(p), p) for p in prompt_score_pairs)
    curve, dev_scores = [], {}
    for step in range(history["step"].max() + 1):
        best = history[history["step"] <= step].sort_values("score", ascending=False, kind="stable").iloc[0]
        if best["prompt"] not in dev_scores:
            dev_scores[best["prompt"]], _, _ = trainer.evaluate(
                by_text[best["prompt"]], dev, task_llm, judge, desc=f"dev step {step}",
                save_path=f"{save_dir}/dev-best-step-{step:03}.csv",
            )
        curve.append({
            "step": step,
            "new_prompts_train": history[history["step"] == step]["score"].tolist(),
            "best_so_far_train": best["score"],
            "best_so_far_dev": dev_scores[best["prompt"]],
        })

    final_prompt = next(iter(prompt_score_pairs))  # sorted best-first by the Trainer
    final_test = evaluate_on_test(trainer, final_prompt, test, task_llm, judge, f"{save_dir}/test-final.csv")
    # no-optimization baseline: the shared initial prompt on the same test set
    initial_prompt = SummarizationTaskPromptWithPlaceholder(C.INITIAL_PROMPT)
    initial_test = evaluate_on_test(trainer, initial_prompt, test, task_llm, judge, f"{save_dir}/test-initial.csv")

    summary = {
        "method": args.method,
        "seed": args.seed,
        "args": vars(args),
        "stop_reason": "stop_score" if prompt_score_pairs[final_prompt] >= args.stop_score else "max_steps",
        "steps_run": int(history["step"].max()),
        "final_prompt": str(final_prompt),
        "final_train_judge": prompt_score_pairs[final_prompt],
        "curve": curve,
        "selection": selector.history,  # jiyul only: one entry per step, the step-0 entry is the initial prompt
        **dict((f"test_{k}", v) for k, v in final_test.items()),
        "initial_prompt_test": initial_test,
        "loop_cost": loop_cost,
        "judge_parse_failures": judge.parse_failures,
    }
    with open(f"{save_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

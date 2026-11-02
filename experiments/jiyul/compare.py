# Method comparison on SAMSum (spec 10.15): each method optimizes with its own signal, picks its final prompt on
# dev with that signal, and every final prompt is scored on test with the task's official metric (ROUGE).
#   python -m experiments.jiyul.compare --method poc --seed 42
#   python -m experiments.jiyul.compare --method opro --seed 42
#   python -m experiments.jiyul.compare --method crispo --seed 42
# --backend claude (default) calls the Anthropic API with C.CLAUDE_MODEL; --backend vllm calls the gpt-oss servers.
import argparse
import json
import statistics

from dotenv import load_dotenv
from rouge_score.rouge_scorer import RougeScorer

from experiments import cdroot
from crispo.llms.anthropic_llm import AnthropicLLM
from crispo.llms.openai_compat import OpenAICompatLLM
from crispo.trainer.trainer import Trainer
from experiments.jiyul import config as C
from experiments.jiyul.judge_metric import JudgeRougeMetric
from experiments.jiyul.rca_prompt import RCAPrompt
from experiments.jiyul.run import JudgeWithRougeFloor, OproJudgeMetaPrompt, load_samsum_train_dev_test, tally
from experiments.jiyul.solgen_meta_prompt import SolGenMetaPrompt
from experiments.samsum.dataset import load_samsum_debug_train_dev_test
from experiments.summarization.constants import ARTICLE_PLACEHOLDER
from experiments.summarization.crispo import SummarizationCritiquePrompt, SummarizationCriSPOMetaPrompt
from experiments.summarization.metrics.rouge1 import Rouge1Fmeasure
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder

ROLE_HEADS = {  # first characters of each meta-level prompt, used to split meta_llm cost by role
    "rca": "# Role\nYou are a failure root cause analysis agent",
    "solution_generation": "# Role\nYou are an LLM optimizer",
    "opro_optimizer": "Your task is to generate the instruction",
    "crispo_critique": "In a summarization task, a writer is given",
    "crispo_optimizer": "Your task is to optimize the instruction",
}


class CriSPOMetaPrompt(SummarizationCriSPOMetaPrompt):
    """Upstream CriSPO meta-prompt; only rejects a candidate with two input placeholders, as the other methods do."""

    def parse(self, generation: str):
        new_prompt = super().parse(generation)
        if new_prompt and new_prompt.prompt.count(ARTICLE_PLACEHOLDER) > 1:  # would insert the dialogue twice
            return None
        return new_prompt


def make_llm(backend, role):
    """role: "task" (the summarizer), "meta" (RCA, Solution Generation, critique, optimizer) or "judge"."""
    effort = C.TASK_REASONING_EFFORT if role == "task" else C.REASONING_EFFORT
    if backend == "claude":
        return AnthropicLLM(C.CLAUDE_MODEL, effort)
    if role == "task":
        return OpenAICompatLLM(C.TASK_MODEL, C.TASK_URL, C.TASK_TOKENIZER, effort, temperature=0.0)
    temperature = C.META_TEMPERATURE if role == "meta" else 0.0
    return OpenAICompatLLM(C.META_MODEL, C.META_URL, C.META_TOKENIZER, effort, temperature=temperature)


def score_on_test(trainer, prompt, test, task_llm, save_path):
    n_task = len(task_llm.records)
    _, _, predictions = trainer.evaluate(prompt, test, task_llm, Rouge1Fmeasure(), desc="test", save_path=save_path)
    rouge = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [rouge.score(e.y, p) for p, e in zip(predictions, test)]
    task_records = task_llm.records[n_task:]
    return {
        "rouge": dict((k, statistics.mean(s[k].fmeasure for s in scores)) for k in ["rouge1", "rouge2", "rougeL"]),
        "words_mean": statistics.mean(len(p.split()) for p in predictions),
        "empty": sum(not p.strip() for p in predictions),
        "output_tokens_mean": {
            "content": statistics.mean(r["content_tokens"] for r in task_records),
            "completion_incl_reasoning": statistics.mean(r["completion_tokens"] for r in task_records),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["poc", "opro", "crispo"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="2 examples per split")
    parser.add_argument("--steps", type=int, default=100)  # CriSPO's SAMSum setting
    parser.add_argument("--backend", choices=["claude", "vllm"], default="claude")
    args = parser.parse_args()

    load_dotenv()  # ANTHROPIC_API_KEY from .env, if there is one
    cdroot()
    load = load_samsum_debug_train_dev_test if args.debug else load_samsum_train_dev_test
    train, dev, test = load(random_seed=args.seed)
    name = f"{args.method}_{args.backend}" + ("" if args.steps == 100 else f"_{args.steps}steps")
    save_dir = f"data/experiments/jiyul/compare/{name}_seed{args.seed}{'_debug' if args.debug else ''}"

    task_llm = make_llm(args.backend, "task")
    meta_llm = make_llm(args.backend, "meta")
    initial_prompt = SummarizationTaskPromptWithPlaceholder(C.INITIAL_PROMPT)
    common = dict(  # shared by all methods (CriSPO's SAMSum runner settings unless noted)
        train=train,
        dev=dev,
        initial_task_prompts={initial_prompt},
        meta_llm=meta_llm,
        task_llm=task_llm,
        num_search_steps=args.steps,
        num_new_prompts_in_each_step=C.K,
        dev_evaluation_per_n_steps=5,
    )
    trainer = Trainer(save_dir)
    selector, judge_llm = None, None
    if args.method == "poc":  # judge (qualitative) + ROUGE (quantitative), spec 10.12-10.20
        judge_llm = make_llm(args.backend, "judge")
        # select (train and dev) on judge + scaled ROUGE-1, so no ROUGE floor; selector "judge" fields hold the
        # combined score (10.17). The current prompt competes with the candidates (10.16).
        judge = JudgeRougeMetric(judge_llm, C.RUBRIC, C.ROUGE_WEIGHT, C.ROUGE_SCALE)
        selector = JudgeWithRougeFloor(train, float("inf"), keep_current=True)
        prompt_score_pairs, best_prompt_dev = trainer.fit(
            metric=judge,
            # Solution Generation asks for a concise prompt (OPRO/CriSPO, 10.18) under 5 lines (ZERA, 10.19)
            meta_prompt=SolGenMetaPrompt(guardrail=C.GUARDRAIL, concise=True, line_limit=True),
            # the RCA sees the means over all outputs plus the 3 best and 2 worst outputs, as ZERA does (10.20)
            critique_prompt=RCAPrompt(judge, meta_llm, C.RCA_BATCH_SIZE, top_bottom=(3, 2)),
            num_examples_in_critique=C.NUM_EXAMPLES_IN_RCA,
            num_task_prompts_in_meta_prompt=1,
            num_few_shot_examples_in_meta_prompt=0,
            critique_selected_only=True,
            replace_current=True,
            select_next=selector,
            stop_score=C.STOP_SCORE,
            **common,
        )
    else:  # the published methods, optimizing ROUGE-1 with 2 random reference examples in the meta-prompt
        crispo = args.method == "crispo"
        prompt_score_pairs, best_prompt_dev = trainer.fit(
            metric=Rouge1Fmeasure(),
            meta_prompt=CriSPOMetaPrompt() if crispo else OproJudgeMetaPrompt(),
            critique_prompt=SummarizationCritiquePrompt() if crispo else None,
            num_few_shot_examples_in_meta_prompt=2,
            few_shot_selection_criteria="random",
            **common,
        )
    final_prompt = best_prompt_dev or next(iter(prompt_score_pairs))

    loop_cost = {"task": tally(task_llm.records)}
    if judge_llm:
        loop_cost["judge"] = tally(judge_llm.records)
    for role, head in ROLE_HEADS.items():
        records = [r for r in meta_llm.records if r["head"].startswith(head)]
        if records:
            loop_cost[role] = tally(records)

    summary = {
        "method": args.method,
        "seed": args.seed,
        "args": vars(args),
        "model": C.CLAUDE_MODEL if args.backend == "claude" else {"task": C.TASK_MODEL, "meta_and_judge": C.META_MODEL},
        "optimization_signal": "rubric judge + scaled ROUGE-1" if args.method == "poc" else "ROUGE-1",
        "final_prompt": str(final_prompt),
        "final_prompt_words": len(str(final_prompt).split()),
        "final_train_signal": prompt_score_pairs[final_prompt],
        "test_final": score_on_test(trainer, final_prompt, test, task_llm, f"{save_dir}/test-final.csv"),
        "test_initial": score_on_test(trainer, initial_prompt, test, task_llm, f"{save_dir}/test-initial.csv"),
        "loop_cost": loop_cost,
    }
    if selector:
        summary["selection"] = selector.history
    with open(f"{save_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

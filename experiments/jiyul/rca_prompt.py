import statistics
from typing import Union, Any, List

from rouge_score.rouge_scorer import RougeScorer

from crispo.task.critique import CritiquePrompt
from crispo.task.example import Example
from crispo.utilities.prompt_util import extract_xml_tag
from experiments.jiyul.judge_metric import RubricJudgeMetric

_P = """
# Role
You are a failure root cause analysis agent for prompt optimization. A task LLM ran the current prompt on a set of dialogues. Each output was evaluated quantitatively (ROUGE against a reference summary) and qualitatively (an evaluation agent applied a rubric and gave an overall score and a comment per criterion). Find out what in the current prompt caused the shortcomings in the evaluation, and set an action policy for revising the prompt.

# Current Prompt
<current_prompt>
{prompt}
</current_prompt>

# Eval Results
{eval_results}

# Instructions
1. Based on the eval results, identify the root causes in the current prompt: point to the specific parts of the prompt (or what is missing from it) that led to the low scores and negative comments.
2. For each root cause, set an action policy: a concrete, general rule for how the prompt should be changed.

Write the root cause analysis in <root_cause> tags and the action policy in <action_policy> tags.
""".strip()

_MERGE = """
# Role
You are a failure root cause analysis agent for prompt optimization. The eval results of the current prompt were too many to read at once, so they were split into batches and each batch was analyzed separately. Merge the batch analyses into one root cause analysis and one action policy for the current prompt.

# Current Prompt
<current_prompt>
{prompt}
</current_prompt>

# Eval Results
{eval_results}

# Batch Analyses
{analyses}

# Instructions
1. Keep the root causes that recur across batches or that explain the largest score losses, and point to the specific parts of the prompt (or what is missing from it).
2. Write one consistent action policy: a concrete, general rule for each root cause. Resolve suggestions that conflict between batches.

Write the root cause analysis in <root_cause> tags and the action policy in <action_policy> tags.
""".strip()

_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
_ROUGE = RougeScorer(_ROUGE_TYPES, use_stemmer=True)


class RCAPrompt(CritiquePrompt):
    """
    PoC "Root Cause Analysis Meta-Prompt". Per the PoC figure its inputs are only the current prompt and the
    eval results (quantitative + qualitative); the dialogues, the outputs and the references are not shown.
    ``parse`` returns the eval results together with the analysis because the Solution Generation
    meta-prompt also takes both.
    """

    def __init__(self, judge: RubricJudgeMetric, llm=None, batch_size=None, top_bottom=None) -> None:
        self.judge = judge
        self.llm = llm  # runs the per-batch analyses when the eval results come in more than one batch
        self.batch_size = batch_size
        self.top_bottom = top_bottom  # (n_top, n_bottom): show only those outputs, as ZERA does (spec 10.20)
        self._eval_results = None

    def fill(
        self,
        prompt,
        predictions: List[Union[str, Any]],
        few_shot_examples: List[Example],
    ) -> str:
        judgments = [self.judge.judgments[(e.x, p)] for p, e in zip(predictions, few_shot_examples)]
        # quantitative = a metric computed against the reference; the judge's score is a rubric rating, so it
        # is reported as qualitative (spec 10.12). Only numbers reach the RCA, never the reference text.
        rouges = [_ROUGE.score(e.y, p) for p, e in zip(predictions, few_shot_examples)]
        if self.top_bottom:  # means over all outputs, per-output results only for the best and worst ones
            self._eval_results = _top_bottom_block(judgments, rouges, *self.top_bottom, self.judge.output_score)
            return _P.format(prompt=prompt, eval_results=self._eval_results)
        size = self.batch_size or len(judgments)
        batches = [slice(i, i + size) for i in range(0, len(judgments), size)]
        if len(batches) == 1:
            self._eval_results = _eval_block(judgments, rouges)
            return _P.format(prompt=prompt, eval_results=self._eval_results)

        # all outputs do not fit in one context: analyze each batch, then merge the analyses (spec 10.13)
        generations = self.llm.batch_generate(
            [_P.format(prompt=prompt, eval_results=_eval_block(judgments[b], rouges[b])) for b in batches],
            desc="RCA batches",
        )
        analyses = "\n\n".join(
            f"## Batch {i}\n<root_cause>\n{a['root_cause']}\n</root_cause>\n<action_policy>\n{a['action_policy']}\n</action_policy>"
            for i, a in enumerate(map(_parse_analysis, generations), start=1)
        )
        # Solution Generation gets the means over all outputs and the per-output results of the first batch
        # (the examples arrive shuffled, so it is a random batch)
        overall = _summary(judgments, rouges)
        first = batches[0]
        self._eval_results = (
            f"{overall}\nPer-output results for a random {len(judgments[first])} of the {len(judgments)} outputs:\n"
            + _per_output(judgments[first], rouges[first])
        )
        return _MERGE.format(prompt=prompt, eval_results=overall, analyses=analyses)

    def parse(self, generation: str):
        return {"eval_results": self._eval_results, **_parse_analysis(generation)}


def _eval_block(judgments, rouges) -> str:
    return _summary(judgments, rouges) + "\n" + _per_output(judgments, rouges)


def _top_bottom_block(judgments, rouges, n_top, n_bottom, output_score) -> str:
    order = sorted(range(len(judgments)), key=lambda i: -output_score(judgments[i]["score"] / 100, rouges[i]["rouge1"].fmeasure))
    top, bottom = order[:n_top], order[len(order) - n_bottom:]
    pick = lambda idx: ([judgments[i] for i in idx], [rouges[i] for i in idx])
    return (
        _summary(judgments, rouges)
        + f"\nPer-output results for the {n_top} highest-scoring outputs:\n" + _per_output(*pick(top))
        + f"\nPer-output results for the {n_bottom} lowest-scoring outputs:\n" + _per_output(*pick(bottom))
    )


def _summary(judgments, rouges) -> str:
    return (
        f"Quantitative (ROUGE against the reference summary, mean over {len(rouges)} outputs): "
        + ", ".join(f"{k} {statistics.mean(r[k].fmeasure for r in rouges):.3f}" for k in _ROUGE_TYPES)
        + f"\nQualitative (rubric judge, mean overall score {statistics.mean(j['score'] for j in judgments):.1f} / 100 over {len(judgments)} outputs):"
    )


def _per_output(judgments, rouges) -> str:
    lines = []
    for i, (j, r) in enumerate(zip(judgments, rouges), start=1):
        lines.append(f"[Output {i}] ROUGE-1 {r['rouge1'].fmeasure:.3f} | judge score {j['score']}")
        lines.extend(f"- {k}: {v}" for k, v in j["comments"].items())
    return "\n".join(lines)


def _parse_analysis(generation: str) -> dict:
    return {
        "root_cause": extract_xml_tag(generation, "root_cause") or generation,
        "action_policy": extract_xml_tag(generation, "action_policy") or _after_open_tag(generation, "action_policy"),
    }


def _after_open_tag(generation: str, tag: str) -> str:
    """gpt-oss often leaves the last tag unclosed; take everything after the opening tag (spec 10.9, bug 1)."""
    begin = generation.rfind(f"<{tag}>")
    return generation[begin + len(f"<{tag}>"):].strip() if begin != -1 else ""

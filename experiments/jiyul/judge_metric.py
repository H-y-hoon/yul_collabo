import re
from typing import Dict, List, Union, Any

from rouge_score.rouge_scorer import RougeScorer

from crispo.llms import LargeLanguageModel
from crispo.metrics.metric import Metric
from crispo.task.example import Example
from crispo.utilities.prompt_util import extract_xml_tag

_P = """
You are evaluating a summary of a dialogue. There is no reference summary; judge the summary against the dialogue only.

<dialogue>
{dialogue}
</dialogue>
<summary>
{summary}
</summary>

Evaluate the summary on each criterion below. For each criterion, write a short comment on how well the summary meets it.
{criteria}

Then give one overall quality score for the summary as an integer from 0 to 100.

Answer in exactly this format:
{format}
<score>integer from 0 to 100</score>
""".strip()


class RubricJudgeMetric(Metric):
    """
    Reference-free LLM judge (PoC "EvalOps", qualitative part): one call per output returns a comment per rubric
    criterion and one overall 0-100 rating. The gold summary is never shown to the judge. The quantitative part
    (ROUGE) is computed in the RCA prompt (spec 10.12).
    Judgments are kept in ``self.judgments[(x, pred)]`` so that the RCA prompt can read them.
    """

    def __init__(self, llm: LargeLanguageModel, rubric: Dict[str, str]) -> None:
        super().__init__()
        self.llm = llm
        self.rubric = rubric
        self.judgments = {}
        self.parse_failures = 0

    def fill(self, pred, x) -> str:
        return _P.format(
            dialogue=x,
            summary=pred,
            criteria="\n".join(f"- {k}: {v}" for k, v in self.rubric.items()),
            format="\n".join(f"<{k}>comment</{k}>" for k in self.rubric),
        )

    def parse(self, generation: str) -> dict:
        found = extract_xml_tag(generation, "score")
        number = re.search(r"\d+", found) if found else None
        if not number:
            self.parse_failures += 1
            return {"score": 0, "comments": {"unparsable judge output": generation}}
        return {
            "score": min(int(number.group()), 100),
            "comments": dict(
                (k, extract_xml_tag(generation, k) or "") for k in self.rubric
            ),
        }

    def score(
        self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None
    ) -> float:
        return self.batch_score([pred], [Example(x=x, y=gold)])[0]

    def output_score(self, judge_score: float, rouge1: float) -> float:
        """One output's selection score from its judge score (0-1) and ROUGE-1; used to rank outputs for the RCA."""
        return judge_score

    def batch_score(self, predictions: List, dataset: List[Example]) -> List[float]:
        generations = self.llm.batch_generate(
            [self.fill(p, e.x) for p, e in zip(predictions, dataset)], desc="Judging"
        )
        scores = []
        for p, e, generation in zip(predictions, dataset, generations):
            judgment = self.parse(generation)
            self.judgments[(e.x, p)] = judgment
            scores.append(judgment["score"] / 100)
        return scores


class JudgeRougeMetric(RubricJudgeMetric):
    """
    Selection signal with a larger ROUGE share (spec 10.17): per output, ``(1 - w) * judge + w * scale * ROUGE-1``.
    ``scale`` puts ROUGE-1 on the judge's range (initial prompt: judge 0.863, ROUGE-1 0.341 on train), so a
    relative change weighs the same in both. The judge still never sees the reference; ROUGE does, as before.
    """

    def __init__(self, llm: LargeLanguageModel, rubric: Dict[str, str], rouge_weight: float, rouge_scale: float) -> None:
        super().__init__(llm, rubric)
        self.rouge_weight, self.rouge_scale = rouge_weight, rouge_scale
        self._rouge = RougeScorer(["rouge1"], use_stemmer=True)

    def output_score(self, judge_score: float, rouge1: float) -> float:
        return (1 - self.rouge_weight) * judge_score + self.rouge_weight * self.rouge_scale * rouge1

    def batch_score(self, predictions: List, dataset: List[Example]) -> List[float]:
        judge_scores = super().batch_score(predictions, dataset)
        return [
            self.output_score(j, self._rouge.score(e.y, p)["rouge1"].fmeasure)
            for j, p, e in zip(judge_scores, predictions, dataset)
        ]

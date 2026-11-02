# Spec 7, check 1: template filling and parsing, without any server.
#   python -m experiments.jiyul.check_units
from crispo.task.example import Example
from experiments.jiyul import config as C
from experiments.jiyul.judge_metric import RubricJudgeMetric
from experiments.jiyul.rca_prompt import RCAPrompt
from experiments.jiyul.solgen_meta_prompt import SolGenMetaPrompt
from experiments.summarization.constants import ARTICLE_PLACEHOLDER
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder

X, PRED, GOLD = "DIALOGUE_TEXT_x1", "PREDICTED_SUMMARY_p1", "REFERENCE_SUMMARY_y1"


def in_order(text, parts):
    positions = [text.index(p) for p in parts]
    return positions == sorted(positions)


judge = RubricJudgeMetric(llm=None, rubric=C.RUBRIC)
judge_prompt = judge.fill(PRED, X)
assert X in judge_prompt and PRED in judge_prompt and GOLD not in judge_prompt
reply = "".join(f"<{k}>ok {k}</{k}>" for k in C.RUBRIC) + "<score>87</score>"
assert judge.parse(reply) == {"score": 87, "comments": dict((k, f"ok {k}") for k in C.RUBRIC)}
assert judge.parse("no tags")["score"] == 0 and judge.parse_failures == 1

judge.judgments[(X, PRED)] = judge.parse(reply)
rca = RCAPrompt(judge)
prompt = SummarizationTaskPromptWithPlaceholder(C.INITIAL_PROMPT)
rca_text = rca.fill(str(prompt), [PRED], [Example(x=X, y=GOLD)])
assert in_order(rca_text, ["# Current Prompt", "# Eval Results"])
assert X not in rca_text and PRED not in rca_text and GOLD not in rca_text, "RCA must see only prompt + eval"
assert "mean overall score 87.0 / 100" in rca_text and "ok clarity" in rca_text
assert "Quantitative (ROUGE against the reference summary, mean over 1 outputs): rouge1 0.333"  # shares "summary" in rca_text
critique = rca.parse("<root_cause>RC</root_cause><action_policy>AP</action_policy>")
assert critique["root_cause"] == "RC" and critique["action_policy"] == "AP"
assert rca.parse("<root_cause>RC</root_cause><action_policy>AP open")["action_policy"] == "AP open"


class _FakeLLM:  # answers every batch RCA call with a fixed analysis
    def batch_generate(self, prompts, desc=None):
        assert all(p.startswith("# Role\nYou are a failure root cause analysis agent") for p in prompts)
        return [f"<root_cause>RC{i}</root_cause><action_policy>AP{i}" for i, _ in enumerate(prompts, start=1)]


examples = [Example(x=f"{X}{i}", y=f"{GOLD}{i}") for i in range(3)]
preds = [f"{PRED}{i}" for i in range(3)]
for e, p in zip(examples, preds):
    judge.judgments[(e.x, p)] = judge.parse(reply)
batched = RCAPrompt(judge, _FakeLLM(), batch_size=2)
merge_text = batched.fill(str(prompt), preds, examples)
assert in_order(merge_text, ["# Current Prompt", "# Eval Results", "# Batch Analyses", "## Batch 1", "AP1", "## Batch 2", "AP2"])
assert "mean over 3 outputs" in merge_text and "[Output 1]" not in merge_text
assert not any(t in merge_text for t in [X, PRED, GOLD]), "merge must see only prompt + eval + analyses"
merged = batched.parse("<root_cause>M</root_cause><action_policy>MP</action_policy>")
assert merged["root_cause"] == "M" and "random 2 of the 3 outputs" in merged["eval_results"]
assert merged["eval_results"].count("[Output") == 2

solgen = SolGenMetaPrompt(guardrail="")
sg_text = solgen.fill([(prompt, 0.87)], None, critiques=[critique])
assert in_order(sg_text, ["# Current Prompt", "# Security Guideline", "# Eval Results", "# Root Cause Analysis & Action Policy"])
assert "(none)" in sg_text and X not in sg_text and PRED not in sg_text and GOLD not in sg_text
new = solgen.parse("<prompt>Summarize the dialogue briefly.</prompt>")
assert ARTICLE_PLACEHOLDER in new.prompt and "<summary>" in new.prompt
assert solgen.parse("nothing") is None
assert solgen.parse(f"<prompt>Read {ARTICLE_PLACEHOLDER}: {ARTICLE_PLACEHOLDER}</prompt>") is None
assert prompt.parse("<summary>S1</summary>") == "S1" and prompt.parse("<summary> S2") == "S2"
from experiments.jiyul.run import OproJudgeMetaPrompt
assert OproJudgeMetaPrompt().parse(f"<instruction>{ARTICLE_PLACEHOLDER} then {ARTICLE_PLACEHOLDER}</instruction>") is None
from experiments.jiyul.run import JudgeWithRougeFloor
train = [Example(x="d", y="amy calls bob today")]
sel = JudgeWithRougeFloor(train, floor=0.01)
cur, close, far, far2 = "CUR", "CLOSE", "FAR", "FAR2"
evaluated = {cur: ([], ["amy calls bob today"]), close: ([], ["Amy calls Bob today!"]), far: ([], ["something else"]),
             far2: ([], ["nothing here"])}
judge_scores = {cur: 0.80, close: 0.82, far: 0.95, far2: 0.90}
assert sel([cur], None, evaluated, judge_scores) == cur  # step 0: no floor yet
assert sel([close, far], cur, evaluated, judge_scores) == close, "far has the higher judge score but fails the floor"
assert sel([far, far2], cur, evaluated, judge_scores) == far and sel.history[-1]["fallback"]
keep = JudgeWithRougeFloor(train, floor=0.01, keep_current=True)
keep([cur], None, evaluated, judge_scores)
assert keep([far, far2], cur, evaluated, judge_scores) == cur and keep.history[-1]["kept_current"], "all fail the floor: keep"
assert keep([close], cur, evaluated, {**judge_scores, close: 0.70}) == cur, "a worse candidate does not replace"
assert keep([close], cur, evaluated, judge_scores) == close and not keep.history[-1]["kept_current"]
from experiments.jiyul.judge_metric import JudgeRougeMetric


class _FakeJudgeLLM:  # every output gets judge score 80
    def batch_generate(self, prompts, desc=None):
        return [reply.replace("87", "80") for _ in prompts]


jr = JudgeRougeMetric(_FakeJudgeLLM(), C.RUBRIC, rouge_weight=0.5, rouge_scale=2.5)
assert jr.batch_score(["amy calls bob", "zzz"], [Example(x="d1", y="amy calls bob"), Example(x="d2", y="amy calls bob")]) == [0.5 * 0.8 + 0.5 * 2.5, 0.4]
assert jr.judgments[("d2", "zzz")]["score"] == 80, "RCA still reads the plain judge score"
critique_in = {"eval_results": "E", "root_cause": "R", "action_policy": "A"}
plain = SolGenMetaPrompt().fill([(prompt, 0.5)], [], critiques=[critique_in])
short = SolGenMetaPrompt(concise=True).fill([(prompt, 0.5)], [], critiques=[critique_in])
assert "concise" not in plain and short.startswith(plain) and short.endswith("generally applicable to all dialogues.")
both = SolGenMetaPrompt(concise=True, line_limit=True).fill([(prompt, 0.5)], [], critiques=[critique_in])
assert both.startswith(short) and both.endswith("\n6. Keep the revised prompt under 5 lines unless multi-stage reasoning or examples are necessary.")
tb_examples = [Example(x=f"{X}tb{i}", y=f"{GOLD}tb{i}") for i in range(6)]
tb_preds = [f"{PRED}tb{i}" for i in range(6)]
for i, (e, p) in enumerate(zip(tb_examples, tb_preds)):
    judge.judgments[(e.x, p)] = {"score": 10 * (i + 1), "comments": {k: f"c{i}" for k in C.RUBRIC}}
tb = RCAPrompt(judge, None, batch_size=2, top_bottom=(3, 2))  # no llm: the top/bottom path must not batch
tb_text = tb.fill(str(prompt), tb_preds, tb_examples)
assert "mean over 6 outputs" in tb_text and "mean overall score 35.0" in tb_text
assert in_order(tb_text, ["3 highest-scoring", "judge score 60", "judge score 50", "judge score 40", "2 lowest-scoring", "judge score 20", "judge score 10"])
assert "judge score 30" not in tb_text and not any(t in tb_text for t in [X, PRED, GOLD])
assert tb.parse("<root_cause>R</root_cause><action_policy>A</action_policy>")["eval_results"].count("[Output") == 5
print("check_units: all passed")

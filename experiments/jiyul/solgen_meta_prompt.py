from dataclasses import dataclass
from typing import Optional, List, Tuple

from crispo.optimizer.meta_prompt import MetaPrompt
from crispo.task.example import Example
from crispo.task.prompt import TaskPrompt
from crispo.utilities.prompt_util import extract_xml_tag
from experiments.summarization.constants import ARTICLE_PLACEHOLDER
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder

_P = f"""
# Role
You are an LLM optimizer that revises the prompt of a dialogue summarization task. Use the eval results and the root cause analysis & action policy below to write an improved prompt.

# Current Prompt
<current_prompt>
{{prompt}}
</current_prompt>

# Security Guideline
{{guardrail}}

# Eval Results
{{eval_results}}

# Root Cause Analysis & Action Policy
<root_cause>
{{root_cause}}
</root_cause>
<action_policy>
{{action_policy}}
</action_policy>

# Instructions
1. Apply the action policy to fix the root causes, so that the revised prompt scores higher in the evaluation.
2. Keep every rule in the security guideline in the revised prompt.
3. Keep {ARTICLE_PLACEHOLDER} exactly once, where the dialogue goes (do not mention it anywhere else), and ask for the summary within <summary> tags.
4. Write the full revised prompt in <prompt> tags.
""".strip()


@dataclass
class SolGenMetaPrompt(MetaPrompt):
    """
    PoC "Solution Generation Meta-Prompt": current prompt, security guideline, eval results, root cause
    analysis & action policy. No trajectory of past prompts and no examples (run with
    num_task_prompts_in_meta_prompt=1 and num_few_shot_examples_in_meta_prompt=0). The Trainer samples it
    K times to get K candidates.
    """

    guardrail: str = ""
    concise: bool = False  # add OPRO/CriSPO's "concise" sentence (spec 10.18)
    line_limit: bool = False  # add ZERA's "under 5 lines" sentence (spec 10.19)

    def fill(
        self,
        prompt_score_pairs: List[Tuple[TaskPrompt, float]],
        few_shot_examples: List[Example],
        **kwargs,
    ) -> str:
        (prompt, _score), critique = prompt_score_pairs[0], kwargs["critiques"][0]
        text = _P.format(
            prompt=prompt.prompt,
            guardrail=self.guardrail or "(none)",
            eval_results=critique["eval_results"],
            root_cause=critique["root_cause"],
            action_policy=critique["action_policy"],
        )
        extra = []
        if self.concise:  # the OPRO/CriSPO sentence; "all examples above" -> "all dialogues", as no examples are shown
            extra.append("The revised prompt should be concise, effective, and generally applicable to all dialogues.")
        if self.line_limit:  # ZERA meta template: "Keep prompts under 5 lines unless multi-stage reasoning or examples are necessary"
            extra.append("Keep the revised prompt under 5 lines unless multi-stage reasoning or examples are necessary.")
        return text + "".join(f"\n{i}. {r}" for i, r in enumerate(extra, start=5))

    def parse(self, generation: str) -> Optional[TaskPrompt]:
        new_prompt = extract_xml_tag(generation, "prompt")
        if not new_prompt or new_prompt.count(ARTICLE_PLACEHOLDER) > 1:  # >1 would insert the dialogue twice
            return None
        if ARTICLE_PLACEHOLDER not in new_prompt:
            new_prompt = f"Given the input:\n\n{ARTICLE_PLACEHOLDER}\n\n" + new_prompt
        if "<summary>" not in new_prompt:
            new_prompt += " Write your summary within <summary> tags."
        return SummarizationTaskPromptWithPlaceholder(new_prompt)

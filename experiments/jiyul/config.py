# Settings for the Jiyul PoC reproduction. See ../../../구현 명세.md.
# Values marked [An] are assumptions listed in section 5 of the spec; everything else is stated in the
# PoC figure / call or fixed by the spec.
import os

from experiments.summarization.constants import ARTICLE_PLACEHOLDER

_HUB = os.path.join(
    os.environ.get("HF_HOME", "/home/work/ham/.cache/huggingface"), "hub"
)

# Models (spec 6.3), served by ../../../serve/start_servers.sh
META_MODEL = "gpt-oss-120b"  # RCA + Solution Generation (meta_llm) and the judge
META_URL = "http://localhost:8001/v1"
META_TOKENIZER = f"{_HUB}/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/tokenizer.json"
TASK_MODEL = "gpt-oss-20b"  # task_llm
TASK_URL = "http://localhost:8002/v1"
TASK_TOKENIZER = f"{_HUB}/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/tokenizer.json"
REASONING_EFFORT = "medium"  # [A14] meta_llm and judge
TASK_REASONING_EFFORT = "low"  # [A14] task_llm; "medium" made gpt-oss-20b run out of tokens on strict prompts (spec 10.4)
META_TEMPERATURE = 1.0  # [A8]
# --backend claude: one Claude model plays every role (task, meta, judge), as in CriSPO; the efforts above apply.
# Claude takes no temperature, so the temperatures above are used by the vllm backend only.
CLAUDE_MODEL = "claude-haiku-5-5"

# Loop
STOP_SCORE = 0.95  # figure: score >= 95 (judge 0-100 mapped to 0-1)
K = 3  # [A5] candidates per step
MAX_STEPS = 10  # [A6]
TRAIN_SIZE, DEV_SIZE, TEST_SIZE = 200, 100, 500  # [A2] 50/50/100 left score noise (~0.01) above the effects (spec 10.10)
NUM_EXAMPLES_IN_RCA = TRAIN_SIZE  # [A2] RCA reads the eval results of every train example (spec 10.13)
RCA_BATCH_SIZE = 50  # 200 results do not fit in 32k: RCA runs per batch of 50, then one call merges the analyses
ROUGE_FLOOR = 0.01  # selection drops candidates whose train ROUGE-1 is more than this below the current prompt's (spec 10.14); run-to-run noise is ~0.004
ROUGE_WEIGHT, ROUGE_SCALE = 0.5, 2.5  # --judge-rouge selection signal (spec 10.17): scale = initial prompt train judge 0.863 / ROUGE-1 0.341
OPRO_NUM_PROMPTS_IN_META_PROMPT = 20  # [A9] OPRO paper default trajectory length

# Content (service-specific in the PoC, research settings here)
GUARDRAIL = ""  # [A11] empty in v0
RUBRIC = {  # [A4] first two are stated in the figure, last three are assumed
    "clarity": "The summary is easy to understand on a single read.",
    "conciseness": "The summary contains no unnecessary words, repetition or filler.",
    "faithfulness": "Every statement in the summary is supported by the dialogue.",
    "coverage": "The summary includes the key information of the dialogue.",
    "coherence": "The summary is well-structured and reads as a connected whole.",
}
INITIAL_PROMPT = f"Here is a document:\n\n{ARTICLE_PLACEHOLDER}\n\nWrite a summary within <summary> tags for it."  # CriSPO SAMSum runner

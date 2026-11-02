# critic 기반 프롬프트 최적화(PoC) 재현 · OPRO · CriSPO 비교

critic 기반 black-box 자동 프롬프트 최적화(APO) PoC를 재현하고, 같은 조건에서 OPRO·CriSPO와 비교하는 코드.
[CriSPO 공식 코드](https://github.com/amazon-science/CriSPO)(AAAI 2025) 위에 PoC 루프를 얹음. OPRO와 CriSPO는 CriSPO 저장소의 구현을 그대로 사용함.

- 데이터: SAMSum(대화 요약). 시드로 train 200 / dev 100 / test 500을 뽑음.
- 최종 성적: test 500의 ROUGE-1(공식 지표).
- 모델: 기본은 Claude API의 `claude-haiku-5-5` 하나가 모든 역할(요약, 최적화, judge)을 맡음. CriSPO 논문처럼 한 모델로 돌림.

## PoC 루프

한 스텝의 순서:

1. 요약 모델이 현재 프롬프트로 train 200개를 요약함.
2. judge가 대화와 요약만 보고(정답 요약은 보지 않음) 기준 5개(명료성·간결성·사실 일치·포괄성·일관성)별 코멘트와 0–100점을 냄.
3. RCA(원인 분석)가 현재 프롬프트와 평가 결과를 보고 원인과 수정 방향을 씀.
4. Solution Generation이 이를 받아 새 프롬프트 후보 3개를 씀.
5. 후보 3개와 현재 프롬프트 중 train 점수가 가장 높은 것이 다음 현재 프롬프트가 됨.

5스텝마다 dev에서 채점하고, dev 점수가 가장 높은 프롬프트를 최종 프롬프트로 씀.

PoC 원 설계에서 디버깅하며 아래 다섯 가지를 기본 설정으로 추가함. 코드 주석의 `spec 10.x`는 공개하지 않은 내부 구현 기록의 절 번호임.

| 설정 | 내용 |
|---|---|
| 현재 프롬프트 유지 | 후보가 모두 현재보다 못하면 현재 프롬프트를 그대로 둠. |
| 점수 = judge + ROUGE | 후보 선택 점수는 `0.5 × judge + 0.5 × 2.5 × ROUGE-1`. 2.5는 두 점수의 크기를 맞추는 값(초기 프롬프트의 train judge 0.863 ÷ ROUGE-1 0.341). |
| "간결하게" | Solution Generation에 OPRO·CriSPO의 문장 "concise, effective, and generally applicable"을 넣음. |
| "5줄 이내" | Solution Generation에 ZERA의 문장 "under 5 lines"를 넣음. |
| RCA 입력 줄이기 | RCA에는 200개 전체의 평균 점수와 점수 상위 3개·하위 2개의 평가만 보여 줌(ZERA 방식). |

## 설치

Python 3.12 이상 필요(고정한 numpy·contourpy 버전이 3.12를 요구함).

```bash
git clone https://github.com/H-y-hoon/yul_collabo.git
cd yul_collabo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`requirements.txt`는 비교 실험을 돌린 환경의 버전으로 고정함. 특히 matplotlib는 3.10 이상에서 CriSPO의 그래프 코드가 오류를 내서 3.8.4로 고정함.

## API 키

```bash
cp .env.example .env
```

`.env`의 `ANTHROPIC_API_KEY=` 뒤에 키를 넣음. `.env`는 `.gitignore`에 들어 있어 커밋되지 않음. 환경변수 `ANTHROPIC_API_KEY`를 직접 export해도 됨.

워크스페이스에 묶이지 않은 조직 키를 쓰면 `This API key is not scoped to a workspace` 오류가 남. 이때는 Claude Console의 워크스페이스 설정에서 워크스페이스 ID(`wrkspc_...`)를 찾아 `.env`의 `ANTHROPIC_WORKSPACE_ID=` 뒤에 넣음. 워크스페이스에 묶인 키라면 비워 둠.

## 실행

저장소 최상위 폴더에서 실행함.

```bash
# 아주 작게(각 데이터 2개, 1스텝) 끝까지 도는지 확인
python -m experiments.jiyul.compare --method poc --debug --steps 1

# PoC 5스텝
python -m experiments.jiyul.compare --method poc --steps 5

# 같은 조건의 OPRO, CriSPO (기본 100스텝)
python -m experiments.jiyul.compare --method opro
python -m experiments.jiyul.compare --method crispo
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--method` | (필수) | `poc`, `opro`, `crispo` |
| `--steps` | 100 | 최적화 스텝 수(CriSPO의 SAMSum 설정) |
| `--seed` | 42 | train/dev/test를 뽑는 시드 |
| `--debug` | 꺼짐 | 각 데이터를 2개만 씀 |
| `--backend` | `claude` | `claude`는 Anthropic API, `vllm`은 직접 띄운 gpt-oss 서버(아래 참고) |

모델, 생각 강도, 후보 수, 데이터 크기 같은 값은 `experiments/jiyul/config.py`에서 바꿈.

- `CLAUDE_MODEL = "claude-haiku-5-5"`
- 생각 강도(effort): 요약은 `low`(`TASK_REASONING_EFFORT`), 최적화와 judge는 `medium`(`REASONING_EFFORT`)

### 결과

`data/experiments/jiyul/compare/{method}_{backend}[_{N}steps]_seed{seed}[_debug]/`에 저장됨.

- `summary.json`: 핵심 결과
  - `test_final.rouge.rouge1`: 최종 프롬프트의 test ROUGE-1
  - `test_initial`: 초기 프롬프트의 test 점수
  - `final_prompt`: 최종 프롬프트
  - `loop_cost`: 역할별 호출 수와 토큰 수. `completion_tokens`는 생각 포함, `content_tokens`는 답만.
- `prompts.json`, `prompts_dev.json`: 스텝별 프롬프트와 train/dev 점수
- `step-xxx/`: 스텝별 입력·출력·점수 csv
- `log.md`: 실행 로그

### 비용

Claude API는 토큰 단위로 과금됨(Haiku 5.5 정가: 입력 100만 토큰당 $0.10, 출력 $0.50).
- Claude의 생각 길이는 gpt-oss와 달라서 실제 비용은 꽤 다를 수 있음. 실제 사용량은 `summary.json`의 `loop_cost`로 확인.

### temperature

원래 설정은 요약과 judge를 temperature 0, 최적화를 temperature 1로 둠. Claude API는 temperature를 받지 않아서 `--backend claude`에서는 모든 역할이 모델 기본값으로 돎. 같은 프롬프트라도 실행마다 결과가 조금씩 달라질 수 있음. temperature 설정은 `--backend vllm`에서만 적용됨.

### vLLM(gpt-oss) 백엔드

아래 결과표는 이 백엔드로 낸 것. 요약은 gpt-oss-20b(`http://localhost:8002/v1`), 최적화와 judge는 gpt-oss-120b(`http://localhost:8001/v1`)가 맡음. 서버 주소와 토크나이저 경로는 `config.py`에 있음. GPU 서버가 필요해서 노트북에서는 `claude` 백엔드를 씀.

## 지금까지의 결과 (gpt-oss, 시드 42, 실행 1회)

| 방법 | 스텝 | test ROUGE-1 (초기 → 최종) |
|---|---|---|
| OPRO | 100 | 0.332 → 0.424 |
| CriSPO | 100 | 0.333 → 0.435 |
| PoC (위 다섯 설정 모두 적용) | 10 | 0.330 → 0.400 |

- 같은 초기 프롬프트를 다시 돌린 test ROUGE-1은 0.330–0.333 사이에서 흔들렸음.
- 시드 하나, 실행 한 번의 결과라 차이를 확정할 수 없음.
- gpt-oss 결과라서 Claude로 돌린 결과와 바로 비교할 수 없음.

## 파일 구성

| 경로 | 내용 |
|---|---|
| `experiments/jiyul/compare.py` | 실행 진입점(`--method poc\|opro\|crispo`) |
| `experiments/jiyul/config.py` | 모델, 생각 강도, 데이터 크기, 평가 기준 등 설정 |
| `experiments/jiyul/judge_metric.py` | judge 프롬프트와 점수(judge + ROUGE) |
| `experiments/jiyul/rca_prompt.py` | RCA 프롬프트 |
| `experiments/jiyul/solgen_meta_prompt.py` | Solution Generation 프롬프트 |
| `experiments/jiyul/check_units.py` | 서버 없이 프롬프트 채우기·파싱 점검(`python -m experiments.jiyul.check_units`) |
| `experiments/jiyul/run.py`, `noise.py`, `score_references.py` | 초기 실험·잡음 측정용 스크립트(vLLM 전용) |
| `crispo/llms/anthropic_llm.py` | Claude API 호출 |
| `crispo/llms/openai_compat.py` | vLLM(OpenAI 호환) 호출 |
| `crispo/trainer/trainer.py` | CriSPO 학습 루프. PoC용으로 현재 프롬프트 교체·선택 단계를 추가함 |

## 출처와 라이선스

- [amazon-science/CriSPO](https://github.com/amazon-science/CriSPO) 기반. 원본 커밋 기록이 그대로 남아 있어 바꾼 부분은 `git log`, `git diff`로 확인 가능.
- CriSPO 논문: He et al., "CriSPO: Multi-Aspect Critique-Suggestion-guided Automatic Prompt Optimization for Text Generation", AAAI 2025 ([arXiv:2410.02748](https://arxiv.org/abs/2410.02748)).
- 라이선스는 원본을 따라 **CC-BY-NC-4.0(비상업적 이용만 허용)**. 자세한 내용은 [LICENSE](LICENSE) 참고.

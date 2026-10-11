# SysDPO-Sampling with intermediate preference supervision

## File layout

| File | Responsibility |
|---|---|
| `train.py` | Main entry point, CLI, collection/cache workflow, pair export, logging, frozen-reference and gradient-accumulation helpers |
| `likelihood.py` | Teacher-forced scores, system/intermediate DPO losses, saved-score validation |
| `schemas.py` | Pydantic dataclasses for prompts, dataset rows, judgments, preferences and scoring/collection records |
| `model_api.py` | Stateless API calls and response validation |
| `prompts.json` | Editable A/B instructions, instruction policy and intermediate-judge rubric |
| `test/test_likelihood.py`, `test/test_smoke.py` | Numerical model fixtures and offline workflow checks |

These repeated interfaces use attributes (`row.question`, `judgment.choice`, `request.intermediates`). [Pydantic dataclasses](https://docs.pydantic.dev/latest/concepts/dataclasses/) validate the fields, and `TypeAdapter` reads/writes their JSON representation. Saved files and raw provider responses remain JSON objects. Existing cached dataset/call files are preserved; cached dataset rows are now validated too. Invalid text, unsupported classifier labels, unknown prompt/judgment fields and mismatched scoring-request hashes fail validation. Requests are copied and revalidated when scoring or evaluating saved scores, so later field or nested-list edits cannot silently reuse a stale hash. Duplicate candidates are recorded as a local discard reason without a judge call. Tensor score/loss dictionaries remain unchanged.

## Actual-data collection

```sh
uv run smoke/train.py --questions 10
```

One round reads the first ten `Intel/orca_dpo_pairs` training rows. Qwen `qwen/qwen-2.5-7b-instruct` generates three ordinary intermediates per question at temperature 0.3. DeepSeek `deepseek/deepseek-v3.2` compares `(y1,y2)`, `(y1,y3)`, then `(y2,y3)` in separate stateless requests with randomized presentation order. A preferred candidate points to the rejected candidate. `topo_sort.py` checks the three directed edges for cycles. An acyclic triple contributes all three intermediate preference pairs; a cyclic triple contributes none. Tie, duplicate or failed comparisons stop the remaining judge calls and discard the entire intermediate preference triple. Complete generations remain available for system scoring even when intermediate preferences are discarded. No retries are made. This path generates **no final answers** and makes **no final-answer judge calls**. Maximum collection workload: 30 generations + 30 judgments.

The API wrapper reads `OPENROUTER_KEY`. If it is exported only in `~/.zshrc`, use an interactive zsh terminal. Requests contain fresh system/user messages; keys are never saved. Outputs, usage, provider/model metadata and failures are cached. Duplicate intermediates skip judging. Both complete candidates remain available to system scoring regardless of the intermediate judgment. An intermediate-judge failure therefore does not discard the system-scoring request or the dataset's system preferences.

Edit `prompts.json` for A/B role instructions, instruction precedence and the intermediate rubric, or pass `--prompt-file PATH`. B scoring conditions on dataset instructions (part of x), the question, the intermediate draft, the instruction policy and b_instruction. Both current B and frozen B reference receive the same context. Like upstream SysDPO, a fixed synthesis instruction is included in conditioning; our wording and message layout differ from upstream. Prompt tokens are excluded from the scored loss but remain visible to the model. `--max-tokens` defaults to 8,192. Partial outputs remain saved, but are not judged or exported as usable pairs. There are no continuation calls. A finite bound and provider context limits can still prevent completion.

## Likelihood scoring and loss

Each ready question produces this scoring plan:

| Model | Context | Supplied completions |
|---|---|---|
| A and frozen A reference | Original A input | `y1`, `y2`, `y3` |
| B and frozen B reference | `(x,y1)` | Dataset `z_chosen`, `z_rejected` |
| B and frozen B reference | `(x,y2)` | Dataset `z_chosen`, `z_rejected` |
| B and frozen B reference | `(x,y3)` | Dataset `z_chosen`, `z_rejected` |

`likelihood.py` provides `completion_logp()` and `score_request()` for **actual PyTorch causal models and tokenizers**. Teacher forcing scores supplied tokens plus EOS, excluding the prompt. Input text is never truncated. The function returns differentiable current-model scores and reference scores under `no_grad`; callers must supply fixed initial references, appropriate model mode/device, and matching tokenizers. Weight loading and real optimizer updates are outside this API smoke.

`preference_losses()` computes:

```text
system_logp(z) = logsumexp_k(log A(yk|x) + log B(z|x,yk))
L_sys = mean(DPO on dataset z_chosen/z_rejected versus fixed references)
L_A_weighted = 0.5 * mean_i(DPO_A(y_chosen_i,y_rejected_i))
L_total = L_sys + L_A_weighted
```

`N` counts scored system questions; `M` counts usable intermediate pairs in this loss evaluation. Each denominator is independent. References are detached; lambda_A is fixed at 0.5. A receives both loss terms; B receives only `L_sys`. Exact duplicate intermediates are masked so they are not counted twice as distinct marginalization candidates. When `M=0`, the auxiliary value is zero for diagnostic loss evaluation; this does not constitute training the proposed auxiliary method.

The judge is a classifier returning only `{"choice":"option1"}`, `{"choice":"option2"}`, or `{"choice":"tie"}`. It supplies no confidence or rationale. All accepted pairs receive the same fixed **lambda_A = 0.5**. Both DPO betas default to 0.5.

The default scorer uses **mean completion token log probabilities**, matching SysDPO's released `get_probs`. `reduction="sum"` explicitly selects literal sequence log likelihoods. Token means are scoring surrogates rather than normalized sequence probabilities; the default auxiliary loss also uses those mean scores. Record the convention and reference snapshots when producing scores.

### Comparison with released SysDPO

The implementation follows the procedural helpers in [the pinned multi-LLM trainer](https://github.com/xwx84768/SysDPO/blob/9e775895e7c01a7cd15276661fbc819f7ffc8c02/sysdpo/multi_llm/train.py), commit `9e775895e7c01a7cd15276661fbc819f7ffc8c02`.

| Operation | Released implementation | This implementation |
|---|---|---|
| Score A's intermediate | Teacher-force generated tokens; shift logits/labels, mask prompt/padding, average token log scores (`get_probs`, lines 663–679 and 894–936) | Same shift/mask/mean in `get_probs`; `completion_logp` constructs the completion mask |
| A's contribution to system scores | Sum `exp(log_mid_probs) * final_probs` over candidates (lines 939–984) | Same candidate products and sum, evaluated stably with `logsumexp(a + b)` |
| Final preference loss | `-F.logsigmoid(beta * main_inside)`, then batch mean (lines 985–1010) | Same margin, loss and system batch mean; adds intermediate DPO with fixed weight 0.5 |
| Reference scoring | No gradients; independently generated reference candidates | No gradients; the same saved candidates are scored by both current and fixed reference models |

The tests compare finite score values and A/B gradients with the release's masked-mean and product-sum formulas. This verifies the score/loss path on supplied candidates, not full training reproduction. Deliberate differences remain: this smoke fixes the ordinary-sample pool instead of refreshing candidates with the released DBS procedure; it explicitly appends EOS to A and B completions; and it rejects nonfinite model logits before masking rather than applying the release's probability clamps and `nan_to_num`. The release does not explicitly append EOS to the dataset's final-answer text. These conventions can affect results and must accompany reported comparisons.

### Connecting scores

The OpenRouter generation endpoint used here does not teacher-force supplied responses. Without a connected scorer, terminal losses are reported as **pending**. Neither classifier labels nor generated-token log probabilities are substituted for those missing scores.

To score a saved request with already loaded models:

```python
from pathlib import Path
from pydantic import TypeAdapter
from schemas import ScoringRequest
from likelihood import score_request, preference_losses
requests = TypeAdapter(list[ScoringRequest]).validate_json(
    Path("smoke/runs/batch10_sampling/scoring_requests.json").read_text())
request = requests[0]
scores = score_request(request, A, B, A_ref, B_ref, tokenizer_A, tokenizer_B)
# A scores have shape [2]; B scores have shape [2,2], ordered candidate then zc/zr.
# Stack question scores into [N,3] and [N,3,2], then call preference_losses().
# Each auxiliary pair is (question_row, chosen_y_index, rejected_y_index).
```

Alternatively, evaluate externally computed scores:

```sh
uv run smoke/train.py --questions 10 --scores /absolute/path/to/scores.json
```

The score file is an object with `reduction` (`sum` or `mean`), `include_eos: true`, and `rows`. Each row contains `id`, the matching `request_sha256` from `scoring_requests.json`, and arrays `a`, `a_ref` of shape `[3]`, and `b`, `b_ref` of shape `[3,2]`. Supply **real completion log scores**, ordered `y1,y2,y3` and `zc,zr`; file values are not independently verified as model-produced. Every scoring-ready question must be covered once. Prompts/completions must match exactly. The file digest and scoring convention are saved with evaluated losses; this still performs zero optimizer steps.

## Outputs and terminal

New runs use `smoke/runs/batch10_sampling/` (workflow version 8). Earlier runs are preserved. A changed prompt/configuration needs a new directory and is rejected before existing records are overwritten.

- `run_config.json`: exact prompts, models, generation bound and workflow version.
- `dataset.json`: source rows, retrieval URL/time.
- `system_preferences.json`: unchanged original system pairs for all questions.
- `calls/*.json`: requests and raw responses, output text, finish reasons and token/cost usage.
- `question_*.json`: intermediates, exact A input, judgments, scoring plans and stage errors.
- `intermediate_preferences.json`: accepted `yc/yr`, system instructions and exact A messages.
- `scoring_requests.json`: all three intermediate contexts and fixed dataset final answers to score.
- `losses.json`: pending status or evaluated `L_sys`, `L_A_weighted`, `L_total`, `N`, `M`, and score provenance.
- `summary.json`: saved-run call/token accounting, stage failures and zero-update status.

If external score loading or validation fails, the run saves `loss_evaluation_failed` and the error in `losses.json` and `summary.json`, retains call/token accounting, and exits with failure. Old loss values are replaced with unavailable values.

Terminal reporting follows the procedural SysDPO pattern: running header, example/sample counts, per-question collection progress, separate system/auxiliary/total losses, chosen/rejected reference-relative log scores (`final_reward_w/l`), and token totals. The latter scores follow the release's unscaled logging convention; multiply by beta for the implicit DPO reward. Pending scores are explicitly marked. Resuming uses cached calls; usage describes all saved calls rather than only new requests in that invocation. Provider routing is not pinned.

## Verification

```sh
uv run smoke/test/test_smoke.py
uv run smoke/test/test_likelihood.py
uv run smoke/train.py --help
```

Offline tests cover dataclass validation and JSON/cache compatibility, the 10-question/60-call collection/cache, no final generation, all six B scoring combinations, context-dependent B scores, failed intermediate judgments, incomplete candidates, exact prompt preservation, prompt/configuration protection, request mutation and score-file binding, scoring-failure accounting, completion/EOS masking, masked nonfinite logits, fixed-reference gradient exclusion, system gradients to A/B, auxiliary gradients to A only, and fixed auxiliary weighting. Fixtures use mock API responses or tiny numerical models, not real Qwen likelihoods. `uv` installs the pinned Torch and Pydantic script dependencies. Test scripts add their parent directory to the import path and run directly from the project root without setting `PYTHONPATH`.

## Numerical gradient smoke check

The numerical models and training assertions live in `test_likelihood.py`. Production workflow modules contain no toy model definitions. The chosen/rejected log-ratio margin and `main_loss = -F.logsigmoid(beta * main_inside)` calculation are adapted from the pinned source above. Attribution and its license are preserved in `likelihood.py` and `LICENSE`.

The smoke-specific changes add fixed auxiliary weighting and independent system/intermediate batch normalization. The reference models remain frozen. Likelihoods stay in log space, so the finite-state test does not need the source's exp/log conversion or nonfinite-value replacement. This is a small adaptation of the loss path, not a copy of the full LLM training pipeline.

Run from the project root (CPU, no model downloads or paid API calls):

```sh
uv run smoke/test/test_likelihood.py
```

The numerical training fixture implements:

`mean(system_DPO over N rows) + 0.5 * mean(intermediate_DPO over M pairs)`

- Two separate trainable PyTorch models, A and B, with frozen initial references.
- Two ordinary samples from frozen A for each collection attempt; a **mock** judge orders them. Duplicate/tied candidates are skipped. The resulting pool stays fixed.
- Fixed synthetic final-answer preferences, independently batched from intermediate preferences. These are invented finite-state fixtures, not Orca data or research results.
- `N=128` system rows and `M=32` intermediate pairs per update. Microbatch size 2 means 64 system backward passes plus 16 auxiliary backward passes, followed by one optimizer step. Every contribution is divided by its own full batch count; a short final chunk is handled correctly.
- A receives both losses; B receives only the system loss. References receive no gradients; lambda_A is the fixed constant 0.5.

Assertions compare accumulated gradients with full-batch gradients (microbatches 1, 2 and 7), check initial DPO loss, gradient routing, unchanged references, parameter updates, and decreasing losses on this fixture.

The models are small categorical distributions over eight question IDs, four intermediate IDs and three final-answer IDs. System likelihood uses the **exact marginal over all four intermediates**, providing a simple mathematical check. It does not reproduce the released SysDPO candidate approximation or scoring heuristics. Adam's learning rate is chosen for the toy fixture, not copied from the paper.

Before V100 training, connect the scoring helpers to tokenized causal LMs, the pinned dataset, the selected system-candidate convention and cached API judgments. Still to validate on real models: chat-template/EOS boundaries, LoRA, FP16/scaling, reference-adapter isolation, checkpoint/resume and actual GPU memory/throughput. Passing this smoke check does not establish that the future LLM implementation is bug-free.

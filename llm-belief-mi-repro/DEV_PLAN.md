Unifying Raw Logging + Endpoint Logprobs + Greedy Logging

Goal
- Save raw provider responses for all providers (endpoint, router/Fireworks, local), both async and sync.
- Include the greedy T0 request in prompts.jsonl and raw/.
- Make endpoint T0 logprobs work per HF docs by placing flags under parameters.

Changes

1) Endpoint logprobs schema (HF Inference API)
- Files: llm_belief_mi_repro/llm_client.py
- Functions to edit:
  - HuggingFaceInferenceClient.chat_completion_with_logprobs
  - AsyncHuggingFaceInferenceClient.chat_completion (when logprobs=True)
- Request body for greedy call must be:
  - inputs: "<prompt>"
  - parameters:
    - temperature: max(1e-6, float(temperature))
    - max_new_tokens: int(max_tokens)
    - return_full_text: false
    - details: true
    - logprobs: 1
- Parsing:
  - Prefer details.tokens[*].logprob (sum over tokens)
  - Fallback: if tokens missing but details.transition_scores exists, sum transition_scores
- Keep temperature epsilon > 0 for endpoint.

2) Raw logging for all providers (chain steps + greedy)
- Files: llm_belief_mi_repro/llm_client.py, llm_belief_mi_repro/cli.py
- Add optional callback on_raw to all client chat methods (sync/async) for both OpenAICompatibleLLMClient and HuggingFaceInferenceClient variants:
  - chat_completion(..., on_raw: Optional[Callable[[Any], None]]=None)
  - chat_completion_with_logprobs(..., on_raw: Optional[Callable[[Any], None]]=None)
  - Ensure legacy call sites still compile (default None).
  - Call on_raw(raw_json) immediately after resp.json() for live responses.
  - When cache returns a hit (SQLiteCache.get), if hit.raw_response is present, call on_raw(hit.raw_response). If not present, synthesize a minimal raw (e.g., {"content": hit.content, "token_logprobs": hit.token_logprobs}).
- In cli.py, set up saver closures for:
  - Chain steps: write JSON files into outputs/logs/run_<id>/raw/<timestamp>_<id>.json
  - Greedy call: write to greedy_<timestamp>_<id>.json
- Wire on_raw saver for:
  - async chain (endpoint and router paths)
  - greedy async (router/endpoint)
  - greedy sync (router/local)

3) Greedy call logging in prompts.jsonl
- File: llm_belief_mi_repro/cli.py
- After successfully obtaining greedy token_logprobs:
  - Append a prompts.jsonl record with:
    - run_id, dataset, question
    - chain_step: "greedy"
    - messages: {"user": <prompt used for greedy>}
    - response_text: the greedy response
    - token_logprobs: list of floats (if not too large; otherwise store aggregate stats and rely on raw/ JSON)
    - latency_ms (if available)
- This should be done for all providers.

4) Auto-skip logic (unchanged behavior)
- Keep the current greedy_capable and verify_capable flags.
- Auto-skip only after a failed attempt returns no token logprobs or errors; do not skip if logprobs were returned.

5) Tests / sanity commands to add to DEV_GUIDE.md
- Endpoint quick run with --cache-mode off to populate raw/:
  - Verify raw/ has chain-step JSONs, prompts.jsonl has no greedy_logprob fields (if endpoint lacks logprobs), CSV has no greedy columns.
- Fireworks router quick run with --cache-mode off:
  - Verify raw/ has chain-step and greedy_*.json files; CSV contains greedy_logprob and greedy_logprob_avg.
- Local LM Studio quick run:
  - Verify raw/ greedy file exists if LM Studio supports logprobs; otherwise auto-skip T0.

Implementation notes
- SQLiteCache(CacheResult) already returns raw_response for some paths; if not, extend cache.put/get to store response_json so raw can be replayed on a cache hit.
- Be careful not to log PII/secrets; on_raw should write only the response payload, not headers or API keys.
- Keep temperature epsilon only in endpoint clients, not router/local.

Optional enhancements
- Add --raw-log flag to disable raw capture if desired.
- Add a purge subcommand for the cache to remove only certain models/providers.

That’s it—apply these edits and re-run the mini-batch checks at the end of DEV_GUIDE to confirm raw/ and greedy behavior across providers.
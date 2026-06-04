"""LLM-as-a-Judge (HuggingFace / OpenAI / Ollama) + BERTScore evaluation."""

from __future__ import annotations
import time
import random

import evaluate
from tqdm import tqdm

import config

# HF free tier: 2 s between calls keeps under the per-minute burst limit
_HF_SLEEP = 2.0
_HF_MAX_RETRIES = 4
_HF_BACKOFF_BASE = 5.0


JUDGE_PROMPT = """Grade the system's answer to the question below.

Question: {q}
Correct answer: {correct}
System answer: {answer}

Rules:
- Reply with ONLY the word PASS or FAIL, nothing else.
- PASS = the system answer is factually correct or semantically equivalent to the correct answer. Accept minor phrasing differences, equivalent titles or roles (e.g. Baron/Lord, Director/filmmaker), truncated but correct names, or slightly more specific versions of the correct answer.
- FAIL = the answer is factually wrong, completely off-topic, says it cannot be found, or directly contradicts the correct answer."""


class AccuracyEvaluator:
    def __init__(self):
        self._hf_client = None
        self._openai_client = None
        self._ollama_llm = None
        self._bertscore = None

    def _get_hf_client(self):
        if self._hf_client is None:
            from huggingface_hub import InferenceClient
            if not config.HF_TOKEN:
                raise ValueError("HF_TOKEN not set. Get one at https://huggingface.co/settings/tokens")
            self._hf_client = InferenceClient(model=config.JUDGE_MODEL, token=config.HF_TOKEN, timeout=30)
        return self._hf_client

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            if not config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set")
            self._openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        return self._openai_client

    def _get_ollama_llm(self):
        if self._ollama_llm is None:
            from langchain_ollama import ChatOllama
            self._ollama_llm = ChatOllama(model=config.OLLAMA_MODEL, temperature=0.0)
        return self._ollama_llm

    def _get_bertscore(self):
        if self._bertscore is None:
            print("[accuracy] Loading BERTScore model (one-time download)...")
            self._bertscore = evaluate.load("bertscore")
        return self._bertscore

    def judge_single(self, question: str, correct_answer: str, system_answer: str) -> bool | None:
        """Return True (PASS) / False (FAIL) / None (API error — no verdict).

        Callers should EXCLUDE None from the pass-rate denominator: the judge
        couldn't actually grade the answer, so it's neither a pass nor a fail.
        """
        prompt = JUDGE_PROMPT.format(q=question, correct=correct_answer, answer=system_answer)

        if config.JUDGE_PROVIDER == "ollama":
            try:
                from langchain_core.messages import HumanMessage
                verdict = self._get_ollama_llm().invoke([HumanMessage(content=prompt)]).content.strip().upper()
                return "PASS" in verdict
            except Exception as e:
                print(f"[accuracy] Ollama judge error: {e}. Marking as NO_VERDICT.")
                return None

        if config.JUDGE_PROVIDER == "openai":
            try:
                oai = self._get_openai_client()
                resp = oai.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0,
                )
                verdict = resp.choices[0].message.content.strip().upper()
                return "PASS" in verdict
            except Exception as e:
                print(f"[accuracy] OpenAI judge error: {e}. Marking as NO_VERDICT.")
                return None

        client = self._get_hf_client()
        for attempt in range(_HF_MAX_RETRIES):
            try:
                response = client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0,
                )
                return "PASS" in response.choices[0].message.content.strip().upper()
            except Exception as e:
                is_rate_limit = any(k in str(e).lower() for k in ("rate limit", "429", "too many", "payment", "credits", "quota"))
                if is_rate_limit and attempt < _HF_MAX_RETRIES - 1:
                    wait = _HF_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
                    print(f"\n[accuracy] HF rate limit. Retrying in {wait:.1f}s ({attempt + 1}/{_HF_MAX_RETRIES})...")
                    time.sleep(wait)
                else:
                    print(f"[accuracy] Judge error (attempt {attempt + 1}): {e}. Marking as NO_VERDICT.")
                    return None
        return None

    def judge_batch(
        self,
        questions: list[str],
        correct_answers: list[str],
        system_answers: list[str],
        sleep_between: float = _HF_SLEEP,
    ) -> list[bool]:
        results = []
        for q, correct, answer in tqdm(zip(questions, correct_answers, system_answers), total=len(questions), desc="LLM-as-a-Judge"):
            results.append(self.judge_single(q, correct, answer))
            if config.JUDGE_PROVIDER == "hf":
                time.sleep(sleep_between)
        return results

    def bertscore_batch(self, predictions: list[str], references: list[str]) -> dict[str, list[float]]:
        print("[accuracy] Computing BERTScore...")
        return self._get_bertscore().compute(
            predictions=predictions,
            references=references,
            lang="en",
            rescale_with_baseline=True,
        )

    def evaluate_pipeline(
        self,
        pipeline_name: str,
        questions: list[str],
        correct_answers: list[str],
        system_answers: list[str],
        skip_judge: bool = False,
    ) -> dict:
        print(f"\n[accuracy] Evaluating: {pipeline_name}")

        # judge_results: list of True (PASS) / False (FAIL) / None (no verdict).
        # None arises either because skip_judge=True (whole batch skipped) or because
        # an individual API call errored. We track these two cases separately so the
        # final pass rate is computed only over questions the judge actually scored.
        if not skip_judge:
            judge_results = self.judge_batch(questions, correct_answers, system_answers)
            batch_skipped = False
        else:
            print("[accuracy] Skipping LLM-as-a-Judge (skip_judge=True)")
            judge_results = [None] * len(questions)
            batch_skipped = True

        # Only items with a real PASS/FAIL verdict count toward the pass rate.
        # Items that errored (or were skipped) are EXCLUDED from the denominator,
        # not silently counted as FAIL.
        scored_judges = [r for r in judge_results if r is not None]
        n_scored      = len(scored_judges)
        n_errored     = 0 if batch_skipped else sum(1 for r in judge_results if r is None)
        n_skipped     = len(questions) if batch_skipped else 0
        pass_rate     = (sum(scored_judges) / n_scored) if n_scored else None
        # Strict pass rate (errors counted as FAIL) — kept for reference / comparison.
        pass_rate_strict = (sum(1 for r in judge_results if r is True) / len(judge_results)
                            if not batch_skipped else None)

        bert_results = self.bertscore_batch(system_answers, correct_answers)
        avg_bertscore_f1 = sum(bert_results["f1"]) / len(bert_results["f1"])

        judge_bonus = pass_rate is not None and pass_rate >= 0.90
        bert_bonus  = avg_bertscore_f1 >= 0.55

        result = {
            "pipeline": pipeline_name,
            "num_evaluated":            len(questions),
            "num_judge_scored":         n_scored,
            "num_judge_errored":        n_errored,
            "num_judge_skipped":        n_skipped,
            "llm_judge_pass_rate":      round(pass_rate, 4) if pass_rate is not None else None,
            "llm_judge_pass_rate_strict": round(pass_rate_strict, 4) if pass_rate_strict is not None else None,
            "bertscore_f1_rescaled":    round(avg_bertscore_f1, 4),
            "judge_bonus_achieved":     judge_bonus,
            "bert_bonus_achieved":      bert_bonus,
            "max_bonus_achieved":       judge_bonus and bert_bonus,
            "per_item": [
                {
                    "question":       q,
                    "correct_answer": c,
                    "system_answer":  a,
                    "judge_pass":     j,                   # True / False / None
                    "judge_error":   (not batch_skipped) and (j is None),
                    "bertscore_f1":   round(f, 4),
                }
                for q, c, a, j, f in zip(questions, correct_answers, system_answers, judge_results, bert_results["f1"])
            ],
        }

        print(f"[accuracy] {pipeline_name}:")
        if pass_rate is not None:
            print(f"  LLM-as-a-Judge pass rate: {pass_rate:.1%}  ({n_scored}/{len(questions)} scored, {n_errored} errored, excluded)")
            if n_errored:
                print(f"    (strict pass rate, errors=FAIL: {pass_rate_strict:.1%})")
        else:
            print("  LLM-as-a-Judge: no verdicts (all skipped or all errored)")
        print(f"  BERTScore F1 (rescaled):  {avg_bertscore_f1:.4f}")
        print(f"  Judge bonus (≥90%):       {'✓' if judge_bonus else '✗'}")
        print(f"  BERTScore bonus (≥0.55):  {'✓' if bert_bonus else '✗'}")

        return result

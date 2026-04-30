"""
alpha_loop.py
=============
Two-stage alpha evolution loop powered by MongoDB.

  Stage 1 — diagnose(ids)
    Fetch alpha docs from MongoDB by name/id → build metrics → call LLM diagnose_alpha
    Output: diagnosis JSON {diagnoses, cross_alpha_patterns}

  Stage 2 — evolve(ids, diagnosis)
    Feed original code + diagnosis → call LLM evolve_alpha
    Output: dict {variant_name: python_code_str}

Usage (Python):
    from alpha_loop import diagnose, evolve, run_loop

    # Pass alpha_name strings (exact match in gen_alpha collection)
    diagnosis, evolved = run_loop(["alpha_xxx_wf", "alpha_yyy_rank"])

Usage (CLI):
    python alpha_loop.py alpha_xxx_wf alpha_yyy_rank
    python alpha_loop.py alpha_xxx_wf --stage diagnose
    python alpha_loop.py alpha_xxx_wf --stage evolve --diagnosis diagnosis.json
    python alpha_loop.py alpha_xxx_wf --intent enhance 
"""

import os
import sys
import json
import re
import argparse
import textwrap
import yaml
from bson import ObjectId

# ── DB ─────────────────────────────────────────────────────────────────────────

from auto.utils import get_mongo_uri
import pymongo

PROMPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "promts.yaml")
DEFAULT_MODEL = "deepseek-v4-flash"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-d2920fe91a98497eadcaf6bdd63506c8")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

YEARS = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]

# ── LLM client (OpenAI-compatible DeepSeek) ───────────────────────────────────
from openai import OpenAI

# ── Singleton LLM client (reuse connection) ────────────────────────────────────
_LLM_CLIENT = None

def _get_client():
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _LLM_CLIENT


# ── Prompt loading (cached in memory) ──────────────────────────────────────────
_PROMPTS_CACHE = None

def _load_prompts():
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is None:
        with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
            _PROMPTS_CACHE = yaml.safe_load(f)
    return _PROMPTS_CACHE


def get_prompt(key: str) -> str:
    prompts = _load_prompts()
    if key not in prompts:
        raise KeyError(
            f"Prompt key '{key}' not found in {PROMPTS_PATH}. "
            f"Available: {list(prompts.keys())}"
        )
    return prompts[key]


# ── MongoDB helpers ────────────────────────────────────────────────────────────

def _get_collection():
    client = pymongo.MongoClient(get_mongo_uri())
    return client["alpha"]["gen_alpha"]


def _parse_scan(scan_dict: dict) -> dict:
    """Parse scan_result / scan_reverse_result → {period: (s0, s1, s2)}."""
    if not scan_dict:
        return {}
    parsed = {}
    for period, val in scan_dict.items():
        try:
            if isinstance(val, str):
                val = json.loads(val)
            parsed[period] = (float(val[0]), float(val[1]), float(val[2]))
        except Exception:
            pass
    return parsed


def _build_metrics(doc: dict) -> dict:
    """Convert a MongoDB gen_alpha document to the metrics dict needed by diagnose."""
    parsed_rev  = _parse_scan(doc.get("scan_reverse_result") or {})
    parsed_orig = _parse_scan(doc.get("scan_result") or {})
    best        = parsed_rev if parsed_rev else parsed_orig

    s0_all = best.get("all", (0, 0, 0))[0]
    s1_all = best.get("all", (0, 0, 0))[1]

    s0_by_year = {
        yr: round(best[yr][0], 1)
        for yr in YEARS
        if yr in best
    }

    # Extra context from statistics_strategy (strategy-level scan, optional)
    stats  = doc.get("statistics_strategy") or {}
    note_parts = []
    if isinstance(stats, dict):
        profit = stats.get("profit")
        sharpe = stats.get("sharpe")
        if profit is not None:
            note_parts.append(f"strategy_profit={profit}%")
        if sharpe is not None:
            note_parts.append(f"strategy_sharpe_gt1={sharpe}%")

    return {
        "s0_all":     round(s0_all, 2),
        "s1_all":     round(s1_all, 2),
        "s0_by_year": s0_by_year,
        "note":       ", ".join(note_parts) if note_parts else "",
    }


def fetch_alphas(ids: list[str]) -> list[dict]:
    """
    Fetch alpha documents from MongoDB by alpha_name (or ObjectId string).

    Args:
        ids: List of alpha_name strings or ObjectId hex strings.

    Returns:
        List of {name, code, metrics} dicts ready for diagnose().
    """
    coll = _get_collection()
    results = []

    for id_str in ids:
        # Try ObjectId first, fallback to alpha_name exact match
        doc = None
        if len(id_str) == 24 and all(c in "0123456789abcdefABCDEF" for c in id_str):
            try:
                doc = coll.find_one({"_id": ObjectId(id_str)})
            except Exception:
                pass

        if doc is None:
            doc = coll.find_one({"alpha_name": id_str})

        if doc is None:
            print(f"[fetch] WARNING: '{id_str}' not found in gen_alpha. Skipping.")
            continue

        code = doc.get("alpha_code", "") or ""
        if not code:
            print(f"[fetch] WARNING: '{id_str}' has no alpha_code. Skipping.")
            continue

        # Error handling: 1=Logic Error, 2=Timeout
        error_info = None
        error_type = doc.get("is_error")
        if error_type in [1, 2]:
            error_info = doc.get("error_detail", "Unknown evaluation error")

        results.append({
            "name":       doc.get("alpha_name", str(doc["_id"])),
            "code":       code,
            "metrics":    _build_metrics(doc),
            "error":      error_info,
            "error_type": error_type,
        })

    if not results:
        raise ValueError("No valid alphas fetched. Check the provided IDs/names.")

    err_count = sum(1 for a in results if a.get("error"))
    print(f"[fetch] Loaded {len(results)} alpha(s) from MongoDB (Errors: {err_count}).")
    return results


# ── LLM call ──────────────────────────────────────────────────────────────────

GLOBAL_TOKEN_USAGE = {
    "prompt_cache_hit_tokens": 0,
    "prompt_cache_miss_tokens": 0,
    "completion_tokens": 0,
}

def get_token_usage_and_cost():
    hits = GLOBAL_TOKEN_USAGE["prompt_cache_hit_tokens"]
    misses = GLOBAL_TOKEN_USAGE["prompt_cache_miss_tokens"]
    outputs = GLOBAL_TOKEN_USAGE["completion_tokens"]
    
    cost = (hits / 1_000_000) * 0.028 + (misses / 1_000_000) * 0.14 + (outputs / 1_000_000) * 0.28
    return {
        "tokens": {
            "prompt_cache_hit_tokens": hits,
            "prompt_cache_miss_tokens": misses,
            "completion_tokens": outputs,
            "total_tokens": hits + misses + outputs
        },
        "total_cost_usd": cost
    }


def _call_llm(system_prompt: str, user_message: str, model: str = DEFAULT_MODEL) -> str:
    """Call LLM with stable system_prompt for prefix cache hits.
    NOTE: Do NOT modify system_prompt here — keep it byte-stable across calls.
    All dynamic context (intent, feedback, prev_code) goes in user_message.
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        reasoning_effort="high",
        extra_body={
            "thinking":{"type": "enabled"}
        }
    )
    
    usage = response.usage
    if usage:
        hits = getattr(usage, "prompt_cache_hit_tokens", 0)
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            hits = getattr(usage.prompt_tokens_details, "cached_tokens", hits)
            
        misses = getattr(usage, "prompt_cache_miss_tokens", getattr(usage, "prompt_tokens", 0) - hits)
        outputs = getattr(usage, "completion_tokens", 0)
        
        GLOBAL_TOKEN_USAGE["prompt_cache_hit_tokens"] += hits
        GLOBAL_TOKEN_USAGE["prompt_cache_miss_tokens"] += misses
        GLOBAL_TOKEN_USAGE["completion_tokens"] += outputs

    return response.choices[0].message.content


def _extract_json(text: str) -> dict:
    """Robustly extract and parse JSON from LLM response (handles fences, trailing commas)."""
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    else:
        start = text.find("{")
        end   = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object found in LLM response:\n{text[:500]}")
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        fixed = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON from LLM response: {e}\n---\n{text[:800]}")


def _clean_code(code: str) -> str:
    """Remove redundant imports and clean up whitespace."""
    lines = code.splitlines()
    cleaned = []
    for line in lines:
        # Skip import lines
        if "import numpy" in line or "import pandas" in line:
            continue
        cleaned.append(line)
    
    # Rejoin and remove excessive empty lines
    result = "\n".join(cleaned)
    result = re.sub(r'\n\s*\n', '\n', result) # Collapse multiple newlines
    
    # Replace volume with matchingVolume
    result = result.replace("'volume'", "'matchingVolume'")
    result = result.replace('"volume"', '"matchingVolume"')
    
    return result.strip()


def fix_error(
    ids: list[str],
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
    _alphas: list[dict] | None = None,
) -> dict:
    """
    Stage: Fix technical errors (timeouts, crashes) in alphas.
    Returns a dict mapping alpha_name -> fixed_code.
    """
    alphas = _alphas if _alphas is not None else fetch_alphas(ids)
    broken = [a for a in alphas if a.get("error")]

    if not broken:
        if verbose:
            print("[fix_error] No alphas with errors found.")
        return {}

    # Group by error type
    logic_broken = [a for a in broken if a.get("error_type") == 1]
    timeout_broken = [a for a in broken if a.get("error_type") == 2]
    
    all_fixes = {}

    # Single system prompt for cache stability; error type context in user message
    system_prompt = get_prompt("fix_logic_alpha")

    # 1. Fix Logic/Syntax Errors
    if logic_broken:
        if verbose:
            print(f"[fix_error] Fixing {len(logic_broken)} logic error(s) with LLM ({model})...")
        user_payload = ("ERROR TYPE: LOGIC/SYNTAX — Fix crashes and runtime errors.\n\n"
                        + json.dumps({"alphas": logic_broken}, ensure_ascii=False, separators=(',',':')))
        raw = _call_llm(system_prompt, user_payload, model=model)
        fixes = _extract_json(raw)
        
        # Robustness: If only one alpha was sent and LLM returned a generic key (like "alpha_name"), 
        # map it back to the original alpha name.
        if len(logic_broken) == 1 and len(fixes) == 1:
            orig_name = logic_broken[0]["name"]
            fixes = {orig_name: list(fixes.values())[0]}
            
        all_fixes.update({name: _clean_code(code) for name, code in fixes.items()})

    # 2. Fix Timeout/Performance Errors (same system prompt for cache)
    if timeout_broken:
        if verbose:
            print(f"[fix_error] Fixing {len(timeout_broken)} timeout error(s) with LLM ({model})...")
        user_payload = ("ERROR TYPE: TIMEOUT/PERFORMANCE — VECTORIZE the code completely. "
                        "Replace all .apply() and loops with Pandas/Numpy vectorized ops. "
                        "Use rolling().cov()/var() for slopes, rolling().rank() for rankings.\n\n"
                        + json.dumps({"alphas": timeout_broken}, ensure_ascii=False, separators=(',',':')))
        raw = _call_llm(system_prompt, user_payload, model=model)
        fixes = _extract_json(raw)
        
        # Robustness: If only one alpha was sent and LLM returned a generic key, map it back.
        if len(timeout_broken) == 1 and len(fixes) == 1:
            orig_name = timeout_broken[0]["name"]
            fixes = {orig_name: list(fixes.values())[0]}
            
        all_fixes.update({name: _clean_code(code) for name, code in fixes.items()})

    if verbose:
        for name in all_fixes:
            print(f"  → Fixed: {name}")

    return all_fixes


# ── Stage 1: Diagnose ──────────────────────────────────────────────────────────

def diagnose(
    ids: list[str],
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
    intent: str = "flaw",
    _alphas: list[dict] | None = None,  # internal: skip fetch if already loaded
) -> tuple[dict, list[dict]]:
    """
    Stage 1: Diagnose alpha(s) fetched from MongoDB.
    Note: Alphas with errors are skipped here (use fix_error stage instead).
    """
    alphas = _alphas if _alphas is not None else fetch_alphas(ids)

    # Filter out broken alphas
    healthy = [a for a in alphas if not a.get("error")]
    broken  = [a for a in alphas if a.get("error")]

    if broken and verbose:
        print(f"[diagnose] WARNING: {len(broken)} alpha(s) have technical errors. "
              f"Skipping them in diagnosis. Use --stage fix to fix them.")

    if not healthy:
        if verbose:
            print("[diagnose] No healthy alphas to diagnose.")
        return {"diagnoses": [], "cross_alpha_patterns": "N/A"}, alphas

    # Always use same system prompt for cache stability; intent goes in user message
    system_prompt = get_prompt("diagnose_alpha")

    if verbose:
        mode_label = "Enhancing" if intent == "enhance" else "Diagnosing flaws in"
        print(f"[diagnose] {mode_label} {len(healthy)} healthy alpha(s) with LLM ({model})...")

    # Clean payload: remove null fields, compact JSON
    clean_alphas = [{"name": a["name"], "code": a["code"], "metrics": a["metrics"]}
                    for a in healthy]

    # Intent-specific instruction in user message (not system prompt)
    if intent == "enhance":
        intent_prefix = ("MODE: ENHANCEMENT — These alphas are already average/okay. "
                         "Focus on ADVANCED OPTIMIZATIONS (dynamic windowing, non-linear transforms, "
                         "volatility scaling, cross-feature orthogonal combinations) while preserving the core idea. "
                         "Use overall_verdict='AVERAGE_NEEDS_ENHANCEMENT'.\n\n")
    else:
        intent_prefix = "MODE: FLAW DETECTION — Identify structural flaws and critical issues.\n\n"

    user_payload = intent_prefix + json.dumps({"alphas": clean_alphas}, ensure_ascii=False, separators=(',',':'))

    raw = _call_llm(system_prompt, user_payload, model=model)

    if verbose:
        print(f"[diagnose] Response: {len(raw)} chars.")

    diagnosis = _extract_json(raw)

    if verbose:
        _print_diagnosis(diagnosis, intent=intent)

    return diagnosis, alphas


def _print_diagnosis(diagnosis: dict, intent: str = "flaw"):
    print("\n" + "=" * 60)
    if intent == "enhance":
        print("  STAGE 1 — ENHANCEMENT PLAN")
    else:
        print("  STAGE 1 — DIAGNOSIS")
    print("=" * 60)
    diagnoses = diagnosis.get("diagnoses", [])
    for i, d in enumerate(diagnoses, 1):
        print(f"\n  ({i}/{len(diagnoses)}) [{d.get('overall_verdict','?')}] {d.get('name','?')}")
        print(f"  Score gap : {d.get('score_gap','')}")
        print(f"  Root cause: {d.get('root_cause','')}")
        for iss in d.get("issues", []):
            print(f"    [{iss.get('severity','?')}] {iss.get('dimension','?')}: {iss.get('finding','')}")
            if iss.get("fix"):
                print(f"           → {iss['fix']}")
        plan = d.get("improvement_plan", {})
        if plan.get("suggested_logic"):
            print(f"  Suggested : {plan['suggested_logic']}")
    xp = diagnosis.get("cross_alpha_patterns", "")
    if xp:
        print(f"\n  Cross-pattern: {xp}")
    print("=" * 60)


# ── Stage 2: Evolve ────────────────────────────────────────────────────────────

def evolve(
    alphas: list[dict],
    diagnosis: dict,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> dict:
    """
    Stage 2: Generate 5 evolved alpha variants from diagnosis.

    Args:
        alphas:    List of {name, code, metrics} from fetch_alphas().
        diagnosis: Output dict from diagnose().
        model:     Gemini model name.
        verbose:   Print progress.

    Returns:
        dict mapping variant_name -> @staticmethod code string.
    """
    system_prompt = get_prompt("evolve_alpha")

    # Slim diagnosis: keep only actionable fields to reduce tokens
    slim_diag = {
        "diagnoses": [
            {
                "name": d["name"],
                "root_cause": d.get("root_cause", ""),
                "issues": [{"dimension": i["dimension"], "fix": i["fix"]}
                           for i in d.get("issues", [])],
                "improvement_plan": d.get("improvement_plan", {}),
            }
            for d in diagnosis.get("diagnoses", [])
        ]
    }

    user_payload = json.dumps(
        {
            "original_alphas": [
                {
                    "name": a["name"],
                    "code": a["code"],
                    "feedback_from_system": a.get("metrics", {}).get("note", "You MUST modify the core logic. DO NOT just return the exact same code.")
                } for a in alphas
            ],
            "diagnosis": slim_diag,
        },
        ensure_ascii=False,
        separators=(',',':'),
    )

    if verbose:
        print(f"\n[evolve] Sending {len(alphas)} alpha(s) + diagnosis to LLM ({model})...")

    raw = _call_llm(system_prompt, user_payload, model=model)

    if verbose:
        print(f"[evolve] Response: {len(raw)} chars.")

    new_alphas = _extract_json(raw)

    if verbose:
        _print_evolve(new_alphas)

    return new_alphas


def combine(
    ids: list[str],
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
    _alphas: list[dict] | None = None,
) -> dict:
    """
    Stage: Combine 2-5 alphas into a single super alpha.
    """
    alphas = _alphas if _alphas is not None else fetch_alphas(ids)
    if len(alphas) < 2:
        raise ValueError("Need at least 2 alphas to combine.")
        
    system_prompt = get_prompt("combine_alphas")

    user_payload = json.dumps(
        {"alphas": [{"name": a["name"], "code": a["code"]} for a in alphas]},
        ensure_ascii=False,
        separators=(',',':'),
    )

    if verbose:
        print(f"\n[combine] Combining {len(alphas)} alpha(s) with LLM ({model})...")

    raw = _call_llm(system_prompt, user_payload, model=model)

    if verbose:
        print(f"[combine] Response: {len(raw)} chars.")

    new_alphas = _extract_json(raw)
    
    # Clean the code
    combined = {name: _clean_code(code) for name, code in new_alphas.items()}

    if verbose:
        _print_evolve(combined)

    return combined


def _print_evolve(new_alphas: dict):
    print("\n" + "=" * 60)
    print("  STAGE 2 — EVOLVED VARIANTS")
    print("=" * 60)
    for name, code in new_alphas.items():
        sig = re.search(r"def\s+\w+\([^)]*\)", code)
        print(f"  {name}")
        if sig:
            print(f"    {sig.group(0)}")
    print("=" * 60)


# ── Full Loop ──────────────────────────────────────────────────────────────────

def run_loop(
    ids: list[str],
    model: str = DEFAULT_MODEL,
    output_dir: str = ".",
    save: bool = True,
    verbose: bool = True,
    intent: str = "flaw",
) -> tuple[dict, dict]:
    """
    Full pipeline: fetch → fix (if broken) → diagnose → evolve.
    """
    # 1. Fetch
    alphas = fetch_alphas(ids)

    # 2. Fix technical errors if any
    broken = [a for a in alphas if a.get("error")]
    if broken:
        if verbose:
            print(f"\n[run_loop] Found {len(broken)} broken alpha(s). Fixing technical errors first...")
        fixes = fix_error(ids, model=model, verbose=verbose, _alphas=alphas)
        
        # Update alpha code with fixed versions
        for a in alphas:
            if a['name'] in fixes:
                a['code'] = fixes[a['name']]
                a['error'] = None # Mark as fixed
                if verbose:
                    print(f"  → '{a['name']}' code updated with technical fix.")

    # 3. Diagnose (now all should be healthy)
    diagnosis, healthy_alphas = diagnose(ids, model=model, verbose=verbose, intent=intent, _alphas=alphas)

    if not diagnosis.get("diagnoses") and not broken:
        if verbose:
            print("[run_loop] No healthy alphas found. Pipeline stopped.")
        return diagnosis, {}

    if save:
        _save_json(diagnosis, os.path.join(output_dir, "diagnosis.json"), verbose)

    # 4. Evolve
    evolved = evolve(healthy_alphas, diagnosis, model=model, verbose=verbose)

    if save:
        _save_json(evolved, os.path.join(output_dir, "evolved_alphas.json"), verbose)

    return diagnosis, evolved


def _save_json(data: dict, path: str, verbose: bool):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[save] → {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Alpha Evolution Loop — fetch from MongoDB, diagnose, evolve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Full loop: diagnose + evolve
          python alpha_loop.py alpha_popbo_advance_v3_034_wf alpha_factor_miner_new_001_wf

          # Stage 1 only (save diagnosis.json)
          python alpha_loop.py alpha_xxx_rank --stage diagnose

          # Stage 2 from existing diagnosis
          python alpha_loop.py alpha_xxx_rank --stage evolve --diagnosis diagnosis.json

          # By ObjectId
          python alpha_loop.py 664f1a2b3c4d5e6f7a8b9c0d

          # Don't save files, just print JSON
          python alpha_loop.py alpha_xxx_wf --no-save
        """),
    )
    parser.add_argument("ids", nargs="+",
                        help="alpha_name(s) or ObjectId hex string(s) from gen_alpha collection")
    parser.add_argument("--stage", choices=["diagnose", "evolve", "fix", "combine", "all"], default="all",
                        help="Which stage to run (default: all)")
    parser.add_argument("--intent", choices=["flaw", "enhance"], default="flaw",
                        help="Diagnose intent: 'flaw' for finding issues, 'enhance' for optimizing average alphas (default: flaw)")
    parser.add_argument("--diagnosis", default=None,
                        help="Path to existing diagnosis JSON (used with --stage evolve)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Gemini model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save output files (default: current dir)")
    parser.add_argument("--no-save", action="store_true",
                        help="Print to stdout instead of saving files")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    save    = not args.no_save
    verbose = not args.quiet

    if args.stage == "fix":
        fixes = fix_error(args.ids, model=args.model, verbose=verbose)
        if save:
            _save_json(fixes, os.path.join(args.output_dir, "fixes.json"), verbose)
        else:
            print(json.dumps(fixes, indent=2, ensure_ascii=False))

    elif args.stage == "combine":
        combined = combine(args.ids, model=args.model, verbose=verbose)
        if save:
            _save_json(combined, os.path.join(args.output_dir, "combined_alphas.json"), verbose)
        else:
            print(json.dumps(combined, indent=2, ensure_ascii=False))

    elif args.stage == "diagnose":
        diag, _ = diagnose(args.ids, model=args.model, verbose=verbose, intent=args.intent)
        if save:
            _save_json(diag, os.path.join(args.output_dir, "diagnosis.json"), verbose)
        else:
            print(json.dumps(diag, indent=2, ensure_ascii=False))

    elif args.stage == "evolve":
        if not args.diagnosis:
            parser.error("--stage evolve requires --diagnosis <path>")
        with open(args.diagnosis, "r", encoding="utf-8") as f:
            diag = json.load(f)
        alphas = fetch_alphas(args.ids)
        result = evolve(alphas, diag, model=args.model, verbose=verbose)
        if save:
            _save_json(result, os.path.join(args.output_dir, "evolved_alphas.json"), verbose)
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))

    else:  # all
        diag, evolved = run_loop(
            args.ids,
            model=args.model,
            output_dir=args.output_dir,
            save=save,
            verbose=verbose,
            intent=args.intent,
        )
        if not save:
            print(json.dumps({"diagnosis": diag, "evolved": evolved}, indent=2, ensure_ascii=False))

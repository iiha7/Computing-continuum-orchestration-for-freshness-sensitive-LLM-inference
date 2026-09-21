
"""Simulate the offline profiling phase for smoke tests.

For paper results, profile a compact local edge LLM with src.profile_llms_real
(default qwen2.5:0.5b through Ollama) plus a larger local cloud LLM
(default qwen2.5:7b through Ollama). This simulator keeps the same JSONL schema
for quick debugging. Simulated confidence is a synthetic calibrated reliability
proxy, not model self-confidence. It intentionally makes the simulated edge LLM constrained
and the simulated cloud LLM stronger on medium/complex temporal queries so that
offloading is a meaningful decision.
"""
from __future__ import annotations
import argparse, math
from typing import Dict
import numpy as np
import pandas as pd
from .common import load_yaml, project_path, set_seed, write_jsonl, ensure_dir

CLASS_IDS = list(range(6))


def noisy_prediction(gt: int, acc: float, rng: np.random.Generator):
    if rng.random() < acc:
        return int(gt), True
    cand = [x for x in CLASS_IDS if x != int(gt)]
    weights = np.array([1/(1+abs(x-gt)) for x in cand], dtype=float)
    weights /= weights.sum()
    return int(rng.choice(cand, p=weights)), False


def response_text(mode: str, complexity: str, pred: int, conf: float) -> str:
    # Simulated outputs omit self-confidence. The separate `confidence` field in
    # the JSONL schema is a synthetic calibrated reliability proxy for smoke tests.
    if mode == 'edge':
        return '{"danger_class": %d}' % pred
    if complexity == 'simple':
        return '{"danger_class": %d, "rationale": "dominant weather and dryness indicators", "action": "monitor"}' % pred
    if complexity == 'medium':
        return '{"danger_class": %d, "rationale": "recent trends in dryness, humidity, wind, and rainfall", "action": "increase monitoring"}' % pred
    return ('{"danger_class": %d, "rationale": "temporal readings indicate changing fuel dryness, wind exposure, and rainfall effects", '
            '"risk_factors": "dryness, wind, humidity, rain", "action": "increase monitoring and escalate if trend persists"}' % pred)


def simulate_one(prompt: Dict, mode: str, rng: np.random.Generator, cfg: Dict) -> Dict:
    comp = prompt['complexity']
    gt = int(prompt['ground_truth_tier'])
    crit = float(prompt['criticality'])
    wlen = int(prompt.get('window_len', {'simple':1,'medium':3,'complex':6}[comp]))

    if mode == 'edge':
        # Constrained edge: very fast, lower accuracy, and degrades on longer
        # temporal/explanation queries.
        base_acc = {'simple':0.74, 'medium':0.64, 'complex':0.56}[comp]
        lat_m = {'simple':0.04, 'medium':0.06, 'complex':0.09}[comp]
        lat_s = 0.12
    elif mode == 'cloud':
        # Cloud: slower but better at temporal reasoning and complex requests.
        base_acc = {'simple':0.84, 'medium':0.91, 'complex':0.95}[comp]
        lat_m = {'simple':0.80, 'medium':1.25, 'complex':2.10}[comp]
        lat_s = {'simple':0.20, 'medium':0.35, 'complex':0.60}[comp]
    else:
        raise ValueError(mode)

    # High-risk and long-window cases are harder for edge and slightly benefit
    # cloud reasoning in the simulation.
    if mode == 'edge':
        acc = base_acc - 0.06*crit - 0.015*max(wlen-1, 0)
    else:
        acc = base_acc - 0.015*crit + 0.005*max(wlen-1, 0)
    acc = float(np.clip(acc + rng.normal(0,0.015), 0.25, 0.995))
    pred, correct = noisy_prediction(gt, acc, rng)
    conf = float(np.clip(rng.normal(0.82 if correct else 0.50, 0.10 if correct else 0.16), 0.03, 0.99))
    text = response_text(mode, comp, pred, conf)

    mult = float(rng.lognormal(mean=0.0, sigma={'simple':0.18,'medium':0.30,'complex':0.45}[comp]))
    response_bytes = int(max(10, len(text.encode()) * mult))
    latency_s = float(max(0.01, rng.lognormal(mean=math.log(lat_m), sigma=lat_s)))
    ttft_s = float(min(latency_s, max(0.005, rng.normal(0.25*latency_s, 0.04))))
    prompt_bytes = int(prompt.get(f'{mode}_prompt_bytes', prompt.get('prompt_bytes', 0)))
    input_tokens = max(1, prompt_bytes//4)
    output_tokens = max(1, response_bytes//4)
    price_in = cfg['cost']['price_input_per_token']
    price_out = cfg['cost']['price_output_per_token']
    cost = 0.0 if mode == 'edge' else input_tokens*price_in + output_tokens*price_out
    # simple uncertainty features for DRL state
    margin = float(np.clip(conf - (1-conf)/5, 0.0, 1.0))
    entropy = float(np.clip(-conf*np.log(conf+1e-12) - (1-conf)*np.log((1-conf+1e-12)/5), 0, 3))
    return {'prompt_id': prompt['prompt_id'], 'record_id': int(prompt['record_id']), 'complexity': comp,
            'mode_role': mode, 'model_name': {'edge':'sim_qwen2.5_0.5b_edge','cloud':'sim_qwen2.5_7b_cloud'}[mode],
            'provider': 'simulated', 'prompt_bytes': prompt_bytes, 'response_text': text,
            'predicted_tier': int(pred), 'ground_truth_tier': gt,
            'predicted_danger_class': int(pred), 'ground_truth_danger_class': gt,
            'correct': int(correct), 'response_quality': float(correct), 'escalation_correct': int(correct), 'risk_factor_f1': float(correct), 'action_correct': int(correct), 'urgency_correct': int(correct), 'verification_correct': int(correct), 'response_bytes': response_bytes,
            'input_tokens': input_tokens, 'output_tokens': output_tokens, 'latency_s': latency_s,
            'ttft_s': ttft_s, 'cost_usd': float(cost), 'confidence': conf,
            'calibrated_edge_confidence': conf if mode == 'edge' else np.nan,
            'confidence_source': 'simulated_calibrated_reliability' if mode == 'edge' else 'not_used',
            'response_valid': 1,
            'edge_margin': margin if mode == 'edge' else np.nan,
            'edge_entropy': entropy if mode == 'edge' else np.nan}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args(); set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = load_yaml(project_path('configs/experiment.yaml'))
    prompts = pd.read_csv(project_path('data/processed/prompts.csv'))
    outdir = ensure_dir(project_path('data/llm_profiles'))
    for mode, fn in [('edge','edge_outputs.jsonl'), ('cloud','cloud_outputs.jsonl')]:
        rows = [simulate_one(p.to_dict(), mode, rng, cfg) for _, p in prompts.iterrows()]
        write_jsonl(rows, outdir/fn)
        print(f"Wrote {outdir/fn} ({len(rows)} rows)")


if __name__ == '__main__':
    main()

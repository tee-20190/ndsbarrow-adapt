"""E0099 on the fixed English Stage 1 landscapes; historical inputs are read-only.

Prepare verifies/replays the saved ADAF/E0097 comparison without model inference.
Smoke runs one predetermined block. Run reuses that verified block and completes
the other 25, then scores all three detectors with the archived Stage 1 matcher.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / 'work/adaf_followup_2026-09-19/english'
ARCHIVE = ROOT / 'targets/lidar-archaeology/v3_pro/external/adaf_stage1_2026-09-20'
PACKAGE = ROOT / 'targets/lidar-archaeology/v3_pro/e0099_landscape_2026-09-22'
MANIFEST = PACKAGE / 'INPUTS.json'
OUTPUT = ROOT / 'outputs/e0099_landscape_2026-09-22'
CHECKPOINT_DIR = ROOT / 'targets/lidar-archaeology/v3_pro/results/E0099_fullcohort_convnext_seed2'
CHECKPOINT = CHECKPOINT_DIR / 'weights.pt'
EXPECTED_CHECKPOINT = '41e468923b78ada2e8a06ade83640f85cfb03e6ec6e01d6f71f82657c70ee858'
MODELS = ('ADAF', 'E0097', 'E0099')
PILOT = 'SCREEN_113_53'
DENSITIES = [i / 4 for i in range(1, 33)]
INFERENCE = dict(device='cuda', ensemble='mean', batch=64, max_void=0.2,
                 nms_m=60.0, grid_phase_px=[0, 0], tta=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    """Atomic new output only; never replace a completed receipt or candidate file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.rename(path)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def historical_scorer():
    path = ARCHIVE / 'scripts/score_staged.py'
    spec = importlib.util.spec_from_file_location('e0099_historical_scorer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Bind input locations only; matching, NMS, coverage and label logic are unchanged.
    module.HERE, module.ROOT = HISTORICAL, ROOT
    module.CONTRACT_PATH = HISTORICAL / 'CONTRACT.json'
    module.FREEZE_PATH = HISTORICAL / 'PREINFERENCE_RECEIPT.json'
    module.LABELS_PATH = ROOT / 'targets/lidar-archaeology/v3_pro/results/E0099_fullcohort_convnext_seed2_screen_eval/labels_SCREEN.geojson'
    return module


def baseline_inputs():
    stage = historical_scorer()
    contract, freeze = stage.read_freeze()
    for name, key in [('score_staged.py', 'scorer'), ('run_staged.py', 'runner'),
                      ('freeze_contract.py', 'freeze_script'), ('verify_stage_list.py', 'stage_list_verifier')]:
        if sha(ARCHIVE / 'scripts' / name) != freeze['frozen_code_hashes'][key]:
            raise ValueError(f'Archived source differs from historical freeze: {name}')
    blocks = contract['stage1']['blocks']
    if len(blocks) != 26 or len({b['id'] for b in blocks}) != 26:
        raise ValueError('Expected the fixed 26 Stage 1 blocks')
    stage.validate_inference_receipt(1, blocks, contract, freeze)
    proposals, labels, coverage, fingerprints = stage.load_inputs(blocks, contract)
    saved = read(ARCHIVE / 'STAGE1_RESULTS.json')
    if sha(ARCHIVE / 'STAGE1_RESULTS.json') != sha(HISTORICAL / 'scores/STAGE1_RESULTS.json'):
        raise ValueError('Historical results differ from the recorded baseline')
    replay = {}
    for name, group in [('primary_full_26', blocks),
                        ('pilot_excluded_sensitivity_25', [b for b in blocks if b['id'] != PILOT])]:
        actual = stage.score_subset(name, group, proposals, labels, coverage, densities=(2,), include_bootstrap=False)
        expected = saved[name]
        for key in ('block_ids', 'shared_valid_area_km2', 'eligible_common_parts', 'eligible_all_class_parts'):
            if actual[key] != expected[key]:
                raise ValueError(f'Historical replay mismatch: {name}/{key}')
        if actual['budgets'][0] != expected['budgets'][0]:
            raise ValueError(f'Historical candidate ranking/matching replay mismatch: {name}')
        replay[name] = dict(area_km2=actual['shared_valid_area_km2'],
                            common_parts=actual['eligible_common_parts'],
                            K=actual['budgets'][0]['shared_K'],
                            hits=actual['budgets'][0]['shared_equal_budget'])
    if replay['primary_full_26']['K'] != 1294 or replay['primary_full_26']['common_parts'] != 101:
        raise ValueError('Historical primary endpoint changed')
    return stage, blocks, proposals, labels, coverage, fingerprints, replay


def prepare():
    if MANIFEST.exists():
        raise FileExistsError(f'Inputs already prepared: {MANIFEST}')
    _, blocks, _, _, _, fingerprints, replay = baseline_inputs()
    if sha(CHECKPOINT) != EXPECTED_CHECKPOINT or read(CHECKPOINT_DIR / 'completion.json')['weights_sha256'] != EXPECTED_CHECKPOINT:
        raise ValueError('E0099 checkpoint does not match its saved completion')
    cfg = read(CHECKPOINT_DIR / 'config.json')
    if cfg['seed'] != 2 or cfg['cfg']['backbone'] != 'convnext_tiny' or cfg['cfg']['recipe'] != 'v2':
        raise ValueError('Unexpected E0099 configuration')
    paths = list((ROOT / 'verify/v3').glob('*.py'))
    paths += [p for p in ARCHIVE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [CHECKPOINT, CHECKPOINT_DIR / 'config.json', CHECKPOINT_DIR / 'completion.json']
    paths += [HISTORICAL / name for name in ('CONTRACT.json', 'PREINFERENCE_RECEIPT.json',
                                           'receipts/STAGE1_INFERENCE.json', 'scores/STAGE1_RESULTS.json')]
    for block in blocks:
        source = HISTORICAL / 'receipts/source_records' / f"{block['id']}.json"
        paths.append(source)
        record = read(source)
        paths += [ROOT / record['source_relative_path'], ROOT / record['valid_relative_path']]
        for model in ('ADAF', 'E0097'):
            paths.append(HISTORICAL / 'objects/proposals' / model / f"{block['id']}.json")
    paths.append(historical_scorer().LABELS_PATH)
    hashes = {relative(p): sha(p) for p in sorted(set(paths))}
    manifest = dict(format_version=1, prepared_utc=datetime.now(timezone.utc).isoformat(),
                    baseline_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    files=hashes, historical_fingerprints=fingerprints, baseline_replay=replay,
                    blocks=blocks, smoke_block=blocks[0]['id'], checkpoint_sha256=EXPECTED_CHECKPOINT,
                    inference=INFERENCE, density_kind='proposals_per_shared_native_valid_km2',
                    densities=DENSITIES, max_model_pipeline_seconds=7200, minimum_free_disk_bytes=8 * 1024**3,
                    scope='Fixed repeated-use SCREEN development blocks only; no training, VALIDATION or TEST.')
    write(MANIFEST, manifest)
    print(json.dumps(dict(status='PREPARED', manifest=relative(MANIFEST), replay=replay)), flush=True)


def verify_manifest():
    manifest = read(MANIFEST)
    if manifest['format_version'] != 1 or manifest['inference'] != INFERENCE or manifest['densities'] != DENSITIES:
        raise ValueError('Runner settings differ from prepared inputs')
    for name, expected in manifest['files'].items():
        path = (ROOT / name).resolve()
        path.relative_to(ROOT)
        if sha(path) != expected:
            raise ValueError(f'Prepared input changed: {name}')
    return manifest


def validate_candidate(record, block, manifest):
    bid = block['id']
    source = read(HISTORICAL / 'receipts/source_records' / f'{bid}.json')
    if record['status'] != 'INFERENCE_COMPLETE' or record['model'] != 'E0099' or record['block_id'] != bid:
        raise ValueError('Wrong candidate identity')
    if record['manifest_sha256'] != sha(MANIFEST) or record['checkpoint_sha256'] != manifest['checkpoint_sha256']:
        raise ValueError('Candidate has stale input/checkpoint identity')
    meta = record['metadata']
    if meta['input_sha256'] != source['source_sha256'] or meta['grid_phase_px'] != [0, 0] or meta['nms_m'] != 60.0:
        raise ValueError('Candidate terrain/grid/NMS mismatch')
    if meta['resolution_m'] != 1.5 or meta['window_m'] != 300 or meta['tta'] or meta['ensemble'] != 'mean':
        raise ValueError('Candidate preprocessing changed')
    objects = record['objects']
    if record['proposal_count'] != len(objects) or len({p['id'] for p in objects}) != len(objects):
        raise ValueError('Candidate count or IDs invalid')
    stage = historical_scorer()
    for p in objects:
        if (p['source_block'] != bid or not p['id'].startswith(f'E0099-{bid}-') or p['crs'] != 'EPSG:27700'
                or not math.isfinite(p['score']) or not 0 <= p['score'] <= 1
                or not stage.half_open(p['easting'], p['northing'], block['interior'])):
            raise ValueError(f'Invalid candidate in {bid}')


def paired_intervals(block_ids, labels, records):
    counts = np.array([[len(labels[bid])] + [sum(a['label_owner_block'] == bid for a in records[m]['common_assignments'])
                                           for m in MODELS] for bid in block_ids], dtype=np.int64)
    rng = np.random.default_rng(20260920)
    sums = counts[rng.integers(0, len(counts), size=(10000, len(counts)))].sum(axis=1)
    sums = sums[sums[:, 0] > 0]
    intervals = {}
    for name, index in [('E0099_minus_ADAF', 1), ('E0099_minus_E0097', 2)]:
        delta = (sums[:, 3] - sums[:, index]) / sums[:, 0]
        intervals[name] = dict(estimate=float((counts[:, 3].sum() - counts[:, index].sum()) / counts[:, 0].sum()),
                               percentile_90=np.quantile(delta, [0.05, 0.95], method='linear').tolist())
    return dict(draws=10000, seed=20260920, level=0.90, intervals=intervals,
                limitation='Conditional descriptive block bootstrap on reused SCREEN; global ranks, K and matches fixed.')


def score_subset(stage, blocks, proposals, labels_by_block, coverage, densities=DENSITIES):
    ids = [b['id'] for b in blocks]
    area = sum(coverage[bid]['shared_valid_area_km2'] for bid in ids)
    common_by_block = {bid: [p for p in labels_by_block[bid] if set(p['classes']) & stage.COMMON] for bid in ids}
    common = [p for rows in common_by_block.values() for p in rows]
    all_labels = [p for bid in ids for p in labels_by_block[bid]]
    pools = {m: stage.global_nms([p for bid in ids for p in proposals[m][bid]]) for m in MODELS}
    primary_requested = min(math.floor(2 * area), len(pools['ADAF']), len(pools['E0097']))
    max_k = max(primary_requested, math.floor(max(densities) * area))
    assignments = {m: dict(common=stage.match_ranked(pools[m][:max_k], common),
                           all=stage.match_ranked(pools[m][:max_k], all_labels)) for m in MODELS}

    def selected(model, k):
        actual = min(k, len(pools[model]))
        hits = [a for a in assignments[model]['common'] if a['proposal_rank'] <= actual]
        hits_all = [a for a in assignments[model]['all'] if a['proposal_rank'] <= actual]
        return dict(proposal_count=actual, actual_density_per_km2=actual / area,
                    hits_common=len(hits), hits_all=len(hits_all), common_assignments=hits,
                    all_assignments=hits_all, selected_ids=[p['id'] for p in pools[model][:actual]])

    primary_k = min(primary_requested, *(len(pools[m]) for m in MODELS))
    primary = {m: selected(m, primary_k) for m in MODELS}
    curve = []
    for density in densities:
        requested = math.floor(density * area)
        shared = min(requested, *(len(pools[m]) for m in MODELS))
        equal = {m: selected(m, shared) for m in MODELS}
        own = {m: selected(m, requested) for m in MODELS}
        for m in MODELS:
            own[m]['pool_exhausted'] = len(pools[m]) < requested
        # Full identities/assignments are retained at the headline; curve entries need only counts.
        compact = lambda rows: {m: {k: v for k, v in r.items() if k not in ('selected_ids', 'common_assignments', 'all_assignments')}
                                for m, r in rows.items()}
        curve.append(dict(target_density_per_km2=density, target_K=requested, shared_K=shared,
                          equal_budget=compact(equal), own_up_to_ceiling=compact(own)))
    per_block = []
    for bid in ids:
        per_block.append(dict(block_id=bid, assessable_common=len(common_by_block[bid]),
                              area_km2=coverage[bid]['shared_valid_area_km2'],
                              hits_common={m: sum(a['label_owner_block'] == bid for a in primary[m]['common_assignments']) for m in MODELS}))
    return dict(block_ids=ids, block_count=len(ids), area_km2=area, common_parts=len(common), all_parts=len(all_labels),
                available_after_global_nms={m: len(pools[m]) for m in MODELS},
                headline=dict(requested_K=primary_requested, shared_K=primary_k,
                              comparable_at_requested_K=primary_k == primary_requested, models=primary),
                curve=curve, per_block=per_block,
                paired_bootstrap_90=paired_intervals(ids, common_by_block, primary) if len(ids) > 1 and common else None)


def render_table(result):
    lines = ['# E0099 English landscape control', '',
             'Repeated-use SCREEN development evidence. Densities count proposals, not adjudicated false alarms.', '']
    for name in ('full_26', 'pilot_excluded_25'):
        rec = result[name]
        headline = rec['headline']
        lines += [f"## {name}", '', f"{rec['common_parts']} common-class parts; {rec['area_km2']:.3f} km²; headline K={headline['shared_K']}.", '',
                  '| Detector | Headline hits | Available candidates |', '|---|---:|---:|']
        for m in MODELS:
            lines.append(f"| {m} | {headline['models'][m]['hits_common']} | {rec['available_after_global_nms'][m]} |")
        lines += ['', '| Target / km² | Equal K | ADAF hits | E0097 hits | E0099 hits | Own counts A / 97 / 99 | Own hits A / 97 / 99 |',
                  '|---:|---:|---:|---:|---:|---|---|']
        for point in rec['curve']:
            equal, own = point['equal_budget'], point['own_up_to_ceiling']
            counts = ' / '.join(str(own[m]['proposal_count']) for m in MODELS)
            hits = ' / '.join(str(own[m]['hits_common']) for m in MODELS)
            lines.append(f"| {point['target_density_per_km2']:g} | {point['shared_K']} | {equal['ADAF']['hits_common']} | {equal['E0097']['hits_common']} | {equal['E0099']['hits_common']} | {counts} | {hits} |")
        lines += ['', 'Paired descriptive 90% intervals (recovery difference):', '']
        uncertainty = rec['paired_bootstrap_90']
        if uncertainty is None:
            lines.append('Not estimable for this subset.')
        for pair, interval in (uncertainty or {}).get('intervals', {}).items():
            low, high = interval['percentile_90']
            lines.append(f"- {pair}: {interval['estimate']:+.3f}, [{low:+.3f}, {high:+.3f}]")
        lines.append('')
    lines += ['ADAF supply exhaustion is explicit in RESULT.json. Own-budget rows above that supply are unequal-budget diagnostics.',
              'A score of 42/101 alone does not establish superiority to ADAF. No detector-selection verdict is automated.', '']
    return '\n'.join(lines)


def gpu_preflight(manifest):
    if shutil.disk_usage(ROOT).free < manifest['minimum_free_disk_bytes']:
        raise RuntimeError('Less than 8 GiB free disk')
    proc = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                          check=True, text=True, capture_output=True)
    if proc.stdout.strip():
        raise RuntimeError('GPU already has a compute process: ' + proc.stdout.strip())


def execute(mode):
    started = time.monotonic()
    manifest = verify_manifest()
    out = OUTPUT / ('smoke' if mode == 'smoke' else 'production')
    if out.exists():
        raise FileExistsError(f'Output already exists; inspect it rather than rerunning: {out}')
    stage, blocks, baseline, labels, coverage, fingerprints, replay = baseline_inputs()
    if fingerprints != manifest['historical_fingerprints'] or replay != manifest['baseline_replay']:
        raise ValueError('Baseline changed since prepare')
    smoke = None
    if mode == 'run':
        smoke = read(OUTPUT / 'smoke/COMPLETION.json')
        if smoke['status'] != 'SMOKE_COMPLETE' or smoke['manifest_sha256'] != sha(MANIFEST) or smoke['block_ids'] != [manifest['smoke_block']]:
            raise ValueError('A matching one-block smoke must pass first')
        for name, digest in smoke['outputs'].items():
            if sha(OUTPUT / 'smoke' / name) != digest:
                raise ValueError('Smoke output changed')
    gpu_preflight(manifest)
    import torch
    from .v3_infer_chip import detect_raster, load_weights
    if not torch.cuda.is_available():
        raise RuntimeError('Run in WSL v3venv with CUDA')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    active = float(smoke['model_pipeline_seconds']) if smoke else 0.0
    selected_blocks = [b for b in blocks if b['id'] == manifest['smoke_block']] if mode == 'smoke' else blocks
    out.mkdir(parents=True)
    write(out / 'STARTED.json', dict(mode=mode, manifest_sha256=sha(MANIFEST),
                                    started_utc=datetime.now(timezone.utc).isoformat(),
                                    torch=torch.__version__, python=sys.version, gpu=torch.cuda.get_device_name()))
    proposals = dict(baseline, E0099={})
    try:
        t0 = time.monotonic()
        model = load_weights(CHECKPOINT, torch.device('cuda'))
        active += time.monotonic() - t0
        if model[1] != read(CHECKPOINT_DIR / 'config.json')['cfg']:
            raise ValueError('Loaded checkpoint configuration differs from saved config')
        for block in selected_blocks:
            if active >= manifest['max_model_pipeline_seconds']:
                raise RuntimeError('Two-hour model-pipeline budget reached')
            if shutil.disk_usage(ROOT).free < manifest['minimum_free_disk_bytes']:
                raise RuntimeError('Disk free space fell below 8 GiB')
            bid = block['id']
            path = out / 'candidates' / f'{bid}.json'
            if smoke and bid == manifest['smoke_block']:
                record = read(OUTPUT / 'smoke/candidates' / f'{bid}.json')
            else:
                source = read(HISTORICAL / 'receipts/source_records' / f'{bid}.json')
                t0 = time.monotonic()
                rows, meta = detect_raster(ROOT / source['source_relative_path'], [model],
                                           interior=block['interior'], **dict(INFERENCE, grid_phase_px=(0, 0)))
                duration = time.monotonic() - t0
                active += duration
                objects = [dict(p, id=f'E0099-{bid}-{i:05d}', source_block=bid) for i, p in enumerate(rows)
                           if stage.half_open(p['easting'], p['northing'], block['interior'])]
                record = dict(status='INFERENCE_COMPLETE', model='E0099', block_id=bid,
                              manifest_sha256=sha(MANIFEST), checkpoint_sha256=manifest['checkpoint_sha256'],
                              proposal_count=len(objects), objects=objects, metadata=meta, inference_seconds=duration)
            validate_candidate(record, block, manifest)
            write(path, record)
            proposals['E0099'][bid] = record['objects']
            print(json.dumps(dict(block=bid, proposals=record['proposal_count'],
                                  seconds=record['inference_seconds'], cumulative_model_seconds=active)), flush=True)
            if active >= manifest['max_model_pipeline_seconds']:
                raise RuntimeError('Two-hour model-pipeline budget exceeded')
        del model
        torch.cuda.empty_cache()
        if mode == 'run':
            stage.require_complete({b['id'] for b in blocks}, set(proposals['E0099']))
            result = dict(status='COMPLETE', manifest_sha256=sha(MANIFEST), density_kind=manifest['density_kind'],
                          full_26=score_subset(stage, blocks, proposals, labels, coverage),
                          pilot_excluded_25=score_subset(stage, [b for b in blocks if b['id'] != PILOT], proposals, labels, coverage))
            # Saved baselines must retain their exact headline matches in the expanded comparison.
            if result['full_26']['headline']['comparable_at_requested_K']:
                for m in ('ADAF', 'E0097'):
                    if result['full_26']['headline']['models'][m]['hits_common'] != replay['primary_full_26']['hits'][m]['hits_common']:
                        raise ValueError(f'{m} headline changed')
            write(out / 'RESULT.json', result)
            (out / 'TABLE.md').write_text(render_table(result), encoding='utf-8', newline='\n')
        else:
            smoke_score = score_subset(stage, selected_blocks, proposals, labels, coverage)
            write(out / 'SMOKE.json', dict(status='PASS', block_id=manifest['smoke_block'],
                                          manifest_sha256=sha(MANIFEST), checks=['checkpoint', 'source hash', 'CRS', 'interior',
                                          'candidate IDs', 'scoring', 'baseline replay'], score=smoke_score))
        outputs = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
        write(out / 'COMPLETION.json', dict(status='SMOKE_COMPLETE' if mode == 'smoke' else 'COMPLETE',
                                           manifest_sha256=sha(MANIFEST), block_ids=[b['id'] for b in selected_blocks],
                                           model_pipeline_seconds=active, wall_seconds=time.monotonic() - started,
                                           reused_smoke=bool(smoke), outputs=outputs,
                                           no_training=True, no_validation_or_test_access=True))
    except Exception as exc:
        write(out / 'STOPPED.json', dict(status='STOPPED', error=f'{type(exc).__name__}: {exc}',
                                        model_pipeline_seconds=active, wall_seconds=time.monotonic() - started))
        raise
    print(json.dumps(dict(status='SMOKE_COMPLETE' if mode == 'smoke' else 'COMPLETE',
                          output=relative(out), model_pipeline_seconds=active)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'smoke', 'run'))
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else execute(args.mode)


if __name__ == '__main__':
    main()

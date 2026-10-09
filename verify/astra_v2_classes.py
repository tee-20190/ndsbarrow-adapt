"""A3 multilabel reporting and A5 exclusive per-epoch sampling quotas."""
import re
import numpy as np

CLASS_MAP = {
    'moated': r'\b(?:moated site|moat)\b',
    'motte_ringwork': r'\b(?:motte|ringwork|castle mound)\b',
    'hillfort': r'\b(?:hillfort|hill fort|camp)\b',
    'enclosure': r'\b(?:enclosure|henge)\b',
    'barrow': r'\b(?:barrow|bowl barrow|round barrow)\b',
}
PRIORITY = ('barrow', 'moated', 'enclosure', 'motte_ringwork', 'hillfort')
FLOOR_PERCENT = {'moated': 25, 'enclosure': 12, 'motte_ringwork': 8, 'hillfort': 8}

def stratum(name):
    tags = classes(name)
    return next((k for k in PRIORITY if k in tags), None)

def classes(name):
    return [k for k, pattern in CLASS_MAP.items() if re.search(pattern, name, re.I)]

def canonical_rows(rows):
    # Acquisition receipts remain immutable; derive the amended annotation.
    return [dict(r, **{'class': ';'.join(classes(r['name'])) if int(r['label']) else 'background'}) for r in rows]

def sampling_plan(rows):
    pools = {k: [] for k in (*PRIORITY, 'negative')}
    for i, r in enumerate(rows):
        tags = classes(r['name']) if int(r['label']) else []
        if int(r['label']) and not tags:
            raise ValueError('Positive has no canonical class: ' + r['id'])
        group = 'negative' if not int(r['label']) else stratum(r['name'])
        pools[group].append(i)
    positive = sum(len(pools[k]) for k in PRIORITY)
    if not positive or not pools['negative']:
        raise ValueError('Both positive and negative samples required')
    floors = {k: (p * positive + 99) // 100 for k, p in FLOOR_PERCENT.items()}
    if sum(floors.values()) > positive:
        raise ValueError('A5 integer floors exceed the positive epoch size')
    barrow = min(len(pools['barrow']), 40 * positive // 100, positive-sum(floors.values()))
    counts = dict(barrow=barrow, **floors, negative=len(pools['negative']))
    if any(counts[k] and not pools[k] for k in pools):
        raise ValueError('A5 required stratum is empty; do not silently relax floors')
    # Free remainder follows non-barrow pool sizes, with exact integer arithmetic.
    remainder = positive - barrow - sum(floors.values())
    total_other = sum(len(pools[k]) for k in FLOOR_PERCENT)
    extra = {k: remainder * len(pools[k]) // total_other for k in FLOOR_PERCENT}
    residual_order = sorted(FLOOR_PERCENT, key=lambda k: (-(remainder * len(pools[k]) % total_other), PRIORITY.index(k)))
    for k in residual_order[:remainder-sum(extra.values())]:
        extra[k] += 1
    for k in extra:
        counts[k] += extra[k]
    return pools, dict(
        amendment='A5',priority=list(PRIORITY),floor_percent=FLOOR_PERCENT,barrow_cap_percent=40,
        class_map=CLASS_MAP, multi_term_policy='Every matching class; totals overlap',
        strata_policy='One stratum per positive: barrow > moated > enclosure > motte_ringwork > hillfort',
        remainder_policy='Cap barrow at min(pool size, floor(40% N), N minus ceil floors); allocate remaining draws to non-barrow pools proportionally by largest remainder, priority breaks ties',
        method='Exact quotas; positive draws with replacement uniformly within stratum; negatives once each; shuffle all',
        counts=counts, pool_sizes={k:len(v) for k,v in pools.items()},
        expected_draws_per_item={k:counts[k]/len(v) if v else 0 for k,v in pools.items()},
        positive_samples=positive, barrow_fraction=barrow/positive,
        non_barrow_enclosure_type_fraction=(positive-barrow)/positive)

def epoch_indices(pools, plan, seed, epoch):
    rng = np.random.default_rng(seed + epoch)
    parts = [rng.choice(pools[k], size=plan['counts'][k], replace=True) for k in PRIORITY if plan['counts'][k]]
    parts.append(np.asarray(pools['negative'], dtype=int))
    result = np.concatenate(parts)
    rng.shuffle(result)
    return result.tolist()

def realised_counts(pools, indices):
    lookup = {i:k for k,values in pools.items() for i in values}
    counts = {k:0 for k in pools}
    for i in indices:
        counts[lookup[i]] += 1
    return counts

def validate_sampling(plan):
    if plan.get('amendment') != 'A5' or plan.get('priority') != list(PRIORITY) or plan.get('class_map') != CLASS_MAP:
        raise ValueError('Missing A5 sampling definition')
    n = plan['positive_samples']; counts = plan['counts']
    if n <= 0 or sum(counts[k] for k in PRIORITY) != n or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError('Invalid epoch counts')
    if 100 * counts['barrow'] > 40 * n or any(100 * counts[k] < p * n for k,p in FLOOR_PERCENT.items()):
        raise ValueError('A5 cap or floor violated')

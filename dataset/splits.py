"""Deterministic rainfall-event holdout shared by dataset constructors."""

import random


def rainfall_split_ids(scenario_ids, opt):
    ids = sorted(set(scenario_ids))
    if len(ids) < 2:
        raise ValueError("At least two rainfall events are required for train/validation.")
    requested = getattr(opt, "val_rain_list", None)
    if requested:
        validation = set(requested)
        if not validation.issubset(ids):
            raise ValueError("Validation rainfall IDs must exist and must not be test IDs.")
    else:
        fraction = getattr(opt, "val_fraction", 0.1)
        if not 0 < fraction < 1:
            raise ValueError("val_fraction must be between zero and one.")
        shuffled = ids.copy()
        random.Random(getattr(opt, "split_seed", 0)).shuffle(shuffled)
        count = max(1, min(len(ids) - 1, round(len(ids) * fraction)))
        validation = set(shuffled[:count])
    training = set(ids) - validation
    if not training or not validation:
        raise ValueError("Training and validation must both contain rainfall events.")
    return training, validation

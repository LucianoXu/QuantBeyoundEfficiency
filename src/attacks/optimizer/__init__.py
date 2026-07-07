from .gcg import GCGOptimizer


def build_optimizer(name: str, objective, disallowed_mask, config):
    if name != "gcg":
        raise ValueError(f"Unknown optimizer {name!r} (only 'gcg' is supported)")
    return GCGOptimizer(objective, disallowed_mask, config)

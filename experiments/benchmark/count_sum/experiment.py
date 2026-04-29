from collections.abc import Sequence

import numpy as np
import polars as pl


def _mk_dataset(n_entries: int, **attrs: Sequence) -> pl.DataFrame:
    if len(attrs) == 0: return pl.DataFrame()

    items = iter(attrs.items())
    attr, domain = next(items)
    domain_entry = pl.from_dict({attr: domain})  
    
    for attr, domain in items:
        domain_attr = pl.from_dict({attr: domain})
        domain_entry = domain_entry.join(domain_attr, how="cross")

    rng = np.random.default_rng()
    owners = rng.integers(0, n_entries, n_entries)
    owner_expr = pl.lit(owners).alias("owner_id")
    entry_expr = pl.row_index("entry_id").over("owner_id")
    
    return (
        domain_entry
        .sample(n_entries, with_replacement=True)
        .with_columns(owner_expr)
        .with_columns(entry_expr)
    )


class Experiment:
    def __init__(self, n_entries: int, n_cat: int, n_num: int):
        domain_cat = list(range(n_cat))
        domain_num = list(range(n_num))
        self.baseline = _mk_dataset(n_entries, cat=domain_cat, num=domain_num)

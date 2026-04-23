from collections.abc import Sequence

import numpy as np
import polars as pl
import scipy.sparse as sp

from qif_micro import mechanism, model
from qif_micro.qif.datatypes import ProbabDist


def _mk_dataset(n_entries: int, **attrs: Sequence) -> pl.DataFrame:
    if len(attrs) == 0: return pl.DataFrame()

    items = iter(attrs.items())
    attr, domain = next(items)
    domain_entry = pl.from_dict({attr: domain})    
    
    for attr, domain in items:
        domain_attr = pl.from_dict({attr: domain})
        domain_entry = domain_entry.join(domain_attr, how="cross")
        
    n_owners = n_entries # One entry per record
    owner_expr = pl.row_index("owner_id") % n_owners
    entry_expr = pl.row_index("entry_id").over("owner_id")
    
    return (
        domain_entry
        .sample(n_entries, with_replacement=True)
        .with_columns(owner_expr)
        .with_columns(entry_expr)
    )


class Experiment:
    def __init__(
        self, 
        n_entries: int,
        n_cat: int,
        n_num: int,
        **mechanisms: mechanism.Mechanism
    ):
        domain_num = list(range(n_num))
        domain_cat = list(range(n_cat))
        domains = {"num": domain_num, "cat": domain_cat}

        self._observed_dataset = _mk_dataset(n_entries, **domains)
        self._mechanisms = mechanisms

        def mk_pair(attr): return domains[attr], mechanisms[attr]
        mechanism_metadata = {attr: mk_pair(attr) for attr in mechanisms}

        self._reverse_mechanisms(mechanism_metadata)
        self._init_baseline() # Depends orig records
        self._init_subdomain_records() # Depends orig records
        self._init_observed_records() # Depends subdomain records

        domain_lens = list(map(len, domains.values()))
        n_records = np.prod(domain_lens)
        dist = np.repeat(1 / n_records, self.subdomain_records.height)
        self._pi = ProbabDist(dist)


    @property
    def pi(self): return self._pi
        
    @property
    def baseline(self): return self._baseline
        
    @property
    def mechanisms(self): return self._mechanisms

    @property
    def observed_dataset(self): return self._observed_dataset
        
    @property
    def observed_records(self): return self._observed_records

    @property
    def subdomain_records(self): return self._subdomain_records


    def _reverse_mechanisms(self, mechanism_metadata):
        map_owners_orig = (
            model._internal._mk_records(self.observed_dataset.lazy())  # ty:ignore[possibly-missing-submodule]
            .group_by("record").agg("owner_id")
            # We assume for these experiments that records have length one,
            # so we only generate the corresponding original singletons.
            .explode("record").unnest("record")
        )

        def get_domain(attr): return (
            self.observed_dataset.lazy()
            .select(attr).unique().sort(attr)
            .collect(engine="streaming")
            .to_series().to_list()
        )

        for attr, (input_domain, m) in mechanism_metadata.items():
            output_domain = get_domain(attr)

            # We need to filter, even for DP mechanisms, because
            # float-point precision might make the materialised channel
            # not DP (e.g., geometric noise which gives a very small prob)
            m_ch = m(input_domain=input_domain, output_domain=output_domain)
            m_ch_coo = sp.coo_array(m_ch.dist)
            p, rows, cols = m_ch_coo.data, *m_ch_coo.coords

            row_labels = (
                pl.LazyFrame({"row_label": input_domain})
                .sort("row_label").with_row_index("row")
            )
            
            col_labels = (
                pl.LazyFrame({"col_label": output_domain})
                .sort("col_label").with_row_index("col")
            )
            
            valid_mappings = (
                pl.LazyFrame({"row": rows, "col": cols, f"p_{attr}": p})
                .join(row_labels, on="row").drop("row")
                .join(col_labels, on="col").drop("col")
                .group_by("col_label").agg("row_label", f"p_{attr}")
            )

            map_owners_orig = valid_mappings.join(
                map_owners_orig, 
                left_on="col_label", 
                right_on=attr
            ).drop("col_label").rename({"row_label": attr})

        # At this point, we have for each observed record:
        # - The non-sanitised attrs
        # - For each sanitised attr, all compatible original values
        # - And the list of owners
        # 
        # Now, there may be duplicate rows excluding owners, which
        # means that some owners with different observed values have
        # the same compatible original records.
        #
        # We want to group them, so we get the subdomain of records
        # along with the compatible owners:
        map_owners_orig = (
            map_owners_orig.with_row_index("map_id")
            # Collect to cache, so we don't lose map ids
            .collect(engine="streaming").lazy()  # ty:ignore[unresolved-attribute]
        )

        self._map_owners = (
            map_owners_orig.select("map_id", "owner_id").explode("owner_id")
        )

        sanitised_attrs = list(self.mechanisms.keys())
        attr = sanitised_attrs[0]

        # If there are multiple sanitised attributes, then we might need
        # to filter some records that will have probability 0.
        # These are not really 0, they become 0 due to float errors.
        map_records = (
            map_owners_orig.drop("owner_id").rename({f"p_{attr}": "p"})
            .explode(attr, "p")
        )
        
        for attr in sanitised_attrs[1:]: map_records = (
            map_records.explode(attr, f"p_{attr}")
            .with_columns(pl.col("p").mul(f"p_{attr}").alias("p"))
            .drop(f"p_{attr}")
            # Just to be sure, we filter p > 0.0000000001,
            # which is already a lot of noise,
            # but good enough precision to not need to restart.
            .filter(pl.col("p") > 0.0000000001)
        )
        
        map_records = (
            map_records.drop("p")
            .group_by(pl.exclude("map_id")).agg("map_id")
        )
        
        self._map_records = map_records.with_row_index("record_id")


    def _init_baseline(self):
        # We first get the number of values we need to sample
        # from each subset of records:
        sample_len = (
            self._map_owners
            .group_by("map_id").agg(pl.len())
            .sort("map_id")
        )

        # Then sample from the subdomain of records, which the
        # adversary sees as possible records:
        sample_expr = pl.exclude("map_id", "len").list.sample(
            "len", with_replacement=True
        )
        
        sample_records = (
            self._map_records.drop("record_id").explode("map_id")
            .group_by("map_id").agg(pl.all())
            .join(sample_len, on="map_id")
            .with_columns(sample_expr).drop("len")
            .explode(pl.exclude("map_id"))  # ty:ignore[invalid-argument-type]
            .sort("map_id").drop("map_id")
        )

        # Now combine the two things to get a dataset:
        owners = self._map_owners.sort("map_id").select("owner_id")
        baseline = pl.concat([owners, sample_records], how="horizontal")
        self._baseline = baseline.collect(engine="streaming")

    
    def _init_observed_records(self):
        columns = self.observed_dataset.collect_schema().names()
        attrs = [c for c in columns if c != "owner_id"]

        observed_records = (
            self.observed_dataset.lazy()
            .drop("owner_id").unique()
        )

        self._observed_records = (
            self.subdomain_records.lazy()
            .join(observed_records, on=attrs)
            .collect(engine="streaming")
        )
        
        
    def _init_subdomain_records(self):
        self._subdomain_records = (
            self._map_records.drop("map_id").collect(engine="streaming")
        )

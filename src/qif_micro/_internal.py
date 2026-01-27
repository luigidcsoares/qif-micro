from collections.abc import Iterable

import polars as pl

def _prepare_records(lf: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    # ==================================================
    # Pre-conditions: record column must either be
    #   in "long" format with rows tagged with a record_id,
    #   or must have a single record column.
    # ==================================================
    lf = lf.lazy()
    
    diff_record, ok_record = _valid_columns(lf, ["record"])
    diff_id, ok_id = _valid_columns(lf, ["record_id"])
    if not (ok_record or ok_id):
        msg_record = "Dataframe must either have a single `record` column"
        msg_id = "have a column `record_id` tagged to every entry"
        raise ValueError(f"{msg_record} or {msg_id}")

    schema = lf.collect_schema()
    if ok_record:
        ok_record_type = schema["record"] == pl.List
        if not ok_record_type:
            raise ValueError("`record` dtype must be list")

        ok_record_inner = schema["record"].inner == pl.Struct
        if not ok_record_inner:
            raise ValueError("`record` inner dtype must be struct")

    else: # Transform dataframe into long form
        record_attrs = [c for c in schema.keys() if c != "record_id"]
        record_expr = pl.struct(record_attrs).alias("record")
        lf = lf.select("record_id", record_expr)
        lf = lf.group_by("record_id").agg(pl.col("record")).drop("record_id")

    return lf


def _prepare_hints(lf: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.lazy()
    
    diff_hint, ok_hint = _valid_columns(lf, ["hint"])
    diff_id, ok_id = _valid_columns(lf, ["hint_id"])
    if not (ok_hint or ok_id):
        msg_hint = "Dataframe must either have a single `hint` column"
        msg_id = "have a column `hint_id` tagged to every entry"
        raise ValueError(f"{msg_hint} or {msg_id}")

    schema = lf.collect_schema()
    if ok_hint:
        # TODO: allow longitudinal hints
        # ok_hint_type = schema["hint"] == pl.List
        # if not ok_hint_type:
        #     raise ValueError("`hint` dtype must be list")

        ok_hint_type = schema["hint"] == pl.Struct
        if not ok_hint_type:
            raise ValueError("`hint` dtype must be struct")

    else: # Transform dataframe into long form
        hint_attrs = [c for c in schema.keys() if c != "hint_id"]
        hint_expr = pl.struct(hint_attrs).alias("hint")
        lf = lf.select( hint_expr)

    return lf
    


def _valid_columns(
    lf: pl.LazyFrame,
    required: Iterable[str]
) -> tuple[set[str], bool]:
    missing = set(required) - set(lf.collect_schema().names())
    return missing, len(missing) == 0

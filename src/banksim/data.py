"""Validated events, exact money and independent raw/normal logical views."""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

FIELDS = (
    "step", "customer", "age", "gender", "zipcodeOri", "merchant",
    "zipMerchant", "category", "amount", "fraud",
)
CATEGORICAL = FIELDS[1:8]
CATEGORIES = frozenset(
    "es_barsandrestaurants es_contents es_fashion es_food es_health es_home "
    "es_hotelservices es_hyper es_leisure es_otherservices es_sportsandtoys "
    "es_tech es_transportation es_travel es_wellnessandbeauty".split()
)
INT64_MAX = 2**63 - 1


class DataError(ValueError):
    """A blocking error with machine-readable code and source evidence."""

    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code = code
        self.details = details

    def as_dict(self) -> dict:
        return {"code": self.code, "message": str(self), **self.details}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_category(value: object) -> str:
    if not isinstance(value, str):
        raise DataError("INVALID_VALUE", "Categorical values must be strings")
    result = value.strip()
    if len(result) >= 2 and result.startswith("'") and result.endswith("'"):
        result = result[1:-1]
    return sys.intern(result)


def parse_money(value: object) -> tuple[Decimal, int]:
    """Never round precision or pass money through binary floating point."""
    if isinstance(value, (float, bool)) or not isinstance(value, (str, int, Decimal)):
        raise DataError("INVALID_VALUE", "Money must be decimal text, Decimal or integer")
    text = str(value).strip()
    try:
        amount = Decimal(text)
    except InvalidOperation as exc:
        raise DataError("INVALID_VALUE", "Invalid decimal amount", value=text) from exc
    if not amount.is_finite() or amount < 0:
        raise DataError("INVALID_VALUE", "Amount must be finite and nonnegative", value=text)
    if amount.as_tuple().exponent < -2:
        raise DataError("INVALID_VALUE", "Amount has more than two decimal places", value=text)
    if amount > Decimal(INT64_MAX) / 100:
        raise DataError("INVALID_VALUE", "Amount exceeds int64 cents storage", value=text)
    cents = int(amount * 100)  # Decimal arithmetic; precision already validated.
    return amount.quantize(Decimal("0.01")), cents


def parse_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise DataError("INVALID_VALUE", "Expected integer text", field=field)
    text = str(value).strip()
    if re.fullmatch(r"[+-]?[0-9]+", text) is None:
        raise DataError("INVALID_VALUE", "Expected integer text", field=field, value=text)
    return int(text)


def validate_coverage(coverage: tuple[int, int]) -> tuple[int, int]:
    start, end = coverage
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (start, end)):
        raise DataError("INVALID_VALUE", "Coverage must use integer simulation steps")
    if not 0 <= start <= end <= 179:
        raise DataError("INVALID_VALUE", "Coverage must be inside steps 0..179")
    return start, end


@dataclass(frozen=True)
class EventViews:
    """Public access returns defensive copies, so a caller cannot mutate the source."""

    _events: pd.DataFrame
    diagnostics: dict

    @property
    def raw_events(self) -> pd.DataFrame:
        return self._events.copy(deep=True)

    @property
    def normal_events(self) -> pd.DataFrame:
        return self._events.loc[self._events["fraud"].eq(0)].copy(deep=True)


def attribute_conflicts(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    counts = frame.groupby("customer")[["age", "gender"]].nunique()
    bad = counts.index[counts.gt(1).any(axis=1)]
    conflicts = []
    for customer, rows in frame.loc[frame["customer"].isin(bad)].groupby("customer", sort=True):
        fields = [field for field in ("age", "gender") if rows[field].nunique() > 1]
        conflicts.append({
            "code": "ATTRIBUTE_CONFLICT", "customer": customer, "fields": fields,
            "sources": rows[["raw_row_id", "step", "age", "gender"]].to_dict("records"),
        })
    return conflicts


def validate_records(records: Iterable[Mapping], coverage=(0, 179)) -> EventViews:
    start, end = validate_coverage(coverage)
    columns = {name: [] for name in ("raw_row_id", *FIELDS, "amount_cents")}
    errors, duplicates = [], []
    seen: dict[tuple, int] = {}
    row_count = 0
    total_cents = 0
    for row_id, record in enumerate(records, start=1):
        row_count = row_id
        try:
            if set(record) != set(FIELDS):
                raise DataError("INVALID_SCHEMA", "Exactly the ten source fields are required")
            cleaned = {name: clean_category(record[name]) for name in CATEGORICAL}
            for field in CATEGORICAL:
                if not cleaned[field].strip():
                    raise DataError("INVALID_VALUE", "Empty categorical field", field=field)
            enums = {
                "age": frozenset("0123456U"), "gender": frozenset("FMEU"),
                "zipcodeOri": {"28007"}, "zipMerchant": {"28007"}, "category": CATEGORIES,
            }
            for field, allowed in enums.items():
                if cleaned[field] not in allowed:
                    raise DataError("INVALID_VALUE", "Invalid enumeration", field=field, value=cleaned[field])
            step = parse_integer(record["step"], "step")
            fraud = parse_integer(record["fraud"], "fraud")
            if not start <= step <= end:
                raise DataError("INVALID_VALUE", "Step outside declared coverage", field="step", value=step)
            if fraud not in (0, 1):
                raise DataError("INVALID_VALUE", "Fraud must be 0 or 1", field="fraud", value=fraud)
            amount, cents = parse_money(record["amount"])
            cleaned.update(step=step, fraud=fraud, amount=amount, amount_cents=cents, raw_row_id=row_id)
            # Original field values, before trimming or numeric normalization.
            signature = tuple(sys.intern(str(record[field])) for field in FIELDS)
            if signature in seen:
                duplicates.append({"first_raw_row_id": seen[signature], "duplicate_raw_row_id": row_id})
            else:
                seen[signature] = row_id
            total_cents += cents
            for field in columns:
                columns[field].append(cleaned[field])
        except DataError as exc:
            errors.append({**exc.as_dict(), "raw_row_id": row_id})
    if errors:
        code = "INVALID_SCHEMA" if any(e["code"] == "INVALID_SCHEMA" for e in errors) else "INVALID_VALUE"
        raise DataError(code, "Input rejected; no partial event set returned", rows_read=row_count,
                        validation_errors=errors)
    if not row_count:
        raise DataError("INVALID_SCHEMA", "Input must contain at least one transaction")
    if total_cents > INT64_MAX:
        raise DataError("INVALID_VALUE", "Total cents exceed int64 aggregation capacity")
    frame = pd.DataFrame(columns)
    for field in CATEGORICAL:
        frame[field] = frame[field].astype("string")
    for field in ("raw_row_id", "step", "fraud", "amount_cents"):
        frame[field] = frame[field].astype("int64")
    return EventViews(frame, {
        "validation_errors": [], "rows_read": row_count,
        "exact_duplicate_rows": len(duplicates), "duplicates": duplicates,
        "duplicate_policy": "Reported and retained; no transaction ID proves identity.",
        "attribute_conflicts": attribute_conflicts(frame),
    })


def load_validate_events(path: str | Path, coverage=(0, 179)) -> EventViews:
    """csv.reader preserves decimal text and never skips malformed/blank rows."""
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if header is None or len(header) != len(FIELDS) or set(header) != set(FIELDS):
                raise DataError("INVALID_SCHEMA", "Invalid CSV header", actual_header=header)

            def records():
                for row_id, row in enumerate(reader, start=1):
                    if len(row) != len(header):
                        raise DataError("INVALID_SCHEMA", "CSV record width differs from header",
                                        raw_row_id=row_id, actual_columns=len(row))
                    yield dict(zip(header, row))

            result = validate_records(records(), coverage)
            result.diagnostics["source_columns"] = header
            return result
    except (csv.Error, UnicodeError) as exc:
        raise DataError("INVALID_SCHEMA", "CSV parsing failed", reason=str(exc)) from exc

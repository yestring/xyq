"""Fixed two-ratio business mapping; independent of every forecast amount."""

from .data import DataError

NAMES = ["财富起步客户", "财富增长客户", "财富安享客户"]
RATIOS = dict(zip(NAMES, [(0.80, 0.20), (0.65, 0.35), (0.85, 0.15)]))
NULL_STATUSES = {"GROUP_UNKNOWN", "NON_PERSON", "NO_HISTORY", "CLUSTER_FIT_FAILED", "DATA_GAP",
                 "ATTRIBUTE_CONFLICT", "CONFIG_MISMATCH", "INVALID_VALUE", "INVALID_SCHEMA"}


def map_allocation(group, rule_version="1.0.0"):
    status = group["group_status"]
    if status in NULL_STATUSES:
        return {"group_name": None, "allocation_status": status, "r_conservative": None,
                "r_aggressive": None, "rule_version": rule_version}
    if status not in ("FULL", "AGE_ONLY") or group.get("group_name") not in RATIOS:
        raise DataError("CONFIG_MISMATCH", "No valid business group mapping")
    conservative, aggressive = RATIOS[group["group_name"]]
    if not (0 <= conservative <= 1 and 0 <= aggressive <= 1 and abs(conservative + aggressive - 1) <= 1e-9):
        raise DataError("CONFIG_MISMATCH", "Invalid fixed allocation ratios")
    return {"group_name": group["group_name"], "allocation_status": status,
            "r_conservative": conservative, "r_aggressive": aggressive, "rule_version": rule_version}

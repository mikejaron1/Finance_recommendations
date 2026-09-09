"""Compatibility exports; the backend owns all page-input persistence policy."""
from finrec.storage import (
    load_page_inputs,
    merge_page_inputs,
    migrate_legacy_inputs,
    page_inputs_migration_status,
)

__all__ = [
    "load_page_inputs", "merge_page_inputs", "migrate_legacy_inputs",
    "page_inputs_migration_status",
]

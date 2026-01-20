"""CSV utility functions for test result export."""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def save_csv(
    data: List[Dict],
    filepath: Path,
    fieldnames: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    add_index: bool = True,
) -> Path:
    """
    Save data to CSV file with optional sorting and index column.

    Args:
        data: List of dictionaries containing the data to save
        filepath: Path where the CSV file should be saved
        fieldnames: List of column names (auto-detected if None)
        sort_by: Field name to sort by (supports numeric values with units)
        add_index: Whether to add an index column

    Returns:
        Path to the created CSV file
    """
    if not data:
        logger.warning("No data to save to CSV")
        return filepath

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect fieldnames if not provided
    if not fieldnames:
        fieldnames = sorted(set().union(*data))

    # Add index column if requested
    if add_index:
        data_with_index = [{"index": idx, **row} for idx, row in enumerate(data)]
    else:
        data_with_index = data

    # Sort if needed
    if sort_by:
        data_with_index = sorted(data_with_index, key=_create_sort_key(sort_by))

    # Remove internal fields and prepare fieldnames
    final_data = [
        {k: v for k, v in row.items() if not k.startswith("_")}
        for row in data_with_index
    ]

    # Prepare final fieldnames list
    if add_index:
        final_fieldnames = ["index"] + [
            f for f in fieldnames if f not in ("index", "_sort_key")
        ]
    else:
        final_fieldnames = [f for f in fieldnames if not f.startswith("_")]

    # Write CSV
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_data)

    logger.info("CSV saved: %s", filepath)
    return filepath


def _create_sort_key(sort_by: str):
    """Create a sort key function that handles various numeric formats."""

    def sort_key(x: Dict) -> float:
        # Check for internal sort key
        if "_sort_key" in x:
            return float(x["_sort_key"])

        # Get the value to sort by
        val = x.get(sort_by, "0")

        # Handle string values with units (e.g., "1024 kB", "10 MB")
        if isinstance(val, str):
            val = (
                val.replace(" kB", "")
                .replace(" MB", "")
                .replace(" GB", "")
                .replace(" TB", "")
                .strip()
            )

        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    return sort_key

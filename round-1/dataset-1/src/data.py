#!/usr/bin/env python3
"""Convert democratic resilience panel rows to exp_sel_data_out schema."""

import json
import sys
from pathlib import Path

from loguru import logger

WORKSPACE = Path(__file__).parent
IN_PATH = WORKSPACE / "data_out.json"
OUT_PATH = WORKSPACE / "full_data_out.json"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WORKSPACE / "logs" / "data.log"), rotation="10 MB", level="DEBUG")

# Columns used as input features (predictors of democratic resilience)
INPUT_COLS = [
    "year", "edi", "ldi",
    "gini_market", "gini_net",
    "redistribution_swiid", "redistribution_swiid_se",
    "redistribution_owid",
    "gross_socx", "gdp_pc", "schooling",
    "ert_onset_gradual", "ert_onset_coup", "ert_episode_id",
    "democratic_stock",
]


@logger.catch(reraise=True)
def main() -> None:
    logger.info(f"Loading {IN_PATH}")
    data = json.loads(IN_PATH.read_text())
    rows = data["rows"]
    logger.info(f"Loaded {len(rows)} rows")

    examples = []
    for row in rows:
        # input: all feature columns as JSON string
        inp_fields = {k: row.get(k) for k in INPUT_COLS if k in row}
        input_str = json.dumps(inp_fields, ensure_ascii=False)

        # output: country_code (the unit of analysis for democratic resilience studies)
        output_str = str(row["country_code"])

        ex = {
            "input": input_str,
            "output": output_str,
            "metadata_country_code": row["country_code"],
            "metadata_year": row["year"],
            "metadata_task_type": "panel_data",
            "metadata_democratic_stock": row.get("democratic_stock"),
            "metadata_edi": row.get("edi"),
        }
        examples.append(ex)

    out = {
        "metadata": data["metadata"],
        "datasets": [
            {
                "dataset": "democratic_resilience_panel",
                "examples": examples,
            }
        ],
    }

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    sz = OUT_PATH.stat().st_size / 1e6
    logger.info(f"Wrote {OUT_PATH} ({sz:.1f} MB, {len(examples)} examples)")


if __name__ == "__main__":
    main()

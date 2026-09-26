from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import database_health_check, get_db
from src.services.expense_service import expense_service
from src.schemas.expense import ExpenseCreate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_expenses")

EXPENSES_JSON_FILE = PROJECT_ROOT / "data" / "expenses.json"


def migrate_json_to_postgres():
    """Migrates existing expenses from data/expenses.json into PostgreSQL database."""
    logger.info("Starting expense data migration script...")

    health = database_health_check()
    if not health.get("connected"):
        logger.error("Cannot migrate: PostgreSQL database is not connected (%s).", health.get("message"))
        sys.exit(1)

    if not EXPENSES_JSON_FILE.exists() or EXPENSES_JSON_FILE.stat().st_size == 0:
        logger.info("No existing JSON expense dataset found at %s. Migration complete.", EXPENSES_JSON_FILE)
        return

    try:
        with open(EXPENSES_JSON_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        logger.error("Failed to read JSON dataset: %s", e)
        sys.exit(1)

    logger.info("Found %d records in %s to migrate.", len(records), EXPENSES_JSON_FILE)

    success_count = 0
    failed_count = 0

    for idx, rec in enumerate(records, 1):
        try:
            amt = rec.get("amount")
            cat = rec.get("category", "general")
            desc = rec.get("description", "Migrated expense")
            dt = rec.get("date")

            res = expense_service.add_expense(
                amount=amt,
                category=cat,
                description=desc,
                expense_date=dt,
            )

            if res.get("success"):
                success_count += 1
                logger.info("  [%d/%d] Migrated: $%.2f - %s (%s)", idx, len(records), float(amt), desc, dt)
            else:
                failed_count += 1
                logger.warning("  [%d/%d] Failed: %s", idx, len(records), res.get("error"))

        except Exception as err:
            failed_count += 1
            logger.error("  [%d/%d] Error migrating record %s: %s", idx, rec, err)

    logger.info("=== MIGRATION SUMMARY ===")
    logger.info("Successfully migrated: %d records", success_count)
    logger.info("Failed records:       %d records", failed_count)


if __name__ == "__main__":
    migrate_json_to_postgres()

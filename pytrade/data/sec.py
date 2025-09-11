import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Collection, Union

import sqlalchemy as sa
from sequel.tables.sec import manual_facts_table, manual_filings_table
from sqlalchemy.engine import Connection

from pytrade.utils.collections import ensure_list

logger = logging.getLogger(__name__)


def get_manual_filings(
        conn: Connection,
        *,
        cik: Optional[Union[int, Collection[int]]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
) -> List[Dict]:
    conds = []
    if cik is not None:
        cik = ensure_list(cik)
        conds.append(manual_filings_table.c.cik.in_(cik))
    if start_time is not None:
        conds.append(manual_filings_table.c.acceptance_time >= start_time)
    if end_time is not None:
        conds.append(manual_filings_table.c.acceptance_time < end_time)
    return conn.execute(sa.select(manual_filings_table).where(*conds)).mappings().all()


def create_manual_filing(conn: Connection, cik: int, accession_number: str,
                         acceptance_time: datetime, form: str) -> int:
    res = conn.execute(
        sa.insert(manual_filings_table).values(
            cik=cik, accession_number=accession_number,
            acceptance_time=acceptance_time, form=form,
        ))
    return res.inserted_primary_key[0]


def get_manual_facts(
        conn: Connection,
        *,
        cik: Optional[Union[int, Collection[int]]] = None,
        accession_numbers: Optional[Collection[int]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
) -> List[Dict]:
    conds = []
    if cik is not None:
        cik = ensure_list(cik)
        conds.append(manual_facts_table.c.cik.in_(cik))
    if accession_numbers is not None:
        conds.append(manual_facts_table.c.accession_number.in_(accession_numbers))
    if start_time is not None:
        conds.append(manual_facts_table.c.start_time >= start_time)
    if end_time is not None:
        conds.append(manual_facts_table.c.end_time < end_time)
    return conn.execute(sa.select(manual_facts_table).where(*conds)).mappings().all()


def create_manual_fact(conn: Connection, cik: int, accession_number: str,
                       name: str, value: str, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       dimensions: Optional[Dict] = None,
                       unit: Optional[str] = None) -> int:
    if dimensions is not None:
        dimensions = json.dumps(dimensions, sort_keys=True)

    res = conn.execute(
        sa.insert(manual_facts_table).values(
            cik=cik, accession_number=accession_number, name=name, value=value,
            start_time=start_time, end_time=end_time, dimensions=dimensions,
            unit=unit,
        ))
    return res.inserted_primary_key[0]


def delete_manual_fact(conn, id_: int) -> None:
    conn.execute(sa.delete(manual_facts_table).where(
        manual_facts_table.c.id == id_))
    logger.info(f"Deleted fact with ID: {id_}")

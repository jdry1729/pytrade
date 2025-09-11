import logging
import os

import sqlalchemy as sa

from pytrade.utils.profile import load_profile

logger = logging.getLogger(__name__)


def get_assets(conn, universe_id: str):
    profile = load_profile()
    universe_path = os.path.join(profile.universes_dir, f"{universe_id}.sql")
    with open(universe_path) as f:
        sql = sa.text(f.read())
    return conn.execute(sql).mappings().all()

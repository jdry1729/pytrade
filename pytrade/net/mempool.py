from itertools import chain
from typing import Tuple

import numpy as np
import pandas as pd
from pytrade.net.http import HttpRequest, _send_request
from pytrade.utils.collections import flatten
from pytrade.utils.retry import retry
from tqdm import tqdm

BASE_URL = "https://mempool.space/api"

BLOCKS_COLUMN_TYPES = {
    "id": str,
    "version": int,
    "timestamp": "datetime64[ns]",
    "bits": int,
    "nonce": int,
    "difficulty": float,
    "merkle_root": str,
    "tx_count": int,
    "size": int,
    "weight": int,
    "previousblockhash": str,
    "mediantime": int,
    "extras.totalFees": int,
    "extras.medianFee": int,
    "extras.feeRange.0": float,
    "extras.feeRange.1": float,
    "extras.feeRange.2": float,
    "extras.feeRange.3": float,
    "extras.feeRange.4": float,
    "extras.feeRange.5": float,
    "extras.feeRange.6": float,
    "extras.reward": int,
    "extras.pool.id": int,
    "extras.pool.name": str,
    "extras.pool.slug": str,
    "extras.avgFee": int,
    "extras.avgFeeRate": int,
    "extras.coinbaseRaw": str,
    "extras.coinbaseAddress": str,
    "extras.coinbaseSignature": str,
    "extras.coinbaseSignatureAscii": str,
    "extras.avgTxSize": float,
    "extras.totalInputs": int,
    "extras.totalOutputs": int,
    "extras.totalOutputAmt": int,
    "extras.medianFeeAmt": float,
    "extras.feePercentiles.0": float,
    "extras.feePercentiles.1": float,
    "extras.feePercentiles.2": float,
    "extras.feePercentiles.3": float,
    "extras.feePercentiles.4": float,
    "extras.feePercentiles.5": float,
    "extras.feePercentiles.6": float,
    "extras.segwitTotalTxs": int,
    "extras.segwitTotalSize": int,
    "extras.segwitTotalWeight": int,
    "extras.header": "object",
    "extras.utxoSetChange": int,
    "extras.utxoSetSize": float,
    "extras.totalInputAmt": float,
    "extras.virtualSize": float,
    "extras.matchRate": float,
    "extras.expectedFees": float,
    "extras.expectedWeight": float,
}


def get_blocks(start_height: int, end_height: int, show_progress: bool = False):
    """
    Gets blocks from start to end inclusive.
    """
    blocks = []
    start = start_height + 14
    heights = range(start, end_height + 1, 15)
    num_reqs = (end_height - start) // 15 + 1
    if (end_height - start) % 15 != 0:
        heights = chain(heights, [end_height])
        num_reqs += 1
    for i in tqdm(heights, total=num_reqs, disable=not show_progress):
        req = HttpRequest(
            base_url=BASE_URL,
            endpoint=f"/v1/blocks/{i}"
        )
        res = retry(_send_request, request=req, initial_interval=10,
                    multiplier=2).json()
        blocks.extend(res)
        if len(res) < 15:
            break
    blocks = [flatten(x) for x in blocks]
    blocks = pd.DataFrame(blocks).set_index("height").sort_index()
    blocks = blocks.loc[start_height:end_height]
    blocks["timestamp"] = pd.to_datetime(blocks["timestamp"], unit="s")
    # explicitly set column types for consistency
    blocks = blocks.astype(BLOCKS_COLUMN_TYPES)
    blocks = blocks.fillna(np.nan)
    return blocks[~blocks.index.duplicated(keep="last")]


def get_hash_rate_and_difficulty(period: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gets hash rate and difficulty.

    Parameters
    ----------
    period
        Can be 3m, 6m, 1y, 2y, 3y or all.

    Returns
    -------
    Hash rate and difficulty.
    """
    req = HttpRequest(
        base_url=BASE_URL,
        endpoint=f"/v1/mining/hashrate/{period}"
    )
    res = _send_request(req).json()
    difficulty = pd.DataFrame(res["difficulty"])
    difficulty["time"] = pd.to_datetime(difficulty["time"], unit="s")
    difficulty = difficulty.set_index("time")

    hash_rate = pd.DataFrame(res["hashrates"])
    hash_rate["timestamp"] = pd.to_datetime(hash_rate["timestamp"],
                                            unit="s")
    # pandas can't store hash rates as int (since numbers are too large), so store
    # as float
    hash_rate = hash_rate.set_index("timestamp")["avgHashrate"].astype(float)
    return hash_rate, difficulty

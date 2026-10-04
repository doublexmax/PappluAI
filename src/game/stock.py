from __future__ import annotations

import random
from typing import List


def stock_draw_available(
    stock_count: int,
    discard_count: int,
    recycle_discard: bool,
) -> bool:
    return stock_count > 0 or recycle_discard and discard_count > 1


def refill_stock(
    stock: List[int],
    discard: List[int],
    rng: random.Random,
    recycle_discard: bool,
) -> bool:
    if stock or not recycle_discard or len(discard) <= 1:
        return False
    recycled = discard[:-1]
    rng.shuffle(recycled)
    stock.extend(recycled)
    del discard[:-1]
    return True

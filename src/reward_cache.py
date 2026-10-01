from collections import OrderedDict

from src.evaluate import (
    _parse_hand,
    _parse_joker,
    _require_nonnegative_integral,
    _require_positive_integral,
)


class RewardCache:
    """Bounded exact results; errors and malformed inputs are never cached."""

    def __init__(self, capacity=512):
        self.capacity = _require_positive_integral(capacity, "capacity")
        self._results = OrderedDict()

    def score(self, hand, joker, required_sequences, cards_in_hand, evaluator):
        counts = _parse_hand(hand)
        selected = _parse_joker(joker)
        quota = _require_nonnegative_integral(required_sequences, "required_sequences")
        target = _require_positive_integral(cards_in_hand, "cards_in_hand")
        key = (counts, selected, quota, target, evaluator)
        if key in self._results:
            self._results.move_to_end(key)
            return self._results[key]
        result = evaluator(
            counts, selected, required_sequences=quota, cards_in_hand=target,
        )
        self._results[key] = result
        if len(self._results) > self.capacity:
            self._results.popitem(last=False)
        return result

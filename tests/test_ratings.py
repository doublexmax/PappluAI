from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from openskill.models import PlackettLuce

from src.evaluation.ratings import RatingStore, discounted_posterior, placements, write_rating_report
from src.game.environment import GameConfig


IDENTITIES = tuple(character * 64 for character in ("a", "b", "c", "d"))


def protocol(players=2):
    return {
        "players": players,
        "game": GameConfig(num_decks=3, cards_in_hand=3, required_sequences=1, max_turns=2).to_dict(),
        "encoding_version": 1, "scoring_version": 1,
        "policy": "greedy-v1", "seating_design": "all-permutations-v1", "seed": 41,
    }


def hand(cards):
    counts = [0] * 52
    for face in cards:
        counts[face] += 1
    return counts


def evidence(job, winner_identity=IDENTITIES[0]):
    winner = None if winner_identity is None else job["participants"].index(winner_identity)
    losers = iter(((6, 19, 38), (32, 45, 25), (7, 20, 38), (33, 46, 25)))
    hands = []
    penalties = []
    for seat in range(len(job["participants"])):
        cards = (1, 2, 3) if seat == winner else next(losers)
        counts = hand(cards)
        hands.append(counts)
        points = 0 if seat == winner else sum(0 if face % 13 == 12 else min(face % 13 + 1, 10) for face in cards)
        penalties.append({
            "points": points, "counted_cards": [0] * 52 if seat == winner else counts,
            "exempt_melds": (
                [{"kind": "sequence", "cards": list(cards), "represented_cards": list(cards), "is_pure": True}]
                if seat == winner else []
            ),
            "qualifying_sequences": int(seat == winner),
        })
    turns = [2] * len(hands) if winner is None else [int(seat <= winner) for seat in range(len(hands))]
    return {
        "seed": job["seed"], "winner": winner,
        "terminal_reason": "turns_exhausted" if winner is None else "win",
        "seat_turns": turns, "action_count": 2 * sum(turns),
        "stock_remaining": 156 - len(hands) * 3 - 2,
        "telemetry": {"stock_draws": [0] * len(hands), "discard_draws": turns, "refill_turns": []},
        "terminal_snapshot": {"hands": hands, "joker": 51},
    }, penalties


def complete(store, job, winner=IDENTITIES[0]):
    raw, penalties = evidence(job, winner)
    store.start_attempt(job["id"])
    store.record_raw(job["id"], raw)
    store.start_attempt(job["id"])
    store.record_scores(job["id"], penalties, 1)
    return raw, penalties


class TestPlacements(unittest.TestCase):
    def test_declared_winner_precedes_tied_losers_even_with_zero_points(self):
        self.assertEqual(placements((0, 14, 14, 38), 0, "win", "undecided"), (1, 2, 2, 4))
        self.assertEqual(placements((0, 0, 5), 0, "win", "points"), (1, 2, 3))

    def test_cap_policies_are_distinct_from_a_declaration(self):
        self.assertEqual(placements((14, 3, 3), None, "turns_exhausted", "points"), (3, 1, 1))
        self.assertEqual(placements((14, 3, 3), None, "turns_exhausted", "tie"), (1, 1, 1))
        self.assertIsNone(placements((14, 3, 3), None, "turns_exhausted", "undecided"))
        self.assertIsNone(placements((14, 3, 3), None, "turns_exhausted", "exclude"))
        for points, winner, ending in (((0, 1), True, "win"), ((0, -1), 0, "win"), ((0, 1), None, "timeout")):
            with self.assertRaises(ValueError):
                placements(points, winner, ending, "points")

    def test_discount_changes_both_estimate_and_information(self):
        self.assertEqual(discounted_posterior(25, 8, 30, 6, 0), (25, 8))
        self.assertEqual(discounted_posterior(25, 8, 30, 6, 1), (30, 6))
        mu, sigma = discounted_posterior(25, 8, 30, 6, 0.25)
        self.assertGreater(mu, 25)
        self.assertLess(mu, 30)
        self.assertGreater(sigma, 6)
        self.assertLess(sigma, 8)
        self.assertAlmostEqual(1 / sigma ** 2, 0.75 / 8 ** 2 + 0.25 / 6 ** 2)


class TestRatingStore(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "ratings.sqlite3"
        self.store = RatingStore(self.path).__enter__()
        for index, identity in enumerate(IDENTITIES):
            self.store.register_competitor(identity, "model-%d" % index, {"fixture": True})
        self.profile = self.store.ensure_protocol(protocol())

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def jobs(self, block=0, profile=None, lineup=None):
        return self.store.schedule_block(
            profile or self.profile, block, 101 + block,
            lineup or IDENTITIES[:2],
        )

    def test_scheduling_is_immutable_and_covers_every_seating(self):
        profile = self.store.ensure_protocol(protocol(4))
        jobs = self.jobs(profile=profile, lineup=IDENTITIES)
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len({tuple(job["seat_order"]) for job in jobs}), 24)
        self.assertEqual([job["id"] for job in jobs], [job["id"] for job in self.jobs(profile=profile, lineup=IDENTITIES)])
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.store.schedule_block(profile, 0, 999, IDENTITIES)

    def test_partial_blocks_are_not_published_as_balanced_evidence(self):
        jobs = self.jobs()
        complete(self.store, jobs[1])
        before = self.store.report(self.profile, IDENTITIES[1], "undecided")
        self.assertEqual(before["completed_blocks"], 0)
        self.assertTrue(all(row["rating"] is None for row in before["rows"]))
        complete(self.store, jobs[0])
        after = self.store.report(self.profile, IDENTITIES[1], "undecided")
        rows = {row["id"]: row for row in after["rows"]}
        self.assertEqual(after["completed_blocks"], 1)
        self.assertGreater(rows[IDENTITIES[0]]["rating"], 1500)
        self.assertEqual(rows[IDENTITIES[1]]["rating"], 1500)
        self.assertEqual(rows[IDENTITIES[0]]["games"], 2)

    def test_live_publication_only_updates_newly_settled_blocks(self):
        engines = []

        def counted_model(**parameters):
            engine = mock.Mock(wraps=PlackettLuce(**parameters))
            engines.append(engine)
            return engine

        for job in self.jobs():
            complete(self.store, job)
        with mock.patch("src.evaluation.ratings.PlackettLuce", side_effect=counted_model):
            first = self.store.report(self.profile, IDENTITIES[1], "points")
            self.assertEqual(sum(engine.rate.call_count for engine in engines), 2)
            self.assertEqual(first, self.store.report(self.profile, IDENTITIES[1], "points"))
            self.assertEqual(sum(engine.rate.call_count for engine in engines), 2)
            next_jobs = self.jobs(1)
            complete(self.store, next_jobs[0])
            self.store.report(self.profile, IDENTITIES[1], "points")
            self.assertEqual(sum(engine.rate.call_count for engine in engines), 2)
            complete(self.store, next_jobs[1])
            current = self.store.report(self.profile, IDENTITIES[1], "points")
            self.assertEqual(sum(engine.rate.call_count for engine in engines), 4)
            with RatingStore(self.path, read_only=True) as reader:
                self.assertEqual(current, reader.report(self.profile, IDENTITIES[1], "points"))

    def test_filtered_and_discounted_mixed_blocks_disclose_effective_imbalance(self):
        jobs = self.jobs()
        complete(self.store, jobs[0])
        complete(self.store, jobs[1], winner=None)
        full = self.store.report(self.profile, IDENTITIES[0], "points")
        completed = self.store.report(self.profile, IDENTITIES[0], "exclude")
        discounted = self.store.report(self.profile, IDENTITIES[0], "weighted-points")
        self.assertTrue(full["raw_seatings_balanced"])
        self.assertTrue(full["effective_seatings_balanced"])
        self.assertEqual(completed["mixed_cap_blocks"], 1)
        self.assertFalse(completed["effective_seatings_balanced"])
        self.assertFalse(discounted["effective_seatings_balanced"])

    def test_duplicate_results_do_not_change_ratings_and_conflicts_fail(self):
        jobs = self.jobs()
        raw, penalties = complete(self.store, jobs[0])
        complete(self.store, jobs[1])
        before = self.store.report(self.profile, IDENTITIES[1], "points")
        self.assertFalse(self.store.record_raw(jobs[0]["id"], raw))
        self.assertFalse(self.store.record_scores(jobs[0]["id"], penalties, 1))
        self.assertEqual(before, self.store.report(self.profile, IDENTITIES[1], "points"))
        corrupt = copy.deepcopy(raw)
        corrupt["stock_remaining"] -= 1
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.store.record_raw(jobs[0]["id"], corrupt)

    def test_reopening_recovers_unfinished_attempts_and_preserves_raw_games(self):
        job = self.jobs()[0]
        self.store.start_attempt(job["id"])
        raw, penalties = evidence(job)
        self.store.record_raw(job["id"], raw)
        self.store.start_attempt(job["id"])
        self.store.close()
        self.store = RatingStore(self.path).__enter__()
        recovered = next(item for item in self.store.jobs(pending_only=True) if item["id"] == job["id"])
        self.assertEqual(recovered["raw"], raw)
        self.assertEqual(self.store.failure_count(job["id"]), 0)
        self.assertEqual(self.store.start_attempt(job["id"]), "score")
        self.store.record_scores(job["id"], penalties, 1)

    def test_raw_result_and_attempt_commit_are_atomic(self):
        job = self.jobs()[0]
        raw, _ = evidence(job)
        with self.assertRaisesRegex(ValueError, "open attempt"):
            self.store.record_raw(job["id"], raw)
        self.assertIsNone(self.store.jobs()[0]["raw"])

    def test_projection_can_change_without_replaying_a_capped_game(self):
        for job in self.jobs():
            complete(self.store, job, winner=None)
        conditional = self.store.report(self.profile, IDENTITIES[0], "undecided")
        points = self.store.report(self.profile, IDENTITIES[0], "points")
        self.assertTrue(all(row["rated_games"] == 0 for row in conditional["rows"]))
        self.assertEqual(sum(row["rated_games"] for row in points["rows"]), 4)
        self.assertEqual(len(self.store.jobs()), 2)
        zero = self.store.report(self.profile, IDENTITIES[0], "weighted-points", cap_weight=0)
        full = self.store.report(self.profile, IDENTITIES[0], "weighted-points", cap_weight=1)
        quarter = self.store.report(self.profile, IDENTITIES[0], "weighted-points", cap_weight=0.25)
        self.assertTrue(all(row["rated_games"] == 0 for row in zero["rows"]))
        self.assertEqual(
            [(row["id"], row["mu"], row["sigma"]) for row in points["rows"]],
            [(row["id"], row["mu"], row["sigma"]) for row in full["rows"]],
        )
        self.assertEqual(sum(row["weighted_games"] for row in quarter["rows"]), 1)
        self.assertFalse(quarter["cap_weight_validated"])

    def test_capped_weight_does_not_discount_declared_games(self):
        for job in self.jobs():
            complete(self.store, job)
        zero = self.store.report(self.profile, IDENTITIES[1], "weighted-points", cap_weight=0)
        full = self.store.report(self.profile, IDENTITIES[1], "weighted-points", cap_weight=1)
        self.assertEqual(zero["rows"], full["rows"])

    def test_read_only_reporter_can_read_but_cannot_mutate_the_live_store(self):
        for job in self.jobs():
            complete(self.store, job)
        expected = self.store.report(self.profile, IDENTITIES[1], "points")
        with RatingStore(self.path, read_only=True) as reader:
            self.assertEqual(reader.report(self.profile, IDENTITIES[1], "points"), expected)
            with self.assertRaisesRegex(RuntimeError, "read-only"):
                reader.register_competitor("e" * 64, "extra", {})

    def test_reporter_cannot_overwrite_coordinator_state(self):
        for name in ("ratings.sqlite3", "status.json", "standings.json", "standings-completed-only.json"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "separate JSON"):
                write_rating_report(self.path, self.path.parent / name, IDENTITIES[0], "points", samples=0)
        result = write_rating_report(self.path, self.path.parent / "report.json", IDENTITIES[0], "points", samples=0)
        self.assertEqual(result["reports"][0]["completed_blocks"], 0)

    def test_bootstrap_replays_whole_blocks_and_reference_uncertainty(self):
        for block in range(32):
            for job in self.jobs(block):
                complete(self.store, job, IDENTITIES[block % 3 == 0])
        report = self.store.report(self.profile, IDENTITIES[1], "points", samples=100, seed=94)
        rows = {row["id"]: row for row in report["rows"]}
        interval = rows[IDENTITIES[0]]["interval95"]
        self.assertIsNotNone(interval)
        self.assertLess(interval[0], interval[1])
        self.assertIsNone(rows[IDENTITIES[1]]["interval95"])
        self.assertEqual(rows[IDENTITIES[1]]["status"], "anchor")
        self.assertEqual(report, self.store.report(self.profile, IDENTITIES[1], "points", samples=100, seed=94))


if __name__ == "__main__":
    unittest.main()

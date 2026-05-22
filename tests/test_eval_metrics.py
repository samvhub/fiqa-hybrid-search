import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval import recall_at_k, mrr


class TestRecallAtK:
    def test_perfect_hit_at_position_1(self):
        assert recall_at_k(["a", "b", "c"], {"a"}, k=10) == 1.0

    def test_hit_beyond_k_not_counted(self):
        assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0

    def test_partial_recall_multiple_relevant(self):
        # 1 of 2 relevant docs in top-3
        result = recall_at_k(["a", "x", "y"], {"a", "b"}, k=3)
        assert result == 0.5

    def test_all_relevant_found(self):
        result = recall_at_k(["a", "b", "c"], {"a", "b"}, k=3)
        assert result == 1.0

    def test_empty_retrieved(self):
        assert recall_at_k([], {"a"}, k=10) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), k=10) == 0.0

    def test_no_overlap(self):
        assert recall_at_k(["x", "y", "z"], {"a", "b"}, k=10) == 0.0


class TestMRR:
    def test_first_hit_rank_1(self):
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_first_hit_rank_2(self):
        assert mrr(["x", "a", "c"], {"a"}) == 0.5

    def test_first_hit_rank_3(self):
        assert abs(mrr(["x", "y", "a"], {"a"}) - 1 / 3) < 1e-9

    def test_no_hit_returns_zero(self):
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_empty_retrieved(self):
        assert mrr([], {"a"}) == 0.0

    def test_empty_relevant(self):
        assert mrr(["a", "b"], set()) == 0.0

    def test_multiple_relevant_uses_first_occurrence(self):
        # "b" is relevant and appears at rank 2; "a" at rank 4 — MRR should be 0.5
        assert mrr(["x", "b", "y", "a"], {"a", "b"}) == 0.5

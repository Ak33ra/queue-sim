"""Smoke tests and invariant checks for QueueSystem.

Covers: basic operation, probabilistic routing, transition matrix
validation, seed reproducibility, and the Server ABC contract.
"""

import pytest

from queue_sim import FCFS, SRPT, QueueSystem, Server, genExp


class TestSeedReproducibility:

    def test_same_seed_same_result(self) -> None:
        system = QueueSystem([FCFS(sizefn=genExp(2.0))], arrivalfn=genExp(1.0))
        r1 = system.sim(num_events=10_000, seed=42)
        r2 = system.sim(num_events=10_000, seed=42)
        assert r1 == r2

    def test_different_seed_different_result(self) -> None:
        system = QueueSystem([FCFS(sizefn=genExp(2.0))], arrivalfn=genExp(1.0))
        r1 = system.sim(num_events=10_000, seed=42)
        r2 = system.sim(num_events=10_000, seed=99)
        assert r1 != r2


class TestTransitionMatrix:

    def test_valid_matrix_accepted(self) -> None:
        """A well-formed transition matrix should not raise."""
        P = [[0, 0.5, 0.5],
             [0.5, 0, 0.5]]
        system = QueueSystem(
            [FCFS(sizefn=genExp(2.0)), SRPT(sizefn=genExp(2.0))],
            arrivalfn=genExp(1.0),
            transitionMatrix=P,
        )
        system.sim(num_events=1_000, seed=1)

    def test_rows_must_sum_to_one(self) -> None:
        P = [[0, 0.5, 0.3],     # sums to 0.8
             [0.5, 0, 0.5]]
        system = QueueSystem(
            [FCFS(sizefn=genExp(2.0)), SRPT(sizefn=genExp(2.0))],
            arrivalfn=genExp(1.0),
            transitionMatrix=P,
        )
        with pytest.raises(ValueError, match="sums to"):
            system.sim(num_events=1_000, seed=1)

    def test_wrong_dimensions_rejected(self) -> None:
        P = [[0.5, 0.5]]  # 1x2 but need 2x3 for 2 servers
        system = QueueSystem(
            [FCFS(sizefn=genExp(2.0)), SRPT(sizefn=genExp(2.0))],
            arrivalfn=genExp(1.0),
            transitionMatrix=P,
        )
        with pytest.raises(ValueError, match="Transition matrix must be"):
            system.sim(num_events=1_000, seed=1)


class TestTandemNetwork:

    def test_two_server_tandem(self) -> None:
        """A tandem FCFS -> SRPT network should produce positive E[N], E[T]."""
        system = QueueSystem(
            [FCFS(sizefn=genExp(4.0)), SRPT(sizefn=genExp(4.0))],
            arrivalfn=genExp(1.0),
        )
        N, T = system.sim(num_events=50_000, seed=7)
        assert N > 0
        assert T > 0


class TestABCEnforcement:

    def test_cannot_instantiate_server_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            Server(sizefn=lambda: 1.0)  # type: ignore[abstract]

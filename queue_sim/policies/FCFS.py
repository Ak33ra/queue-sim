"""First Come First Served (FCFS) scheduling policy.

Jobs are served in arrival order. Intuitive picture: one checkout line,
one register.
"""

from typing import Callable

from ..server import Server


class FCFS(Server):

    def __init__(self, sizefn: Callable[[], float]) -> None:
        super().__init__(sizefn)

    def nextJob(self) -> float:
        return self.genSize()


__all__ = ['FCFS']

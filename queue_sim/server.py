import random
from abc import ABC, abstractmethod
import math
from collections import deque
from typing import List, Callable

'''
Parent class from which specific service policies inherit their methods

Implements the core functionality of a server: processing jobs and reporting event times to the system

To create a custom policy, create a child of this class and implement the nextJob function,
which defines the service policy
'''

class Server:

    def genExp(mu):
        return lambda:-(1/mu)*math.log(1-random.random())

    # jobs must be an array where elements are tuples: (arrival time, job size)
    # optionally supply functions to generate job size, default to Exp(1)
    def __init__(self, sizefn : Callable[[], float]):
        self.genSize : Callable[[], float] = sizefn
        self.clock = 0
        self.arrivalTimes = deque()
        self.TTNC = math.inf
        self.area_N = 0
        self.T = 0
        self.N = 0
        self.num_completions = 0
        self.state = 0

    # defines the service policy
    # needs to be implemented by a server class inheritor
    @abstractmethod
    def nextJob(self) -> float: ...

    def updateET(self): # works if jobs are served fifo
        t = self.clock - self.arrivalTimes.popleft()
        self.T = self.T*(self.num_completions-1)/self.num_completions + t/(self.num_completions)
        
    def arrival(self): # server just got an arrival
        self.arrivalTimes.append(self.clock)
        if (self.state == 0):
            self.TTNC = self.nextJob()
        self.state += 1

    def queryTTNC(self):
        return self.TTNC
        
    def update(self, timeElapsed : float): # return (True, TTNC) if just completed a job, (FALSE, TTNC) otherwise
        self.TTNC -= timeElapsed
        self.clock += timeElapsed
        if (self.TTNC <= 0.0):
            self.state -= 1
            if (self.state == 0):
                self.TTNC = math.inf
            else:
                self.TTNC = self.nextJob()
            self.num_completions += 1
            self.updateET()
            return True
        return False
    
__all__ = ["Server"]
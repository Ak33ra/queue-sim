import math
from enum import Enum
from .server import Server
from typing import List
from .lib import rvGen

# the system is a class that chains together multiple servers and simulates a job sequence on it
# reports mean response time, num jobs, and load
# at each time step: compute TTNA for each server and send it
# servers report back TTNE
# execute servers with min TTNE, update all TTNAs

# TODO: deterministic arrivals and/or job sizes
# TODO: probabilistic routing

class Event(Enum):
    ARRIVAL = 0
    COMPLETION = 1

class QueueSystem:

    def __init__(self, servers, arrivalfn, deterministicArrivals = False, transitionMatrix = []):
        self.startServer : Server = servers[0]
        self.deterministicArrivals = deterministicArrivals
        self.genArrival = arrivalfn
        self.servers : List[Server] = servers
        self.numServers = len(servers)
        self.transitionMatrix = transitionMatrix
        self.T = 0

    def addServer(self, server):
        self.servers.append(server)
        self.numServers += 1

    def updateTransitionMatrix(self, M):
        self.transitionMatrix = M

    def verifyTransMatrix(self):
        if (self.transitionMatrix == []):
            return
        n = len(self.transitionMatrix)
        if (n != len(self.transitionMatrix[0])-1):
            raise ValueError("Transition matrix must be n by n+1, where n = numServers\n" \
                             "The n+1th column is the probability of the job exiting the system\n"
                             "update the transition matrix to [] if non-probabilistic routing is desired")
        for row in range(n):
            if sum(self.transitionMatrix[row]) != 1.0:
                raise ValueError("Transition matrix rows must sum to 1.0\n" \
                                 "The n+1th column is the probability of the job exiting the system\n"
                                 "update the transition matrix to [] if non-probabilistic routing is desired")

    def getNextServer(self, currServer):
        u = rvGen.Uniform(0,1)
        probabilities = self.transitionMatrix[currServer]
        acc = 0
        for (i,p) in enumerate(probabilities):
            acc += p
            if (u < acc):
                return i
            
    def processStats(self):
        return self.T

    def computeTTNC(self):
        m = math.inf
        for server in self.servers:
            TTNC = server.queryTTNC()
            m = min(m, TTNC)
        return m

    def sim(self, NUM_EVENTS = 10**6):
        self.verifyTransMatrix()
        deterministicRouting = (self.transitionMatrix == [])

        num_completions = 0
        TTNA = self.genArrival()
        TTNC = math.inf
        TTNE = TTNA
        area_N = 0
        state = 0
        clock = 0

        while (num_completions < NUM_EVENTS):
            TTNE = min(TTNC, TTNA)
            clock += TTNE
            area_N += state * TTNE

            completed = []
            for id, server in enumerate(self.servers):
                done = server.update(TTNE)
                if (done):
                    completed.append(id)
                    
            TTNA -= TTNE
            TTNC -= TTNE

            if (TTNC == 0.0):
                for id in completed:
                    if (deterministicRouting):
                        nextServer = id + 1
                    else:
                        nextServer = self.getNextServer(id)
                    if (nextServer == self.numServers):
                        num_completions += 1
                        state -= 1
                    else:
                        next = self.servers[nextServer]
                        next.arrival()

            if (TTNA  == 0.0):
                state += 1
                self.startServer.arrival()
                TTNA = self.genArrival()
                        
            TTNC = self.computeTTNC()

        self.T = area_N / max(1, num_completions)
        return (area_N / clock, self.T)
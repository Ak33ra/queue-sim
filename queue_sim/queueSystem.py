import math
from enum import Enum
from .server import Server
from typing import List

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
        # tuples (TIME TO EVENT, SERVERID, EVENT)
        # self.nextEvent = (self.genArrival(), 0, Event.ARRIVAL)
        self.T = 0

    def addServer(self, server):
        self.servers.append(server)
        self.numServers += 1

    def verifyTransMatrix(self):
        # make sure dimensions match with number of servers, and that all probabilities add up to 1.0
        # if not, throw an error
        pass

    def processStats(self):
        return self.T

    def computeTTNC(self):
        m = math.inf
        for server in self.servers:
            TTNC = server.queryTTNC()
            m = min(m, TTNC)
        return m

    def sim(self, NUM_EVENTS = 10**6):
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
                    if (id == self.numServers - 1):
                        num_completions += 1
                        state -= 1
                    else:
                        next = self.servers[id + 1]
                        next.arrival()

            if (TTNA  == 0.0):
                state += 1
                self.startServer.arrival()
                TTNA = self.genArrival()
                        
            TTNC = self.computeTTNC()

        self.T = area_N / max(1, num_completions)
        return (area_N / clock, self.T)
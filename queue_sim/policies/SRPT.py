from queue_sim.server import *
import heapq

class SRPT(Server):

    def __init__(self, sizefn = None):
        if (sizefn == None):
            super().__init__()
        else:
            super().__init__(sizefn)
        self.jobs = []
        heapq.heapify(self.jobs)

    def nextJob(self):
        return heapq.heappop(self.jobs)
    
    def updateET(self):
        return

    def arrival(self):
        if (self.state > 0):
            heapq.heappush(self.jobs, self.TTNC)
        heapq.heappush(self.jobs, self.genSize())
        self.TTNC = heapq.heappop(self.jobs)
        self.state += 1
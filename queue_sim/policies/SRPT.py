from ..server import *
import heapq

'''
Implements the Shortest Remaining Processing Time (SRPT) policy
Always serves the job with the lowest remaining processing time, with preemptions
This policy is locally optimal for minimizing mean response time, but may make globally suboptimal decisions
Intuitively, if you're at a grocery store and only buying a piece of gum, you wouldn't want to wait behind someone
    buying half the snacks aisle!
'''
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

__all__ = ['SRPT']
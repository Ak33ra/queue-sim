from queue_sim.server import *

class FCFS(Server):

    def nextJob(self):
        return self.genSize()
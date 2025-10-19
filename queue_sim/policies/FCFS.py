from queue_sim.server import *

'''
Implements a standard First Come First Served (FCFS) policy
Intuitive picture: one checkout line, one register
'''
class FCFS(Server):

    def nextJob(self):
        return self.genSize()
    
__all__ = ['FCFS']
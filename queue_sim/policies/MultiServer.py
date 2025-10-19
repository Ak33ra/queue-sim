from queue_sim.server import *

'''
Multiple servers taking jobs from one queue
Could be used to simulate a work-queue threading model
Intuitive picture: one checkout line and multiple registers
'''
class MultiServer(Server):

    def nextJob(self):
        return self.genSize()
    

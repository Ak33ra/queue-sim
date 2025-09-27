from queue_sim.server import *

# multiple servers taking jobs from one queue
# currently only supports servers with the same policy
class MultiServer(Server):

    def nextJob(self):
        return self.genSize()
    

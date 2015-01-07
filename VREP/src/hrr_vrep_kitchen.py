# kleinhans.ashley@gmail.com
# 07 Jan 2015
# -------------------------------------------------------------------
# Make sure to have the server side running in V-REP: 
# in a child script of a V-REP scene, add following command
# to be executed just once, at simulation start:
#
# simExtRemoteApiStart(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!

import vrep

print 'Program started'
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19997,True,True,2000,5)
if clientID!=-1:
    print 'Connected to remote API server'
    res,objs=vrep.simxGetObjects(clientID,vrep.sim_handle_all,vrep.simx_opmode_oneshot_wait)
    if res==vrep.simx_return_ok:
        print 'Number of objects in the scene: ',len(objs)
        print 'Connection to remote API server is open number:', clientID
        
        res = vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)      
        
    else:
        print 'Remote API function call returned with error code: ',res
else:
    print 'Failed connecting to remote API server'
    
    res, bowl_handle = vrep.simxGetObjectHandle(clientID, 'Bowl_upperPart', vrep.simx_opmode_oneshot_wait )
    print 'bowl_handle', bowl_handle
    
#    res = simxStopSimulation(clientID, vrep.simx_opmode_oneshot)
    
    vrep.simxFinish(clientID)
    
print 'Program ended'

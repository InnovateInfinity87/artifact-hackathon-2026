"""Stage A on lost-beam days, from challenges/optimisation:
    python MFopt/test_threading.py <misalignment_variation> table|notable
"""
import sys,functools,numpy as np; sys.path[:0]=['.','RL_framework']
from rl_framework.config import RLConfig
from MFopt.optics_env import OpticsEnv
from MFopt.harness import run_episode, summarise
from MFopt.steering import optics_and_steering
mv=float(sys.argv[1]); table=sys.argv[2]=='table'
env=OpticsEnv(RLConfig(misalignment_variation=mv)); res=[]
pol=functools.partial(optics_and_steering, table=table)
for seed in range(1000,1012):
    r=run_episode(env,pol,seed); res.append(r)
    reach=[(t['transmission']) for t in r.trace]
    print(seed,'score %.2f err %.2f T %.2f steps %d'%(r.score,r.relative_error,r.transmission,r.steps),'T trace',[round(x,2) for x in reach],flush=True)
print(summarise(res))

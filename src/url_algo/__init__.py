from .wurl import apwd
from .gwd import gwd 


REGISTRY = {}
REGISTRY["wurl"] = apwd.assign_reward
REGISTRY["gwd"] = gwd.assign_reward

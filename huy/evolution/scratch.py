import pymongo
from bson.objectid import ObjectId
import sys
sys.path.insert(0, "/home/ubuntu/nevir")
from auto.utils import get_mongo_uri
client = pymongo.MongoClient(get_mongo_uri())
col = client["alpha"]["gen_alpha"]
doc = col.find_one({"alpha_name": "alpha_mining_002_wf"})
if doc:
    print(doc['alpha_code'])
else:
    print("Not found")


from pymongo import MongoClient
from bson import ObjectId
from auto.utils import get_mongo_uri
from base_auto.utils import make_key_base
mongo_client = MongoClient(get_mongo_uri())
db = mongo_client["base"]
base_coll = db["base_collection"]
base_results = db["wfa_results"]


doc = base_results.delete_many({})
print(doc)

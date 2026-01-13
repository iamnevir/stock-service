
from pymongo import MongoClient
from bson import ObjectId
from auto.utils import get_mongo_uri
from base_auto.utils import make_key_base
mongo_client = MongoClient(get_mongo_uri())
db = mongo_client["base"]
correlation_results = db["correlation_results"]
correlation_backtest = db["correlation_backtest"]


from pymongo import ASCENDING

from pymongo import MongoClient
from bson import ObjectId
from auto.utils import get_mongo_uri
from base_auto.utils import make_key_base
mongo_client = MongoClient(get_mongo_uri())
db = mongo_client["base"]
base_coll = db["base_collection"]
base_results = db["wfa_results"]
correlation_coll = db['correlation_results']
# index (_id_) đã có sẵn mặc định, KHÔNG cần tạo lại

# 1. Index: x_1_y_1_c_1
correlation_coll.create_index(
    [
        ("x", ASCENDING),
        ("y", ASCENDING),
        ("c", ASCENDING),
    ],
    name="x_1_y_1_c_1"
)

# 2. Index unique: x_1_y_1
correlation_coll.create_index(
    [
        ("x", ASCENDING),
        ("y", ASCENDING),
    ],
    name="x_1_y_1",
    unique=True
)

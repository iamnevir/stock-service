from pymongo import MongoClient

from busd_auto.utils import get_mongo_uri


db = MongoClient(get_mongo_uri())['busd']
busd_collection = db["busd_collection"]
correlation_coll = db["correlation_results"]

from pymongo import ASCENDING

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


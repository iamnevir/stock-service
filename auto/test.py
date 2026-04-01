from pymongo import MongoClient, ASCENDING
from utils import get_mongo_uri

client = MongoClient(get_mongo_uri())
db = client["alpha"]
collection = db["correlation_results"]
collection.drop()
collection = db["correlation_results"]
# Index (_id_) MongoDB tự tạo, không cần tạo lại

# Index: x_1_y_1_c_1
collection.create_index(
    [("x", ASCENDING), ("y", ASCENDING), ("c", ASCENDING)],
    name="x_1_y_1_c_1"
)

# Index: x_1_y_1 (unique)
collection.create_index(
    [("x", ASCENDING), ("y", ASCENDING)],
    name="x_1_y_1",
    unique=True
)

print("Đã tạo xong các index")
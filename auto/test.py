from pymongo import MongoClient
import os

from utils import get_mongo_uri

client = MongoClient(get_mongo_uri())

print(client.list_database_names())
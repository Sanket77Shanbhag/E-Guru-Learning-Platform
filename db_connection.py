from pymongo import MongoClient

# MongoDB connection

client = MongoClient('mongodb://localhost:27017/')
db = client['eguru']



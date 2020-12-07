# pymongo
import pymongo
import pandas as pd
from gridfs import GridFS
# util
import io
import os

class MongoGridFS:
    def __init__(self,uri): 
        self.connection = pymongo.MongoClient(uri)
    def setDB(self, dbname):
        self.db = self.connection[dbname]
    def setGridFS(self):
        self.fs = GridFS(self.db)
    # def put(self, data, **kwargs):
    #     self.fs.put(data, **kwargs)
    def findFile(self):
        return self.db.fs.files.find()
    def get(self, filename): 
        content = self.fs.get_version(filename).read() #get grifs file from mongoDB
        content = content.decode("utf-8") #필수!
        return content #string으로 반환
    
def getFileFromDB(filename):
    # 객체생성 접속
    o = MongoGridFS(os.environ["mongoURI"])
    # db connect
    o.setDB('test')
    # GridFS 객체 생성
    o.setGridFS()
    ret = o.get(filename)
    return ret

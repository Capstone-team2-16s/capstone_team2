import getFile
import modelResult

res = getFile.getFileFromDB("sample2.txt")
print(res)

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from getFile import MongoGridFS

app = Flask(__name__)
CORS(app)
#app.register_blueprint(bye, url_prefix="/bye") #blueprint 등록

@app.route('/')
def serverTest():
    return "flask server"

@app.route('/result',methods=['GET'])
def resultTest():
    #get name or id from node server
    if(request.method=='GET'):
        name = request.args.get('name')
        file_name = 'twitter_'
        fileformat = '.txt'
        filename = file_name + name + fileformat
        filename = name
        tweets = getFile.getFileFromDB(filename)
        res = modelResult.analysisResult(tweets)
    return res

#running server
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
if __name__ == '__main__':
    app.run()

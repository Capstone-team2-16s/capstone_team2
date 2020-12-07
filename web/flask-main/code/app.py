import getFile
import modelResult

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from getFile import MongoGridFS

app = Flask(__name__)
CORS(app)

@app.route('/')
def serverTest():
    return "flask server"

#get name from node server and return analysis result 
@app.route('/result',methods=['GET']) 
def resultTest():
    #get name or id from node server
    if(request.method=='GET'):
        name = request.args.get('account')
        print(name)
        file_name = 'twitter_'
        fileformat = '.txt'
        filename = file_name + name + fileformat
        tweets = getFile.getFileFromDB(filename) #txt file.read(<class 'str'>)
        res = modelResult.analysisResult(tweets)
    return res

@app.route('/test',methods=['GET'])
def test():
    print(request.args.get('name'))
    return "HI"
#running server
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
if __name__ == '__main__':
    app.run(host='0.0.0.0') #추후 수정 필요
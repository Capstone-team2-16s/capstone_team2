from flask import Flask, jsonify, render_template, request
from modelResult import resDict
from flask_cors import CORS
from werkzeug.utils import secure_filename
from wordcloudEx import showing

app = Flask(__name__)
CORS(app)
#app.register_blueprint(bye, url_prefix="/bye") #blueprint 등록

@app.route('/')
def serverTest():
    return "flask server"

#node - flask
@app.route('/test',methods=['GET','POST'])
def test00():
    data = request.args.get('name')
    data2 = request.args.get('email')
    if request.method == 'POST':
        #data = request.data
        test = {"data":"testdata"}
        print(data)
        print("data type :",type(data))
        return data
    elif request.method=='GET':
        return {"data1":data,"data2":data2}
    else:
        return "hello"

@app.route('/result')
def resultTest():
    #show wordcloud
    #showing
    return resDict


#running server
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
if __name__ == '__main__':
    app.run()
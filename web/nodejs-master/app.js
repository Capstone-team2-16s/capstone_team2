const express = require('express'); //express 모듈요청
const path = require('path');
const app = express(); //app을 express 프레임워크로 킴
const ejs = require("ejs"); //ejs 모듈 요청
const port = 8000;
const bodyParser = require('body-parser');

//application/x-www-form-urlencoded를 분석해서 가져옴
app.use(bodyParser.urlencoded({extended: true}));
//application/json분석해서 가져옴
app.use(bodyParser.json());

const mongoose = require('mongoose') 
const { Target } = require('./models/Target');

// 외부 DB 접속
mongoose.connect(process.env.mongoURI,{
    useNewUrlParser: true, useUnifiedTopology: true, useCreateIndex: true, useFindAndModify: false
}).then(()=>console.log("mongoDB connected.."))
.catch(err=>console.log(err))

/*-- ejs 설정 --*/
app.set("view engine","ejs"); //app에 view engine을 설치. ejs를 템플릿으로
app.use(express.static(__dirname+'/public')); //view 폴더 경로는 프로젝트 폴더.(__dirname이 폴더위치) =>public folder로 마운트

var mainRouter = require('./routes/main');
app.use('/',mainRouter);

// 분석결과 페이지
app.get('/result',(req,res)=>{ 
    var key = req.query.searchKey;
    Target.findOne({name:{$regex: new RegExp('^' + key, 'i')}},function(err,targets){
        if(err) {
            return res.status(500).send({error: err.message});
          }
            console.log("dataInfo:",targets);
            res.render('result',{targetInfo:targets});
         });
    res.locals.key = req.query.searchKey;
});


app.listen(port,()=>console.log(`Example app listening on port ${port}`));


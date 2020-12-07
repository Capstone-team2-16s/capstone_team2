const mongoose = require('mongoose');
const Schema = mongoose.Schema;
const targetSchema = mongoose.Schema({
    //필드 작성
    name : {
        type: String,
        maxlength:50
    },
    account:{
        type:String,
        trim:true
    },
    image:String,
    job:{
        type: String
    }
});
//{"name":"Donald Trump","account":"realDonaldTrump","image":"C:\data\db\web.png","job":"President"},

//모델로 감싸주기. 모델이름, 스키마 이름
const Target = mongoose.model('Target',targetSchema);

//다른 파일에서도 모델 쓸 수 있게 
module.exports = { Target }
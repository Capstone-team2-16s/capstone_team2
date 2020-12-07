const mongoose = require('mongoose');
const Schema = mongoose.Schema;
const userSchema = mongoose.Schema({
    //필드 작성
    name : {
        type: String,
        maxlength:50
    },
    email:{
        type:String,
        trim:true
        //unique:1
    },//오브젝트형태
    image:String
});

//모델로 감싸주기. 모델이름, 스키마 이름
const User = mongoose.model('User',userSchema);

//다른 파일에서도 모델 쓸 수 있게 
module.exports = { User }
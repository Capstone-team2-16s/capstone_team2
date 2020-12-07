if(process.env.NODE_ENV === 'production'){
    module.exports=require('./prod'); //배포판이면 
}else{
    module.exports=require('./dev'); //배포판이면 
}
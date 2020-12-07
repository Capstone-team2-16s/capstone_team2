var getTargetInfo = (function () {
    var targetInfo;
    $.ajax({
        url:'/targetInfoData',
        async:false,
        method:'get',
        dataType:'json',
        success:function(res){
            targetInfo =res;
        },error:function(err){
            console.log("/getData Err",err)
            targetInfo = err;
        }
    });
    return{
        //getter
        getData : function () {
           return targetInfo;
        }
    };
})();
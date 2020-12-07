/* 트위터 공유하기 코드
* 개인정보 공유에 대한 우려가 있어 페이지에서 삭제 조치
*/
$(function(){
    $("#shareBtn").click(function(){
        shareAct();
    });
});

function shareAct(){
    var cUrl = "https://capstoneteam2.tk/";
    cUrl = 'https://twitter.com/intent/tweet?:&url='+cUrl;
    window.open(cUrl,'','width=600,height=300,top=100,left=100,scrollbars=yes');
}
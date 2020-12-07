//show profile 
var showProfile = function(targetInfo){
    // console.log(targetInfo);
        for(var i=0;i<targetInfo.length;i++){
                $('.media-list').append(
                `<div class="col-lg-4 col-md-4" style="margin:5px 0">
                    <div class="media-left">
                        <a href="#" id="figure">
                        <img id="img" name="${targetInfo[i].name}" class="media-object profile-image" src="${targetInfo[i].image}" alt="personThumnail">
                        </a>
                    </div>
                    <div class="media-body">
                        <div>
                            <a href="#" style="color:rgb(69, 69, 80)"><p class="body-top header-h3" id="name" name="${targetInfo[i].name}" >${targetInfo[i].name}</p>
                            </a>
                        </div>
                        <div style="font-weight:600">
                            <p>${targetInfo[i].account}</p>
                            <p>${targetInfo[i].job}</p>
                        </div>
                    </div>
                </div>`
                )
        }
}

//autocomplete search key
var autoKey = function(targetInfo){
    const names = targetInfo.map(i=>i.name);
    $("#completeBar").autocomplete({
        source : names,
        select : function(event,ui){
            //console.log("dd",ui.item);
        },
        focus : function(event, ui) {	//포커스 가면
            return false;//한글 에러 잡기용도로 사용됨
            },
        minLength: 1,// 최소 글자수
        autoFocus: true, //첫번째 항목 자동 포커스 기본값 false
        classes: {	//잘 모르겠음
            "ui-autocomplete": "highlight"
            },
        delay: 500,	//검색창에 글자 써지고 나서 autocomplete 창 뜰 때 까지 딜레이 시간(ms)
        //			disabled: true, //자동완성 기능 끄기
        position: { my : "right top", at: "right bottom" },	//잘 모르겠음
        close : function(event){	//자동완성창 닫아질때 호출
       // console.log(event);
        }
    });
        

}


var check = function(data,name){
    //검색하려는 인물이 db에 있는지
    for(i in data){
        if(name==data[i].name){
            return 1; //있으면 1
        }
    }
    return 0; //없으면 0
};

//click event : 분석 여부 묻는 함수
var isClick = function(data){
    $("#submitBtn1, #submitBtn2, #name, #img").on("click",function(e){
        
        //if clicked name or img 
        var clickedTarget = e.target.id;

        if(!(clickedTarget=='submitBtn1'||clickedTarget=='submitBtn2')){ //버튼이 아니면
            var cardId = $(this).attr("name");
            $('#completeBar').val(cardId);
        }
         
        var result = confirm('분석을 진행하시겠습니까?'); 
        if(result) { //yes 
            //검색 값없으면 거절
            if($('#completeBar').val()==""){
                alert("검색어를 입력해주세요.");
                $('#completeBar').focus();
                return false;
            }
            //데이터에 없어도 분석 불가
            else if(!check(data, $('#completeBar').val())){
                alert("분석할 수 없는 인물입니다.");
                return false;
            }
            else{
                $("#searchForm").submit();
            }
        } else { //no 
            return false;
        }
    });
};

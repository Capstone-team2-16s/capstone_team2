// 결과 페이지 전반적인 기능을 담고있음
$(function () {
    key = $("#key").text(); //미들웨어 이용하는 방법 같은거 찾아보기 
    console.log(key);
    $.ajax({
        //nodejs와 flask 서버 사이에서 데이터 주고받으려니 CORS 이슈가 생겼다. (로컬에선 괜찮았는데 호스팅하면서 도메인을 다르게 주어서)
        headers: {  'Access-Control-Allow-Origin': 'http://The web site allowed to access' },  
        // url: 'https://api.capstoneteam2.tk/result',
        url: 'https://api.capstoneteam2.co/result',
        data: {
            account: key
        },
        method: 'get',
        dataType: 'json',
        success: function (res) {
            console.log(res)
            words = res.countWords.words //자주 쓰는 단어
            counts = res.countWords.counts //자주 쓰는 단어별 빈도수 
            ratio = res.sentiment.ratio //label 별 비율
            sentences = res.sentiment.sentence //문장 관련 정보들 : 긍정,부정,중립 문장 예시
            lines = res.sentiment.lines //전체 문장 수 

            chart.updateOptions({
                series: [{
                    name: 'POSITIVE',
                    data: [ratio[0]]
                }, {
                    name: 'NEGATIVE',
                    data: [ratio[1]]
                }, {
                    name: 'NEUTRAL',
                    data: [ratio[2]]
                }]
            });
             
            $('#lines').append(`<smallheader>(전체 문장 수 : `+lines+`)</smallheader>`);

            var msg_pos = '';
            var msg_neg = '';
            var msg_neu = '';
            for (var i in sentences.pos) {
                msg_pos += '<li>' + sentences.pos[i] + '</li>';
                msg_neg += '<li>' + sentences.neg[i] + '</li>';
                msg_neu += '<li>' + sentences.neu[i] + '</li>';
            }

            $('#pos').html('<ul class="list-content">' + msg_pos + '</ul>');
            $('#neg').html('<ul class="list-content">' + msg_neg + '</ul>');
            $('#neu').html('<ul class="list-content">' + msg_neu + '</ul>');


            //자주쓰는 단어 TOP 5 
            var wordMsg = '';
            for (var i in words) {
                wordMsg += '<div><div class="box black"></div>'+'<li style="display:inline-block;vertical-align: center;">' + words[i] + '</li></div>';
            }

            $('#wordsTop').html('<ul>' + wordMsg + '</ul>'); //마우스 hover 시 개수 나타내야함

            // 워드클라우드 데이터
            wordsdata = res.countWords.all_worlds; //사용된 단어중 100개
            var wordList=[];
            for(i in wordsdata){
                var ob = {text:wordsdata[i][0],frequency:wordsdata[i][1]};
                wordList.push(ob);
            }
            wordCloud(wordList);
            
            //로딩이미지 hide
            $('.loader').hide(300);
        },
        error: function (err) {
            console.log("에러", err);
        }

    });

});
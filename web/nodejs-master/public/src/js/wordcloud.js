
var wordCloud = function(data){
    var width = 800,
    height = 500
    
    // 단어 빈도수에따라 색상 다르게
    data.sort(function(a, b) {
        return b.frequency - a.frequency;
      }); //내림차순 정렬

    data_len = data.length;
    data_len *=0.8;
    
    var _max = data[0].frequency;
    var max = data[4].frequency; //상위 5번쨰 단어 frequency
    var min = data[data_len].frequency; //하위 20번째


    var svg = d3.select("#graph").append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("id","wordcloud")
    
    
    wordScale = d3.scale.linear().domain([0, _max]).range([0, 100]).clamp(true);
    var svg = d3.select("#wordcloud") //select는 안되고 selectAll은 된다=> 아펙스차트도 svg라 둘 중 어떤거 골라야할지 몰라 안나온듯 -> id 따로 추가
                .append("g")
                .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")")
    
    
    show(data)
    setInterval(function(){
         show(data)
    },3000) 

    function show(data){

        d3.layout.cloud().size([width, height])
            //클라우드 레이아웃에 데이터 전달
            .words(data)
            .rotate(function (d) {
            return d.text.length > 3 ? 0 : 90;
            })
            .font('Impact')
            //스케일로 각 단어의 크기를 설정
            .fontSize(function (d) {
                    return wordScale(d.frequency);
            })
            //클라우드 레이아웃을 초기화 > end이벤트 발생 > 연결된 함수 작동  
            .on("end", draw)
            .start();
    
        function draw(data) { 
            var cloud = svg.selectAll("text").data(data)
            //Entering words
            cloud.enter()
                .append("text")
                .style("font-family", "Impact")
                .style("line-height","2")
                .style("fill", function (d) {
                    //default
                    var color = "#cfcdf7"; 
                    //자주 쓰는단어 5개
                    if(d.frequency>=min){
                        color = "#9a96f2";
                    }
                    if(d.frequency>=max){
                        color = "#8c87ed";
                    }
                    return (color);
                })
                .style("fill-opacity", .5)
                .attr("text-anchor", "middle") 
                .attr('font-size', 1)
                .text(function (d) {
                    return d.text;
                }); 
            cloud
                .transition()
                .duration(600)
                .style("font-size", function (d) {
                    return d.size + "px";
                })
                .attr("transform", function (d) {
                    return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
                })
                .style("fill-opacity", 1); 
        }
    
    }
    
}


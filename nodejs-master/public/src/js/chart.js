//문장 비율 그래프를 그림
var options = {
    series: [{
      x: "2018-09-10",
    name: 'POSITIVE',
    data: [59.76]
  }, {
    name: 'NEGATIVE',
    data: [20.42]
  }, {
    name: 'NEUTRAL',
    data: [19.82]
  }],
    chart: {
    type: 'bar',
    height: 150,
    stacked: true,
    stackType: '100%',
    toolbar:{
    show: false
    }
  },
  plotOptions: {
    bar: {
      horizontal: true,
    },
  },
  stroke: {
    width: 1,
    colors: ['#fff']
  },
  yaxis : {
    show: false,
  },
  xaxis: {
    labels: {
        show: false},
        axisBorder: {
            show: false,
            color: '#2E294E',
            height: 0,
            width: '100%',
            offsetX: 0,
            offsetY: 0
        }
  },
  tooltip: {
    x: {
      show: false,
  },
    y: {
      formatter: function (val) {
        return val + "%"
      }
    }

  },
  fill: {
    opacity: 1
  
  },
  legend: {
    show : false,
    position: 'top',
    horizontalAlign: 'right',
    offsetX: 40
  },
  fill: {
    colors: ['rgb(167, 194, 234)','rgb(234, 181, 167)', 'rgb(224, 224, 224)']
  },
  grid: {
    show: true,
    borderColor: '#2E294E',
    xaxis: {
        lines: {
            show: false
        }
    },   
    yaxis: {
        lines: {
            show: false
        }
    }
  }
  
  };

  var chart = new ApexCharts(document.querySelector("#chart"), options);
  chart.render();
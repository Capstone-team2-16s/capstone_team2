$(document).ready(function(){
    $(window).on('load', function () {
        $('.countdown').on('change',function(){
            console.log($('.countdown').text());

        });
        
            $('.loader').hide(300);
    });

});

$(document).on('change',function(){
    $('.loader').hide(300);
})
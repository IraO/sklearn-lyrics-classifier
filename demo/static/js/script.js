var classificationResponse = function(response){
    sentence = 'This sentence is';
    if (response.is_bad_word) {
        $('#resultPic').attr('src', '/static/img/bad_icon.png')
        sentence += ' <span class="styleResponseBad">Bad</span>';
    }else{
        $('#resultPic').attr('src', '/static/img/good_icon.png')
        sentence += ' <span class="styleResponseGood">Good</span>';
    }
    if (parseFloat(response.score) > 0) {
        sentence += '</br>Score: ' + response.score
    }
    return sentence;
};

$(document).ready(function() {
  $('#input_sentence').click(function() {
    $('#input_sentence').parent().removeClass('has-danger');
    $('#resultPic').attr('src', '');
    $('#responseText p').remove();
    $('small').addClass('invisible');
    $('#input_sentence').val('');
  });

  $('#generateForm').submit(function(e) {
    $('#errorText').hide();
    $('#input_sentence').parent().removeClass('has-danger');
    $('small').addClass('invisible');
    $('#resultPic').attr('src', '');
    $('#responseText p').remove();
    var sentence =  $('#input_sentence').val();
    if ($.trim(sentence) === "") {
        $('#input_sentence').parent().addClass('has-danger');
        $('small').removeClass('invisible');
        return false;
    }
    $('.progress').removeClass('invisible');
    $(".progress-bar").css("width", '70%');
    $(".resLabel").show();
	var checked_classifier = $('input:checked').val();
	var qurl = '/generate';
    $.ajax({
        type: 'POST',
        contentType: 'application/json',
        url: qurl,
        data: JSON.stringify({sentence: sentence, checked_classifier: checked_classifier}),
        success:function(result, status, xhr) {
            $(".progress-bar").css("width", '100%');
            var json = result.results;
            responses = ""
            for(var i=0; i < json.length; i++){
                responses += "<p>" + classificationResponse(json[i]) + "</p>";
            }
            $("#responseText").append(responses);

        },
        error: function(){
            $('#errorText').show();
        },
        complete: function() {
            $(".progress").addClass('invisible');
            $(".resLabel").hide();
        }
    });
    e.preventDefault();
  });
}).ajaxStop(function() {});

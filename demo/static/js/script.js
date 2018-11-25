$(document).ready(function() {
  $('#input_sentence').click(function() {
    $('#input_sentence').parent().removeClass('has-danger');
    $('#responseText').text('');
    $('small').addClass('invisible');
    $('#input_sentence').val('');
  });

  $('#generateForm').submit(function(e) {
    $('#errorText').hide();
    $('#input_sentence').parent().removeClass('has-danger');
    $('small').addClass('invisible');
    $('#responseText').text('');
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
            $("#responseText").text(result.genre);
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

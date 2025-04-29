// JavaScript für den Digitalen Buchungsassistenten

$(document).ready(function() {
    // System-Initialisierung
    $('#init-btn').on('click', function() {
        let $initBtn = $(this);
        let $initStatus = $('#init-status');
        
        // Button deaktivieren und Ladeanzeige einblenden
        $initBtn.prop('disabled', true);
        $initStatus.html('<div class="alert alert-info">System wird initialisiert... <span class="loading"></span></div>');
        
        // AJAX-Anfrage an den Server
        $.ajax({
            url: '/initialize',
            method: 'POST',
            success: function(response) {
                if (response.success) {
                    $initStatus.html('<div class="alert alert-success">' + response.message + '</div>');
                    $('#init-section').fadeOut(1000); // Ausblenden nach erfolgreicher Initialisierung
                } else {
                    $initStatus.html('<div class="alert alert-danger">' + response.message + '</div>');
                    $initBtn.prop('disabled', false);
                }
            },
            error: function() {
                $initStatus.html('<div class="alert alert-danger">Fehler bei der Verbindung zum Server</div>');
                $initBtn.prop('disabled', false);
            }
        });
    });
    
    // Formular-Absendung
    $('#input-form').on('submit', function(e) {
        e.preventDefault();
        
        let inputText = $('#input-text').val().trim();
        if (!inputText) return;
        
        let $sendBtn = $('#send-btn');
        let $outputContainer = $('#output-container');
        
        // Button deaktivieren und Ladeanzeige einblenden
        $sendBtn.prop('disabled', true);
        $outputContainer.html('<p>Verarbeite Anfrage... <span class="loading"></span></p>');
        
        // AJAX-Anfrage an den Server
        $.ajax({
            url: '/process',
            method: 'POST',
            data: {
                input_text: inputText
            },
            success: function(response) {
                if (response.success) {
                    $outputContainer.html(response.output);
                    updateHistoryList(response.input);
                } else {
                    $outputContainer.html('<div class="alert alert-danger">' + response.message + '</div>');
                }
                $sendBtn.prop('disabled', false);
            },
            error: function() {
                $outputContainer.html('<div class="alert alert-danger">Fehler bei der Verbindung zum Server</div>');
                $sendBtn.prop('disabled', false);
            }
        });
    });
    
    // Historie-Einträge klickbar machen
    $(document).on('click', '.history-item', function() {
        let text = $(this).text();
        $('#input-text').val(text);
    });
    
    // Aktualisierung der Historie
    function updateHistoryList(newInput) {
        $.ajax({
            url: '/history',
            method: 'GET',
            success: function(response) {
                let $historyList = $('#history-list');
                $historyList.empty();
                
                response.history.forEach(function(item) {
                    $historyList.append('<li class="list-group-item history-item">' + item + '</li>');
                });
            }
        });
    }
});

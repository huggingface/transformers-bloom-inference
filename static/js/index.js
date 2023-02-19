const textGenInput = document.getElementById('text-input');
const clickButton = document.getElementById('submit-button');

const temperatureSlider = document.getElementById('temperature-slider');
const temperatureTextBox = document.getElementById('temperature-textbox')

const top_pSlider = document.getElementById('top_p-slider');
const top_pTextBox = document.getElementById('top_p-textbox');

const top_kSlider = document.getElementById('top_k-slider');
const top_kTextBox = document.getElementById('top_k-textbox');

const repetition_penaltySlider = document.getElementById('repetition_penalty-slider');
const repetition_penaltyTextBox = document.getElementById('repetition_penalty-textbox');

const max_new_tokensInput = document.getElementById('max-new-tokens-input');

const textLogOutput = document.getElementById('log-output');

function get_temperature() {
    return parseFloat(temperatureSlider.value);
}

temperatureSlider.addEventListener('input', async (event) => {
    temperatureTextBox.innerHTML = "temperature = " + get_temperature();
});

function get_top_p() {
    return parseFloat(top_pSlider.value);
}

top_pSlider.addEventListener('input', async (event) => {
    top_pTextBox.innerHTML = "top_p = " + get_top_p();
});

function get_top_k() {
    return parseInt(top_kSlider.value);
}

top_kSlider.addEventListener('input', async (event) => {
    top_kTextBox.innerHTML = "top_k = " + get_top_k();
});

function get_repetition_penalty() {
    return parseFloat(repetition_penaltySlider.value);
}

repetition_penaltySlider.addEventListener('input', async (event) => {
    repetition_penaltyTextBox.innerHTML = "repetition_penalty = " + get_repetition_penalty();
});

function get_max_new_tokens() {
    return parseInt(max_new_tokensInput.value);
}

clickButton.addEventListener('click', async (event) => {
    clickButton.textContent = 'Processing'
    clickButton.disabled = true;

    var jsonPayload = {
        text: [textGenInput.value],
        temperature: get_temperature(),
        top_k: get_top_k(),
        top_p: get_top_p(),
        max_new_tokens: get_max_new_tokens(),
        repetition_penalty: get_repetition_penalty(),
        do_sample: true,
        remove_input_from_output: true
    };

    if (jsonPayload.temperature == 0) {
        jsonPayload.do_sample = false;
    }

    console.log(jsonPayload);

    $.ajax({
        url: '/generate/',
        type: 'POST',
        contentType: "application/json; charset=utf-8",
        data: JSON.stringify(jsonPayload),
        headers: { 'Access-Control-Allow-Origin': '*' },
        success: function (response) {
            var input_text = textGenInput.value;

            if ("text" in response) {
                if (response.is_encoder_decoder) {
                    textLogOutput.value = response.text[0] + '\n\n';
                } else {
                    textGenInput.value = input_text + response.text[0];
                    textLogOutput.value = '';
                }

                textLogOutput.value += 'total_time_taken = ' + response.total_time_taken + "\n";
                textLogOutput.value += 'num_generated_tokens = ' + response.num_generated_tokens + "\n";
                textLogOutput.style.backgroundColor = "lightblue";
            } else {
                textLogOutput.value = 'total_time_taken = ' + response.total_time_taken + "\n";
                textLogOutput.value += 'error: ' + response.message;
                textLogOutput.style.backgroundColor = "#D65235";
            }

            clickButton.textContent = 'Submit';
            clickButton.disabled = false;
        },
        error: function (error) {
            console.log(JSON.stringify(error, null, 2));
            clickButton.textContent = 'Submit'
            clickButton.disabled = false;
        }
    });
});

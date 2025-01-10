let video = document.getElementById('webcam');
let startButton = document.getElementById('start_button');
let rollNumberDiv = document.getElementById('roll_number');
let webcamStream;

startButton.addEventListener('click', function () {
    if (video.srcObject) {
        // Turn off the webcam if it's already on
        stopWebcam();
        rollNumberDiv.innerText = "";
        startButton.innerText = "Turn On Webcam";
    } else {
        // Turn on the webcam
        startWebcam();
        startButton.innerText = "Turn Off Webcam";
    }
});

// Start webcam stream
function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            video.srcObject = stream;
            webcamStream = stream;
            captureFace(); // Start capturing images
        })
        .catch(function (err) {
            console.error("Error accessing webcam: " + err);
        });
}

// Stop webcam stream
function stopWebcam() {
    if (webcamStream) {
        let tracks = webcamStream.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
    }
}

// Capture a frame from the webcam and send it to the Flask backend for prediction
function captureFace() {
    setInterval(function () {
        let canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        let ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert canvas to a Blob and send it to the backend
        canvas.toBlob(function (blob) {
            let formData = new FormData();
            formData.append('image', blob);

            $.ajax({
                url: '/predict',
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                success: function (response) {
                    if (response.roll_number) {
                        rollNumberDiv.innerText = 'Roll Number: ' + response.roll_number;
                    } else {
                        rollNumberDiv.innerText = 'No face detected';
                    }
                },
                error: function (err) {
                    console.error('Error during prediction:', err);
                }
            });
        });
    }, 1000); // Capture every second
}

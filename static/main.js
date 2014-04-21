var socket = new WebSocket('ws://127.0.0.1:8000/webcam');
var live = document.querySelector('video#live');
var draw = document.querySelector('canvas#draw');
var target = document.querySelector('img#target');


navigator.getUserMedia = navigator.getUserMedia 
						|| navigator.webkitGetUserMedia 
						|| navigator.mozGetUserMedia;

var constraints  = {
	video: {
		mandatory: {
			maxWidth: 320,
			maxHeight: 240
		}
	}
};

function getMedia() {
	if(!!window.stream) {
		live.src = null;
		stream.stop();
	}
	navigator.getUserMedia(
		constraints, 
		function(stream) {
			window.stream = stream;
			live.src = window.URL.createObjectURL(stream);
			live.play();
		},
		function(error) {
			alert('navigator.getUserMedia error: ', error);
		});
}

function record() {
	draw.getContext('2d').drawImage(live, 0, 0, 320, 240);
	var data = draw.toDataURL('image/jpeg', 1.0);
	//socket.send(data);
	socket.send(dataURItoBlob(data));
}

function dataURItoBlob(dataURI) {
	var byteString;
	if (dataURI.split(',')[0].indexOf('base64') >= 0)
    	byteString = atob(dataURI.split(',')[1]);
	else
    	byteString = unescape(dataURI.split(',')[1]);
	var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0]
	var ab = new ArrayBuffer(byteString.length);
	var ia = new Uint8Array(ab);
	for (var i = 0; i < byteString.length; i++) {
		ia[i] = byteString.charCodeAt(i);
	}
	var blob = new Blob([ab], { type: mimeString });
	return blob;
}

(function() {
	getMedia();
	setInterval(record, 500);
	socket.onmessage = function(event) {
		//var url = window.URL.createObjectURL(event.data);
		//target.onload = function() {
		//	window.URL.revokeObjectURL(url);
		//};
		//target.src = url;
		target.src = event.data;
	}
	//setTimeout($.get('/info2'), 10000)
})();

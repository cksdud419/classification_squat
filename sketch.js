// 스쿼트 버전!
// 모션 추정 코드
let video;
let countSound;

let poseNet;
let pose;
let skeleton;
let brain;

let pastState;
let curState;
let count = 0;
let countName = 'None';
let inp;
let angle;

let videoURL;
let videoPlayer;
let cameraActive = true; // 카메라 활성화 상태를 나타내는 변수 추가
let averageScore = 0.0;

function preload() {
  motionList = loadJSON('model/model_meta.json');
} // 저장된 동작 리스트를 알기 위해 json파일을 프로그램 실행 전 미리 로드

function setup() {
  createCanvas(384, 512);
  video = createCapture(VIDEO);
  video.size(width, height);
  video.hide();

  countSound = loadSound('sound/check.wav');
  countName = motionList.outputs[0].uniqueValues[1];

  // let poseOptions = {
  //   architecture: 'MobileNetV1',
  //   imageScaleFactor: 0.3,
  //   outputStride: 16,
  //   flipHorizontal: false,
  //   minConfidence: 0.5
  // }

  poseNet = ml5.poseNet(video);
  poseNet.on('pose', extraction);

  let options = {
    inputs: 34,
    outputs: [],
    task: 'classification',
    debug: true
  };
  brain = ml5.neuralNetwork(options);

  const modelInfo = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin'
  };

  brain.load(modelInfo, classification);

  // 화면 녹화 버튼
  recordingBtn = createButton('Record');
  recordingBtn.position(10, 10);
  recordingBtn.mousePressed(toggleRecording);

  playBtn = createButton('Play');
  playBtn.position(recordingBtn.x + recordingBtn.width + 10, recordingBtn.y);
  playBtn.mousePressed(playRecording);
  playBtn.hide(); // 일단 숨겨둠
}

function toggleRecording() {
  if (recording) {
    recordingBtn.html('Record');
    mediaRecorder.stop();
    playBtn.show(); // 녹화 중지 시 playBtn 표시
  } else {
    recordingBtn.html('Stop');
    startRecording();
    playBtn.hide(); // 녹화 시작 시 playBtn 숨김
  }
}

let chunks = [];
let mediaRecorder;
let recording = false;

function playRecording() {
  noLoop();
  cameraActive = false;
  pose = null;
  if (videoURL) {
    videoPlayer = createVideo([videoURL]);
    videoPlayer.loop();
    videoPlayer.show();
    const canvasPosition = select('canvas').position();
    videoPlayer.position(canvasPosition.x, canvasPosition.y);
    videoPlayer.size(width, height);

    // "Back" 버튼 생성
    backBtn = createButton('Back');
    backBtn.position(canvasPosition.x + width - 60, canvasPosition.y + 10);
    backBtn.mousePressed(stopPlayback);
  }
}

function stopPlayback() {
  if (videoPlayer) {
    videoPlayer.pause();
    videoPlayer.hide();
    videoPlayer.remove();
    backBtn.remove();
    loop();
    cameraActive = true;
  }
}

function startRecording() {
  if (typeof MediaRecorder === 'undefined') {
    return;
  }

  chunks = [];
  const stream = document.querySelector('canvas').captureStream();
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = handleDataAvailable;
  mediaRecorder.onstop = handleStop;
  mediaRecorder.start();
  recording = true;
}

function handleDataAvailable(event) {
  if (event.data.size > 0) {
    chunks.push(event.data);
  }
}

function handleStop() {
  const blob = new Blob(chunks, { type: 'video/webm' });
  videoURL = URL.createObjectURL(blob);
  chunks = [];
  recording = false;
  playBtn.show(); // 녹화 중지 후 playBtn 표시
}

function classification() {
  if (pose) {
    let inputs = [];
    averageScore = 0.0;
    for (let i = 0; i < pose.keypoints.length; i++) {
      inputs.push(pose.keypoints[i].position.x);
      inputs.push(pose.keypoints[i].position.y);
      averageScore += pose.keypoints[i].score;
    }
    averageScore /= 17.0;
    if (averageScore >= 0.8) {
      brain.classify(inputs, classifyResult);
    }
    else {
      setTimeout(classification, 100);
    }
  }
  else {
    setTimeout(classification, 100); //포즈가 인식되지 않았을 때 100밀리초마다 포즈추정 반복
  }
}

let pastStateTime = 0;
const stateChangeThreshold = 700; // 상태 변경 임계값
let isCounting = false; // 횟수를 세고 있는지 여부를 나타내는 변수

function classifyResult(error, results) {
  pastState = curState;
  if (error) {
    console.error(error);
    return;
  }
  curState = results[0].label;

  if (countName == curState && !isCounting && pastState == 'Default') {
    // countName과 curState가 일치하고, 이전에 'Default'자세였으며 횟수를 세고 있지 않은 경우
    pastStateTime = Date.now(); // 현재 시간으로 초기화
    isCounting = true;
  }
  else if (isCounting && curState != countName && curState != 'Default') {
    isCounting = false;
  }
  else if (countName != curState && isCounting && (Date.now() - pastStateTime) >= stateChangeThreshold) {
    // countName과 curState가 다르고, 횟수를 세고 있으며, 임계값 이상 시간이 경과한 경우
    count++;
    countSound.play();
    console.log('Exercise:', countName, ':', count);
    isCounting = false; // 횟수 세는 상태를 종료
  }

  classification(); // 반복 포즈 추정
}

function extraction(poses) {
  if (cameraActive) {
    if (poses.length > 0) {
      pose = poses[0].pose;
      skeleton = poses[0].skeleton;
    }
    else pose = null;
  }
}

function draw() {
  //push();
  translate(video.width, 0);
  scale(-1, 1);
  image(video, 0, 0, width, height);

  if (pose) {
    for (let i = 0; i < pose.keypoints.length; i++) {
      if(pose.keypoints[i].score > 0.5) {
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        fill(0, 0, 255);
        ellipse(x, y, 16, 16);
        strokeWeight(2);
        stroke(255);
      }
    }

    for (let i = 0; i < skeleton.length; i++) {
      if(pose.keypoints[i].score > 0.5){
        let a = skeleton[i][0];
        let b = skeleton[i][1];
        strokeWeight(2);
        stroke(255);
        line(a.position.x, a.position.y, b.position.x, b.position.y); 
      }
    }
  }

  if(averageScore <= 0.7) {
    scale(-1,1);
    textSize(20);
    fill(0,0,0);
    stroke(255);
    strokeWeight(3);
    textAlign(RIGHT, TOP);
    text('전신이 카메라에 보이도록 해주세요.', -30, 50);
    scale(-1,1);
  }
  //pop();
}

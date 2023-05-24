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
}

function toggleRecording() {
  if (recording) {
    recordingBtn.html('Record');
    mediaRecorder.stop();
  } else {
    recordingBtn.html('Stop');
    startRecording();
  }
}

let chunks = [];
let mediaRecorder;
let recording = false;

function startRecording() {
  if (typeof MediaRecorder === 'undefined') {
    console.log('MediaRecorder API is not supported.');
    return;
  }

  chunks = [];
  const stream = document.querySelector('canvas').captureStream();
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = handleDataAvailable;
  mediaRecorder.onstop = handleStop;
  mediaRecorder.start();
  recording = true;
  console.log('Recording started.');
}

function handleDataAvailable(event) {
  if (event.data.size > 0) {
    chunks.push(event.data);
  }
}

function handleStop() {
  const blob = new Blob(chunks, { type: 'video/webm' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  document.body.appendChild(a);
  a.href = url;
  a.download = 'recorded_video.webm';
  a.click();
  window.URL.revokeObjectURL(url);
  chunks = [];
  recording = false;
  console.log('Recording stopped.');
}


// 각도는 양 옆 점(p1, p3) 사이에 끼인 점(p2)의 사이각을 구함
// ex = getAngle(pose.leftShoulder, pose.leftElbow, pose.leftWrist)
function getAngle(p1, p2, p3) {
  if (
    p1.confidence < 0.65 ||
    p2.confidence < 0.65 ||
    p3.confidence < 0.65
  )
    return null; // 키포인트 인식이 제대로 되지 않으면 각도 계산을 하지 않음

  rad =
    Math.atan2(p3.y - p2.y, p3.x - p2.x) -
    Math.atan2(p1.y - p2.y, p1.x - p2.x); // atan2 함수를 이용하여 라디안 계산
  deg = rad * (180 / Math.PI); // 라디안을 각도로 환산하기 위해 180/파이 곱
  vecdeg = 360 - Math.abs(deg); // 반대쪽 각

  if (deg < 0) return Math.abs(deg); // 음각일 때 절대값으로 반환
  else return deg;
}

function angleUpdate() {
  // 추가적인 각도 계산이 필요하면 추가할 것
  leftElbow = getAngle(
    pose.leftShoulder,
    pose.leftElbow,
    pose.leftWrist
  ); //좌측 팔꿈치각도
  rightElbow = getAngle(
    pose.rightShoulder,
    pose.rightElbow,
    pose.rightWrist
  ); //우측 팔꿈치각도

  leftKnee = getAngle(pose.leftHip, pose.leftKnee, pose.leftAnkle); //좌측 무릎
  rightKnee = getAngle(pose.rightHip, pose.rightKnee, pose.rightAnkle); //우측 무릎

  leftShoulder = getAngle(
    pose.leftElbow,
    pose.leftShoulder,
    pose.leftHip
  ); //좌측 어깨(겨드랑이)
  rightShoulder = getAngle(
    pose.rightElbow,
    pose.rightShoulder,
    pose.rightHip
  ); //우측 어깨

  return {
    leftElbow: leftElbow,
    rightElbow: rightElbow,
    leftKnee: leftKnee,
    rightKnee: rightKnee,
    leftShoulder: leftShoulder,
    rightShoulder: rightShoulder
  };
}

function classification() {
  if (pose) {
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      inputs.push(pose.keypoints[i].position.x);
      inputs.push(pose.keypoints[i].position.y);
    }
    brain.classify(inputs, classifyResult);
  } else {
    setTimeout(classification, 100); //포즈가 인식되지 않았을 때 100밀리초마다 포즈추정 반복
  }
}

function classifyResult(error, results) {
  pastState = curState;
  if (error) {
    console.error(error);
    return;
  }
  curState = results[0].label;

  if (countName == curState && pastState != curState) {
    count++;
    countSound.play();
  }

  classification(); //반복 포즈추정
}

function extraction(poses) {
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
  }
}

function draw() {
  //push();
  translate(video.width, 0);
  scale(-1, 1);
  image(video, 0, 0, width, height);

  if (pose) {
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0, 255, 0);
      ellipse(x, y, 16, 16);
    }

    for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(2);
      stroke(255);
      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
  }
  //pop();
}

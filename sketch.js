let video;
let poseNet;
let pose;
let skeleton;
let brain;

let count = 0;
let countName = 'None';
let angle;
let isRecording = false;
let recorder;
let chunks = [];
let frontCamera = true;

function setup() {
  createCanvas(384, 512);

  // 카메라 방향에 따라 비디오 캡처 객체를 생성합니다
  if (frontCamera) {
    video = createCapture({
      audio: false,
      video: {
        facingMode: "user"
      }
    });
  } else {
    video = createCapture({
      audio: false,
      video: {
        facingMode: {
          exact: "environment"
        }
      }
    });
  }
  
  video.size(width, height);
  video.hide();

  poseNet = ml5.poseNet(video, () => {
    console.log('PoseNet 모델 로드 완료');
  });
  poseNet.on('pose', extraction);

  let options = {
    inputs: 34,
    outputs: [],
    task: 'classification',
    debug: true,
  };
  brain = ml5.neuralNetwork(options);

  // 모델 로드 및 분류 함수 호출

  resetBtn = createButton('Reset');
  resetBtn.mousePressed(selectReset);

  sel = createSelect();
  sel.option('None');
  for (let i = 0; i < motionList.outputs[0].uniqueValues.length; i++) {
    sel.option(motionList.outputs[0].uniqueValues[i]);
  }
  sel.changed(selectCount);

  selectAngle = createCheckbox('각도 보기', false);

  recordBtn = createButton('Record');
  recordBtn.mousePressed(startRecording);

  stopBtn = createButton('Stop');
  stopBtn.mousePressed(stopRecording);

  // 카메라 방향 전환 버튼 생성
  switchCameraBtn = createButton('Switch Camera');
  switchCameraBtn.mousePressed(switchCamera);
}

function startRecording() {
  isRecording = true;
  chunks = [];
  recorder = new p5.MediaRecorder();
  recorder.start();
}

function stopRecording() {
  isRecording = false;
  recorder.stop();
  recorder.save('recordedVideo.webm');
}

function switchCamera() {
  frontCamera = !frontCamera;

  // 현재 비디오 캡처 객체를 중지하고 제거합니다
  video.stop();
  video.remove();

  // 카메라 방향에 따라 새로운 비디오 캡처 객체를 생성합니다
  if (frontCamera) {
    video = createCapture({
      audio: false,
      video: {
        facingMode: "user"
      }
    });
  } else {
    video = createCapture({
      audio: false,
      video: {
        facingMode: {
          exact: "environment"
        }
      }
    });
  }
  
  video.size(width, height);
  video.hide();

  // 새로운 비디오 캡처 객체를 PoseNet에 연결합니다
  poseNet.video = video.elt;
  poseNet.video.play();
}

function draw() {
  image(video, 0, 0, width, height);

  if (pose) {
    drawSkeleton();
    drawPose();
  }

  if (countName !== 'None') {
    fill(255);
    textSize(20);
    textAlign(RIGHT, TOP);
    text(`동작: ${countName}\n${count}회`, width - 30, 30);
  }

  if (selectAngle.checked()) {
    fill(255);
    textSize(17);
    textAlign(RIGHT, TOP);
    let angleText = `무릎 각도\n좌: ${angle.leftKnee}\n우: ${angle.rightKnee}\n`
      + `어깨 각도\n좌: ${angle.leftShoulder}\n우: ${angle.rightShoulder}\n`
      + `팔꿈치 각도\n좌: ${angle.leftElbow}\n우: ${angle.rightElbow}`;
    text(angleText, width - 30, 30);
  }

  if (isRecording) {
    recorder.record(canvas);
  }
}

function drawSkeleton() {
  for (let i = 0; i < skeleton.length; i++) {
    let a = skeleton[i][0];
    let b = skeleton[i][1];
    stroke(255);
    line(a.position.x, a.position.y, b.position.x, b.position.y);
  }
}

function drawPose() {
  for (let i = 0; i < pose.keypoints.length; i++) {
    let x = pose.keypoints[i].position.x;
    let y = pose.keypoints[i].position.y;
    fill(0, 255, 0);
    ellipse(x, y, 16);
  }
}

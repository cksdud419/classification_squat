let video;
let resetSound;
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

let startRecordingBtn;
let stopRecordingBtn;
let switchCameraBtn;
let recorder;
let isRecording = false;
let recordedChunks = [];

function preload() {
  motionList = loadJSON('model/model_meta.json');
}

function setup() {
  createCanvas(384, 512);

  video = createCapture({
    audio: false,
    video: {
      facingMode: {
        exact: "environment"
      }
    }
  });
  video.size(width, height);
  video.hide();
  video.show(); // 비디오 요소 표시

  countSound = loadSound('sound/check.wav');
  resetSound = loadSound('sound/reset.mp3');

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
    weights: 'model/model.weights.bin',
  };

  brain.load(modelInfo, classification);

  resetBtn = createButton('Reset');
  resetBtn.mousePressed(selectReset);

  // 동작 선택
  sel = createSelect();
  sel.option('None');
  for (let i = 0; i < motionList.outputs[0].uniqueValues.length; i++) {
    sel.option(motionList.outputs[0].uniqueValues[i]);
  }
  sel.changed(selectCount);

  // 각도 선택
  selectAngle = createCheckbox('각도 보기', false);

  // Create buttons
  startRecordingBtn = createButton("Start Recording");
  startRecordingBtn.mousePressed(startRecording);

  stopRecordingBtn = createButton("Stop Recording");
  stopRecordingBtn.mousePressed(stopRecording);
  stopRecordingBtn.attribute("disabled", "true");

  switchCameraBtn = createButton("Switch Camera");
  switchCameraBtn.mousePressed(switchCamera);
}

function selectReset() {
  count = 0;
  resetSound.play();
}

function selectCount() {
  countName = sel.value();
  count = 0;
}

function startRecording() {
  isRecording = true;
  startRecordingBtn.attribute("disabled", "true");
  stopRecordingBtn.removeAttribute("disabled");
  recordedChunks = [];
  recorder.start();
  console.log("Recording started.");
}

function stopRecording() {
  isRecording = false;
  startRecordingBtn.removeAttribute("disabled");
  stopRecordingBtn.attribute("disabled", "true");
  recorder.stop();
  console.log("Recording stopped.");
  const blob = new Blob(recordedChunks, { type: "video/webm" });
  save(blob, "recorded_video.webm");
}

function switchCamera() {
  if (video.facingMode === "user") {
    video.facingMode = "environment";
  } else {
    video.facingMode = "user";
  }
  video.remove();
  video = createCapture({
    audio: false,
    video: {
      facingMode: video.facingMode
    }
  });
  video.size(width, height);
  video.hide();
  video.show(); // 비디오 요소 표시
  poseNet = ml5.poseNet(video);
  poseNet.on('pose', extraction);
}

function setupRecorder(stream) {
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = handleDataAvailable;
  recorder.onstop = handleRecordingStop;
}

function handleDataAvailable(event) {
  if (event.data.size > 0) {
    recordedChunks.push(event.data);
  }
}

function handleRecordingStop(event) {
  console.log("Recording saved.");
}

function extraction(data) {
  if (data.length > 0) {
    pose = data[0].pose;
    skeleton = data[0].skeleton;
  } else {
    pose = null;
  }
}

function posedraw() {
  for (let i = 0; i < pose.keypoints.length; i++) {
    let x = pose.keypoints[i].position.x;
    let y = pose.keypoints[i].position.y;
    fill(0, 0, 255);
    ellipse(x, y, 16);
  }
  for (let i = 0; i < skeleton.length; i++) {
    let a = skeleton[i][0];
    let b = skeleton[i][1];
    strokeWeight(4);
    stroke(255);
    line(a.position.x, a.position.y, b.position.x, b.position.y);
  }
}

function angleUpdate() {
  leftElbow = getAngle(pose.leftShoulder, pose.leftElbow, pose.leftWrist);
  rightElbow = getAngle(pose.rightShoulder, pose.rightElbow, pose.rightWrist);

  leftKnee = getAngle(pose.leftHip, pose.leftKnee, pose.leftAnkle);
  rightKnee = getAngle(pose.rightHip, pose.rightKnee, pose.rightAnkle);

  leftShoulder = getAngle(pose.leftElbow, pose.leftShoulder, pose.leftHip);
  rightShoulder = getAngle(pose.rightElbow, pose.rightShoulder, pose.rightHip);

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
    setTimeout(classification, 100);
  }
}

function classifyResult(error, results) {
  if (error) {
    console.error(error);
    return;
  }
  curState = results[0].label;

  if (pose) {
    let angles = angleUpdate();
    if (selectAngle.checked()) {
      showAngle(angles);
    }

    if (curState != pastState) {
      count = 0;
      countSound.play();
    }

    if (curState == countName) {
      count++;
    }
  }

  pastState = curState;
  classification();
}

function showAngle(angles) {
  let angleSize = 20;
  textSize(angleSize);
  fill(255, 0, 0);
  text("Left Elbow: " + nf(angles.leftElbow, 2, 2), width / 2, height - angleSize * 3);
  text("Right Elbow: " + nf(angles.rightElbow, 2, 2), width / 2, height - angleSize * 2);
  text("Left Knee: " + nf(angles.leftKnee, 2, 2), width / 2, height - angleSize);
  text("Right Knee: " + nf(angles.rightKnee, 2, 2), width / 2, height);
  text("Left Shoulder: " + nf(angles.leftShoulder, 2, 2), width / 2, angleSize);
  text("Right Shoulder: " + nf(angles.rightShoulder, 2, 2), width / 2, angleSize * 2);
}

function draw() {
  background(0);
  image(video, 0, 0, width, height);
  if (pose) {
    posedraw();
  }
  fill(255);
  noStroke();
  textSize(30);
  textAlign(CENTER, CENTER);
  text(count, width / 2, height / 2);
}

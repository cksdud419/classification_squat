let video;
let recorder;
let isRecording = false;

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

function preload(){
  motionList = loadJSON('model/model_meta.json');
}

function setup() {
  createCanvas(384, 512);

  video = createCapture(VIDEO);
  video.size(width, height);
  video.hide();

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
    weights: 'model/model.weights.bin'
  };

  brain.load(modelInfo, classification);

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
  recordBtn.mousePressed(toggleRecording);
}

function toggleRecording() {
  if (isRecording) {
    recorder.stop();
    isRecording = false;
    recordBtn.html('Record');
  } else {
    recorder = new p5.SoundRecorder();
    recorder.setInput(video);

    let soundFile = new p5.SoundFile();
    recorder.record(soundFile);

    isRecording = true;
    recordBtn.html('Stop');
  }
}

function selectReset() {
  count = 0;
  resetSound.play();
}

function selectCount() {
  countName = sel.value();
  count = 0;
}

function getAngle(p1, p2, p3) {
  if (p1.confidence < 0.65 || p2.confidence < 0.65 || p3.confidence < 0.65)
    return null;

  rad = Math.atan2(p3.y - p2.y, p3.x - p2.x) - Math.atan2(p1.y - p2.y, p1.x - p2.x);
  deg = rad * (180 / Math.PI);
  vecdeg = 360 - Math.abs(deg);

  if (deg < 0)
    return Math.abs(deg);
  else
    return deg;
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
  }
}

function extraction(results) {
  if (results.length > 0) {
    pose = results[0].pose;
    skeleton = results[0].skeleton;

    if (pose) {
      curState = angleUpdate();
      if (pastState && curState) {
        angle = [
          pastState.leftElbow - curState.leftElbow,
          pastState.rightElbow - curState.rightElbow,
          pastState.leftKnee - curState.leftKnee,
          pastState.rightKnee - curState.rightKnee,
          pastState.leftShoulder - curState.leftShoulder,
          pastState.rightShoulder - curState.rightShoulder
        ];
      }
      pastState = curState;
    }
  }
}

function classification() {
  let inputs = {
    leftElbow: angle[0],
    rightElbow: angle[1],
    leftKnee: angle[2],
    rightKnee: angle[3],
    leftShoulder: angle[4],
    rightShoulder: angle[5]
  };

  brain.classify(inputs, gotResults);
}

function gotResults(error, results) {
  if (error) {
    console.error(error);
    return;
  }

  let confidenceThreshold = 0.8;
  let confidence = results[0].confidence.toFixed(2);
  let label = results[0].label;

  if (confidence > confidenceThreshold) {
    if (countName == label) {
      if (count == 0) {
        countSound.play();
      }
      count += 1;
    }
  }

  classification();
}

function draw() {
  background(0);
  image(video, 0, 0, width, height);

  if (pose) {
    drawSkeleton();
    drawPose();
  }

  fill(255);
  textSize(16);
  text(countName + ' Count: ' + count, 10, height - 20);

  if (selectAngle.checked()) {
    fill(255);
    textSize(16);
    text('Left Elbow: ' + (angle ? angle[0].toFixed(2) : '-'), 10, 20);
    text('Right Elbow: ' + (angle ? angle[1].toFixed(2) : '-'), 10, 40);
    text('Left Knee: ' + (angle ? angle[2].toFixed(2) : '-'), 10, 60);
    text('Right Knee: ' + (angle ? angle[3].toFixed(2) : '-'), 10, 80);
    text('Left Shoulder: ' + (angle ? angle[4].toFixed(2) : '-'), 10, 100);
    text('Right Shoulder: ' + (angle ? angle[5].toFixed(2) : '-'), 10, 120);
  }

  if (isRecording) {
    fill(255, 0, 0);
    noStroke();
    ellipse(20, 20, 10, 10);
  }
}

function drawPose() {
  for (let i = 0; i < pose.keypoints.length; i++) {
    let x = pose.keypoints[i].position.x;
    let y = pose.keypoints[i].position.y;
    fill(0, 255, 0);
    ellipse(x, y, 16, 16);
  }
}

function drawSkeleton() {
  for (let i = 0; i < skeleton.length; i++) {
    let a = skeleton[i][0];
    let b = skeleton[i][1];
    strokeWeight(2);
    stroke(255);
    line(a.position.x, a.position.y, b.position.x, b.position.y);
  }
}

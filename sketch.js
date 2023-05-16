//ml5 포즈넷 명령어 및 메소드 참고 - https://learn.ml5js.org/#/reference/posenet
//ml5 neuralNetwork - https://learn.ml5js.org/#/reference/neural-network
//p5 레퍼런스 - https://p5js.org/ko/reference/
//모션 추정 코드

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

function preload(){
  motionList = loadJSON('model/model_meta.json');
}  //저장된 동작 리스트를 알기 위해 json파일을 프로그램 실행 전 미리 로드

function setup() {
  //createCanvas(640, 480);
  //createCanvas(480, 640);
  //createCanvas(320, 240);
  //createCanvas(240, 320);
  createCanvas(1440, 3200);
  


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
  }
  brain = ml5.neuralNetwork(options);
  
  const modelInfo = {
  model: 'model/model.json',
  metadata: 'model/model_meta.json',
  weights: 'model/model.weights.bin',
  };
  
  brain.load(modelInfo, classification);
  
  //resetBtn = createButton('Reset');
  //resetBtn.mousePressed(selectReset);
  
  // 동작 선택
  sel = createSelect();
  sel.option('None');
  for(let i = 0; i < motionList.outputs[0].uniqueValues.length; i++) {
    sel.option(motionList.outputs[0].uniqueValues[i]);  }
  sel.changed(selectCount);
  
  // 각도 선택
  selectAngle = createCheckbox('각도 보기', false);
}




function selectReset() {
  count = 0;
  resetSound.play();
  console.log('횟수 카운트 초기화');
}

function selectCount() {
  countName = sel.value();
  count = 0;
  if(countName != 'None')
    console.log(countName, ': 동작 횟수를 측정');
}

//각도는 양 옆 점(p1, p3) 사이에 끼인 점(p2)의 사이각을 구함
//ex = getAngle(pose.leftShoulder, pose.leftElbow, pose.leftWrist)
function getAngle(p1,p2,p3){
  if(p1.confidence < 0.65 || p2.confidence < 0.65 || p3.confidence < 0.65)
    return null;  // 키포인트 인식이 제대로 되지 않으면 각도 계산을 하지 않음
  
  rad = Math.atan2(p3.y - p2.y, p3.x - p2.x) - Math.atan2(p1.y - p2.y, p1.x - p2.x) //atan2 함수를 이용하여 라디안 계산  
  deg = rad * (180 / Math.PI) // 라디안을 각도로 환산하기 위해 180/파이 곱
  vecdeg = 360-Math.abs(deg); // 반대쪽 각
  
  if(deg < 0)
    return Math.abs(deg);     // 음각일 때 절대값으로 반환
  else
    return deg;
}

function angleUpdate(){
  // 추가적인 각도 계산이 필요하면 추가할 것
  leftElbow = getAngle(pose.leftShoulder, pose.leftElbow, pose.leftWrist);      //좌측 팔꿈치각도
  rightElbow = getAngle(pose.rightShoulder, pose.rightElbow, pose.rightWrist);  //우측 팔꿈치각도
  
  leftKnee = getAngle(pose.leftHip, pose.leftKnee, pose.leftAnkle);      //좌측 무릎
  rightKnee = getAngle(pose.rightHip, pose.rightKnee, pose.rightAnkle);  //우측 무릎  
  
  leftShoulder = getAngle(pose.leftElbow, pose.leftShoulder, pose.leftHip);      //좌측 어깨(겨드랑이)
  rightShoulder = getAngle(pose.rightElbow, pose.rightShoulder, pose.rightHip);  //우측 어깨
  
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
  if(pose){
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      inputs.push(pose.keypoints[i].position.x);
      inputs.push(pose.keypoints[i].position.y);  
    }
    brain.classify(inputs, classifyResult);
  }
  else{
    setTimeout(classification, 100);  //포즈가 인식되지 않았을 때 100밀리초마다 포즈추정 반복
  }
}

function classifyResult(error, results) {
  pastState = curState;
  if(results[0].confidence > 0.95){  //정확도가 95%이상일 때만 결과 출력
    curState = results[0].label;
  }
  
  if(curState != pastState){
    console.log(results[0].label, results[0].confidence);
    if(curState == countName){   //카운트
      count++;
      countSound.play();
      console.log(countName, ':', count);
    }
  }
  
  classification();
}

function extraction(data) {
  if (data.length > 0) {          //포즈를 찾으면 배열의 길이를 확인
    pose = data[0].pose;           //배열에서 첫 번째 포즈를 찾으면 전역변수에 저장-키포인트
    skeleton = data[0].skeleton;   //골격도 마찬가지
  }
  else pose = null;
}

function posedraw() {
  for (let i = 0; i < pose.keypoints.length; i++) {
    let x = pose.keypoints[i].position.x;
    let y = pose.keypoints[i].position.y;
    fill(0,0,255);
    ellipse(x, y, 16);
  }
  for (let i = 0; i < skeleton.length; i++){
    let a = skeleton[i][0];
    let b = skeleton[i][1];
    strokeWeight(4);  //굵기
    stroke(255);      //흰색
    line(a.position.x, a.position.y, b.position.x, b.position.y);
  }
}

function draw() {
  translate(video.width, 0);
  scale(-1,1);
  image(video, 0, 0, video.width, video.height);
  //거울과 같이 보이게 비디오를 좌우반전 및 위치조정
  
  strokeWeight(4);  //굵기
  stroke(255);      //흰색
  if(pose) {
    posedraw();
    angle = angleUpdate();
  }
  
  scale(-1,1);

  if(countName != 'None'){
    textSize(20);
    fill(0,0,0);
    strokeWeight(3);
    textAlign(RIGHT, TOP);
    text('동작 - ' + countName + '\n' + count + ' 회', -30, 30);
  }

  if(selectAngle.checked()) {
    textSize(17);
    fill(0,0,0);
    strokeWeight(2);
    textAlign(RIGHT, TOP);
    if(countName == 'None'){
      text('무릎 각도\n' + '좌: ' + angle.leftKnee + '\n우: '+ angle.rightKnee + '\n'
          + '어깨 각도\n' + '좌: ' + angle.leftShoulder + '\n우: ' + angle.rightShoulder + '\n'
          + '팔꿈치 각도\n' + '좌: ' + angle.leftElbow + '\n우: ' + angle.rightElbow, -30, 30);
    }
    else{
      text('무릎 각도\n' + '좌: ' + angle.leftKnee + '\n우: '+ angle.rightKnee + '\n'
          + '어깨 각도\n' + '좌: ' + angle.leftShoulder + '\n우: ' + angle.rightShoulder + '\n'
          + '팔꿈치 각도\n' + '좌: ' + angle.leftElbow + '\n우: ' + angle.rightElbow, -30, 90);
    }
  }
}

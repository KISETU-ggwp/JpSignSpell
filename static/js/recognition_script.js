// =========================================================================
// DOM要素の参照 (HTML要素との連携)
// =========================================================================
const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const recognitionResultDiv = document.getElementById('recognition_result');
const statusMessageDiv = document.getElementById('status_message');
// const debugInfoDiv = document.getElementById('debug_info'); // コメントアウト: HTML側でこのIDの要素がコメントアウトされているため
// const landmarksJsonPre = document.getElementById('landmarks_json'); // コメントアウト: 同上
// const predictionTextP = document.getElementById('prediction_text'); // コメントアウト: 同上

const startCameraButton = document.getElementById('start_camera');
const stopCameraButton = document.getElementById('stop_camera');
const backToHomeButton = document.getElementById('back_to_home');

// =========================================================================
// グローバル変数・定数
// =========================================================================
let camera = null; // Cameraオブジェクトのインスタンスを保持
let isRecognizing = false; // 認識処理がアクティブかどうか
let lastSendTime = 0; // バックエンドへの送信頻度を制御するためのタイムスタンプ
const SEND_INTERVAL_MS = 300; // ランドマークデータをバックエンドに送信する間隔 (ミリ秒)

// MediaPipe Handsのパス (CDN)
const MEDIAPIPE_HANDS_CDN_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands';

// カメラの解像度
const CAMERA_WIDTH = 640;
const CAMERA_HEIGHT = 480;

// MediaPipeの描画設定 (色や線の太さ、点のサイズ)
const LANDMARK_COLOR = '#FF0000'; // ランドマークの色 (赤)
const LANDMARK_LINE_WIDTH = 1;   // ランドマークの線の太さ (例: 1ピクセル)
const CONNECTOR_COLOR = '#00FF00'; // コネクタ（骨格線）の色 (緑)
const CONNECTOR_LINE_WIDTH = 2;  // コネクタの線の太さ (例: 2ピクセル)
const LANDMARK_RADIUS = 3;       // ランドマークの点の半径 (例: 3ピクセル)

// =========================================================================
// MediaPipe Hands の初期化と設定
// =========================================================================
const hands = new Hands({
  locateFile: (file) => {
    // MediaPipeの各種ファイルをCDNからロードする設定
    return `${MEDIAPIPE_HANDS_CDN_PATH}/${file}`;
  }
});

hands.setOptions({
  maxNumHands: 1, // 検出する手の最大数 (1つに制限)
  modelComplexity: 1, // モデルの複雑さ (0:高速・低精度, 1:標準, 2:高精度・低速)
  minDetectionConfidence: 0.7, // 手の検出の最小信頼度 (0.0-1.0)
  minTrackingConfidence: 0.7   // 手の追跡の最小信頼度 (0.0-1.0)
});

// MediaPipe の結果を受け取った際のコールバック関数を設定
hands.onResults(onResults);

// =========================================================================
// カメラの初期化とフレーム処理
// =========================================================================
function initializeCamera() {
  // Cameraオブジェクトのインスタンスを作成
  // videoElementからフレームを取得し、handsモデルに送信
  camera = new Camera(videoElement, {
    onFrame: async () => {
      // videoが十分にロードされているか確認 (readyStateが2以上でデータが利用可能)
      if (videoElement.readyState >= 2) { 
        await hands.send({ image: videoElement });
      }
    },
    width: CAMERA_WIDTH,
    height: CAMERA_HEIGHT
  });
}

// MediaPipe からの結果処理 (onResultsコールバック)
function onResults(results) {
    // キャンバスの描画コンテキストを保存
    canvasCtx.save();
    // キャンバスをクリア
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  
    // 映像を描画（左右反転はCSSのtransform: scaleX(-1)で行う）
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
  
    // 手が検出された場合
    if (results.multiHandLandmarks && results.multiHandedness.length > 0) {
      for (let index = 0; index < results.multiHandLandmarks.length; index++) {
        const classification = results.multiHandedness[index]; // 右手/左手の情報
        const landmarks = results.multiHandLandmarks[index];   // ランドマーク座標
  
        // 手の骨格（コネクタ）を描画
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: CONNECTOR_COLOR, lineWidth: CONNECTOR_LINE_WIDTH });
        
        // ランドマーク（関節点）を描画
        drawLandmarks(canvasCtx, landmarks, { 
          color: LANDMARK_COLOR, 
          lineWidth: LANDMARK_LINE_WIDTH, 
          radius: LANDMARK_RADIUS // <-- ここで点のサイズを調整
        });
  
        // 認識モードがアクティブな場合のみ、ランドマークを送信して認識処理
        if (isRecognizing) {
          // ランドマークを正規化
          const normalizedLandmarks = normalizeLandmarks(landmarks);
          
          // --- デバッグ表示関連の行をコメントアウト ---
          // landmarksJsonPre.textContent = JSON.stringify(normalizedLandmarks, null, 2); 
          
          // 一定間隔でバックエンドに送信
          if (Date.now() - lastSendTime > SEND_INTERVAL_MS) {
              sendLandmarksForRecognition(normalizedLandmarks);
              lastSendTime = Date.now();
          }
        }
      }
    } else {
      // 手が検出されない場合
      recognitionResultDiv.textContent = '手を検出できません';
    }
    // キャンバスの描画コンテキストを復元
    canvasCtx.restore();
}

// ランドマークを正規化する関数
// 手首 (landmarks[0]) を原点とし、各ランドマークを最大距離でスケーリング
function normalizeLandmarks(landmarks) {
  if (!landmarks || landmarks.length === 0) {
    console.warn("normalizeLandmarks: ランドマークが空です。");
    return [];
  }

  const wrist = landmarks[0]; // 手首を基準点とする
  const centeredLandmarks = landmarks.map(lm => ({
    x: lm.x - wrist.x,
    y: lm.y - wrist.y,
    z: lm.z - wrist.z // z座標も考慮
  }));

  // 各ランドマークと原点（手首）からの最大距離を計算
  let maxDist = 0;
  for (const lm of centeredLandmarks) {
    const dist = Math.sqrt(lm.x * lm.x + lm.y * lm.y + lm.z * lm.z);
    if (dist > maxDist) {
      maxDist = dist;
    }
  }

  // ゼロ除算対策: maxDistが0の場合（全てのランドマークが手首と同じ位置など）
  if (maxDist === 0) {
    console.warn("normalizeLandmarks: maxDistが0です。ランドマークを全て0に設定します。");
    return centeredLandmarks.map(lm => ({ x: 0, y: 0, z: 0 }));
  }

  // 最大距離で割って正規化
  return centeredLandmarks.map(lm => ({
    x: lm.x / maxDist,
    y: lm.y / maxDist,
    z: lm.z / maxDist
  }));
}

// =========================================================================
// バックエンド (Flask) との連携
// =========================================================================

// バックエンドに正規化されたランドマークデータを送信し、認識結果を受け取る
async function sendLandmarksForRecognition(normalizedLandmarks) {
  try {
    const response = await fetch('/predict_sign', { // FlaskのAPIエンドポイント
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ landmarks: normalizedLandmarks })
    });

    if (response.ok) {
      const data = await response.json();
      if (data.prediction) {
        recognitionResultDiv.textContent = data.prediction; // 認識結果を表示
        // predictionTextP.textContent = `最新の認識: ${data.prediction}`; // コメントアウト
        console.log('認識結果:', data.prediction);
      } else if (data.error) {
        recognitionResultDiv.textContent = 'エラー';
        // predictionTextP.textContent = `認識エラー: ${data.error}`; // コメントアウト
        console.error('認識APIエラー:', data.error);
      }
    } else {
      recognitionResultDiv.textContent = '通信エラー';
      // predictionTextP.textContent = `API通信エラー: ${response.status} ${response.statusText}`; // コメントアウト
      console.error('認識API通信エラー:', response.status, response.statusText);
    }
  } catch (error) {
    recognitionResultDiv.textContent = '通信失敗';
    // predictionTextP.textContent = `API呼び出し失敗: ${error.message}`; // コメントアウト
    console.error('認識API呼び出し中にエラー:', error);
  }
}

// =========================================================================
// イベントリスナーと初期処理
// =========================================================================

// カメラ開始ボタンのクリックイベント
startCameraButton.addEventListener('click', async () => {
    // 既にカメラが動いている場合は一度停止
    if (camera) {
        camera.stop();
        camera = null;
    }

    // カメラがまだ初期化されていなければ初期化
    if (!camera) {
      initializeCamera();
    }
    
    try {
        await camera.start(); // カメラを起動
        isRecognizing = true; // 認識モードをアクティブにする
        statusMessageDiv.textContent = '認識中...';
        recognitionResultDiv.textContent = '認識中...';
        // debugInfoDiv.style.display = 'block'; // コメントアウト
        console.log("カメラ起動: 認識処理を開始しました。");
    } catch (error) {
        console.error('カメラの起動に失敗しました:', error);
        statusMessageDiv.textContent = 'カメラの起動に失敗しました。';
        recognitionResultDiv.textContent = 'エラー';
        alert('カメラの起動に失敗しました。ブラウザのカメラ許可設定を確認してください。');
    }
});

// カメラ停止ボタンのクリックイベント
stopCameraButton.addEventListener('click', () => {
    if (camera) {
        camera.stop(); // カメラを停止
        isRecognizing = false; // 認識モードを非アクティブにする
        camera = null; // カメラオブジェクトをリセット
        statusMessageDiv.textContent = 'カメラ停止中';
        recognitionResultDiv.textContent = '停止';
        // debugInfoDiv.style.display = 'none'; // コメントアウト
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height); // キャンバスをクリア
        console.log("カメラ停止: 認識処理を終了しました。");
    }
});

// トップページに戻るボタンのクリックイベント
backToHomeButton.addEventListener('click', () => {
    if (camera) {
        camera.stop(); // ページ遷移前にカメラを停止
    }
    isRecognizing = false; // 認識モードを非アクティブにする
    // ここで直接 HOME_URL 変数を使用
    window.location.href = HOME_URL;
});

// ページロード時の初期処理 (任意)
// document.addEventListener('DOMContentLoaded', () => {
//    startCameraButton.click(); // 例: ページロード時に自動開始
// });

// =========================================================================
// 初期設定と補足
// =========================================================================
// 注意: drawConnectors と drawLandmarks 関数は、
// @mediapipe/drawing_utils/drawing_utils.js から提供されます。
// HAND_CONNECTIONS 定数も @mediapipe/hands/hands.js から提供されます。
// これらのライブラリがHTMLで正しく読み込まれていることを確認してください。

console.log("recognition_script.js が読み込まれました。");
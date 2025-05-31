import os
import numpy as np
import joblib  # 機械学習モデル (scikit-learn SVM) のロードに使用
from flask import Flask, render_template, request, jsonify

# =========================================================================
# Flaskアプリケーションの初期設定
# =========================================================================
app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(__file__), 'static'),
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

# =========================================================================
# 機械学習モデルのロード (アプリ起動時に一度だけ実行)
# =========================================================================
# モデルファイルへのパスを定義
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'svm_model_finger.pkl')
sign_recognition_model = None  # モデルを初期化

try:
    # joblibライブラリを使用してモデルをロード
    sign_recognition_model = joblib.load(MODEL_PATH)
    print("機械学習モデルをロードしました。")
except FileNotFoundError:
    # モデルファイルが見つからない場合の、サーバーコンソールへのエラー出力
    print(f"エラー: モデルファイルが見つかりません。パスを確認してください: {MODEL_PATH}")
    # この場合、sign_recognition_model は None のままとなり、
    # APIエンドポイントへのリクエスト時に適切にエラーハンドリングされます。
except Exception as e:
    # その他のモデルロード中の予期せぬエラーに対する処理
    print(f"モデルのロード中に予期せぬエラーが発生しました: {e}")
    # 同上

# =========================================================================
# ルーティング設定 (Webページの表示)
# =========================================================================

# ルートURL ('/') へのアクセス時にトップページ (index.html) を表示
@app.route('/')
def home():
    """
    トップページ (index.html) をレンダリングして表示します。
    """
    return render_template('index.html')

# '/sign' URLへのアクセス時に指文字認識ページ (sign.html) を表示
@app.route('/sign')
def sign_page():
    """
    指文字認識ページ (sign.html) をレンダリングして表示します。
    """
    return render_template('sign.html')

# =========================================================================
# 認識APIエンドポイント (JavaScriptからのデータ受け取りと推論)
# =========================================================================

# '/predict_sign' URLへのPOSTリクエストを受け付け、機械学習推論結果を返します
@app.route('/predict_sign', methods=['POST'])
def predict_sign():
    """
    フロントエンドから送られてくる正規化済みランドマークデータを受け取り、
    機械学習モデルで指文字を推論し、その結果をJSONで返します。
    """
    # モデルが正常にロードされていない場合のエラーハンドリング (ユーザーへの表示は簡潔に)
    if sign_recognition_model is None:
        print("エラー: モデルがサーバーにロードされていません。推論リクエストをスキップします。")
        return jsonify({"prediction": "エラー", "error": "モデルがロードされていません。"}), 500

    # フロントエンド (JavaScript) から送られてくるJSONデータを取得
    data = request.json
    # JSONデータから 'landmarks' キーに対応する正規化済みランドマークのリストを取得
    landmarks_data = data.get('landmarks')

    # ランドマークデータが存在しない場合のエラーハンドリング (ユーザーへの表示は簡潔に)
    if not landmarks_data:
        print("警告: ランドマークデータがありません。推論をスキップします。")
        return jsonify({"prediction": "エラー", "error": "ランドマークデータがありません。"}), 400

    # ---------------------------------------------------------------------
    # JSから送られてきた正規化済みデータを、モデルが期待するNumPy配列形式に変換
    # 各ランドマーク (x, y, z) を一つの平坦なリストに展開 (21 * 3 = 63要素)
    # ---------------------------------------------------------------------
    flattened_landmarks = []
    for lm in landmarks_data:
        flattened_landmarks.extend([lm['x'], lm['y'], lm['z']])
    
    # NumPy配列に変換し、形状を (1, N_FEATURES) にリシェイプ
    # ここでの '-1' は、要素数 (63) を自動で計算させます
    input_features = np.array(flattened_landmarks).reshape(1, -1)

    # デバッグ用: 受け取った入力データの形状をサーバーコンソールに出力
    print(f"Received landmarks shape for prediction: {input_features.shape}")

    try:
        # ---------------------------------------------------------------------
        # 機械学習モデルによる推論の実行
        # ---------------------------------------------------------------------
        prediction_result = sign_recognition_model.predict(input_features)
        
        # ---------------------------------------------------------------------
        # モデルの出力（クラスインデックス）を、対応する指文字（ひらがな）にマッピング
        # この `class_labels` リストの順序は、モデルが学習した際のクラスインデックスの順序と
        # 完全に一致させる必要があります。
        # (以前ご提示いただいたアルファベット順のリストに基づいています)
        # ---------------------------------------------------------------------
        class_labels = [
            'a', 'chi', 'e', 'fu', 'ha', 'he', 'hi', 'ho', 'i', 'ka',
            'ke', 'ki', 'ko', 'ku', 'ma', 'me', 'mi', 'mo', 'mu', 'n',
            'na', 'ne', 'ni', 'no', 'nu', 'o', 'ra', 're', 'ri', 'ro',
            'ru', 'sa', 'se', 'shi', 'so', 'su', 'ta', 'te', 'to', 'tsu',
            'u', 'wa', 'ya', 'yo', 'yu'
        ]
        
        recognized_char = "認識結果なし"  # デフォルトの認識結果

        if prediction_result is not None and len(prediction_result) > 0:
            # モデルの予測結果が数値インデックスか文字列ラベルかを判断
            # scikit-learn の predict() は通常数値インデックスを返しますが、
            # もし学習時のラベルが文字列で、かつモデルが直接文字列を返すように設計されている場合も考慮
            if isinstance(prediction_result[0], (int, np.integer)):
                # 数値インデックスが返された場合、class_labelsを使ってマッピング
                predicted_class_index = int(prediction_result[0])
                if 0 <= predicted_class_index < len(class_labels):
                    recognized_char = class_labels[predicted_class_index]
                else:
                    # 予測されたインデックスが定義済みのクラスラベルの範囲外の場合
                    recognized_char = "不明"  # ユーザーには「不明」と表示
                    print(f"警告: 予測されたクラスインデックス {predicted_class_index} が class_labels の範囲外です ({len(class_labels)}). モデルの出力とclass_labelsの対応を再確認してください。")
            else:
                # 予測結果が直接文字列ラベルだった場合 (例: 'a', 'chi' など)
                recognized_char = str(prediction_result[0])
                print(f"予測されたラベルが直接文字列です: {recognized_char}")
        else:
            recognized_char = "推論失敗" # モデルが有効な予測結果を返さなかった場合
        
        # 推論結果とモデルからの生出力をサーバーコンソールに出力（デバッグ用）
        print(f"推論結果: {recognized_char} (モデル出力: {prediction_result})")

        # フロントエンド (JavaScript) へ、シンプルな予測結果（文字）のみをJSONで返す
        return jsonify({"prediction": recognized_char})

    except Exception as e:
        # モデル推論中に予期せぬエラーが発生した場合のハンドリング
        print(f"モデル推論中に予期せぬエラーが発生しました: {e}", exc_info=True) # スタックトレースも出力
        # ユーザー側には「エラー」とだけ返し、詳細なエラー情報はサーバーコンソールに隠蔽
        return jsonify({"prediction": "エラー", "error": "推論中にエラーが発生しました。"}), 500

# =========================================================================
# Flaskアプリケーションの起動設定
# =========================================================================
if __name__ == '__main__':
    # アプリケーションを開発モードで実行
    app.run(debug=True,       # デバッグモードをオン (コード変更時に自動リロード、デバッグ情報表示)
            host='0.0.0.0',   # 全てのネットワークインターフェースからの接続を許可
            port=5001)        # ポート5001でリスン (AirPlay Receiverとの競合を避けるため)
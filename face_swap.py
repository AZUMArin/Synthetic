import cv2
import mediapipe as mp
import numpy as np

# 入力ファイル
video_path = 'driving.mp4'
face_img_path = 'honke.png'  # 透過PNG
output_path = 'complete.mp4'

# 合成設定
face_scale = 2.3     # 顔サイズ倍率
offset_x = 75      # 左ずらし量
offset_y = 140     # 上ずらし量

# PNG画像（透過あり）を読み込み
honke_face = cv2.imread(face_img_path, cv2.IMREAD_UNCHANGED)
if honke_face is None:
    raise FileNotFoundError("顔画像が読み込めません")

has_alpha = honke_face.shape[2] == 4
if not has_alpha:
    raise ValueError("透過PNGではありません（アルファチャンネルが存在しない）")

# MediaPipe初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# 動画設定
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w_frame, h_frame))

# 顔外周インデックス
face_outline_idx = [10, 338, 297, 332, 284, 251, 389, 356,
                    454, 323, 361, 288, 397, 365, 379, 378,
                    400, 377, 152, 148, 176]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            points = []
            for idx in face_outline_idx:
                x_pt = int(face_landmarks.landmark[idx].x * w_frame)
                y_pt = int(face_landmarks.landmark[idx].y * h_frame)
                points.append([x_pt, y_pt])
            points = np.array(points, dtype=np.int32)

            x, y, w_face, h_face = cv2.boundingRect(points)

            # スケーリング適用
            scaled_w = int(w_face * face_scale)
            scaled_h = int(h_face * face_scale)

            paste_x = max(0, x - offset_x)
            paste_y = max(0, y - offset_y)
            paste_x = min(paste_x, w_frame - scaled_w)
            paste_y = min(paste_y, h_frame - scaled_h)

            # 顔画像リサイズ（RGBA）
            resized_face = cv2.resize(honke_face, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            face_rgb = resized_face[:, :, :3]
            face_alpha = resized_face[:, :, 3] / 255.0  # 0〜1 に正規化
            face_alpha = face_alpha[:, :, np.newaxis]

            # ROI取り出し
            roi = frame[paste_y:paste_y + scaled_h, paste_x:paste_x + scaled_w]

            if roi.shape[:2] != face_rgb.shape[:2]:
                continue  # サイズ合わない場合スキップ

            # アルファブレンディング合成
            blended = (face_rgb * face_alpha + roi * (1 - face_alpha)).astype(np.uint8)
            frame[paste_y:paste_y + scaled_h, paste_x:paste_x + scaled_w] = blended

    out.write(frame)

cap.release()
out.release()
face_mesh.close()

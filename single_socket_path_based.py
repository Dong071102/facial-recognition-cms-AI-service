import asyncio
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(sys.path[0], 'MTCNN'))
import torch
import websockets
import cv2
import numpy as np
import base64
import json
from collections import defaultdict
from functools import partial
from DB.database import attencace_student, get_all_classes_in_next_2_hours, get_pytorch_embedding, init_attendace
from MTCNN import create_mtcnn_net
from face_model import MobileFaceNet
from facebank import load_facebank
from torchvision import transforms as trans
from PIL import Image, ImageDraw, ImageFont
from websockets.server import serve

# ================================================================
# LOAD FACE MODEL 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detect_model = MobileFaceNet(512).to(device)
detect_model.load_state_dict(torch.load('Weights/MobileFace_Net', map_location=device))
detect_model.eval()

# IMAGE PREPROCESSING
test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ✅ Fetch latest schedule and camera mappings
nearest_schedules = get_all_classes_in_next_2_hours()
classrooms_info = defaultdict(dict)
print(nearest_schedules)
if nearest_schedules:
    classrooms_info = {
        row[4]: {  
            "schedule_id":row[0],
            "class_id": row[1],
            "classroom_id": row[2],
            "camera_url": row[3],
            

        } for index, row in enumerate(nearest_schedules)
    }
print(f"📷 Camera data loaded: {classrooms_info}")
detected_student_ids = set()
#=================================================================
#===================SAVE THE IMAGE FOLDER=========================
def create_evidence_image_url(schedule_id,student_id):
    BASE_VIDEO_DIR = "evidence_image"
    today = datetime.now()  # Lấy ngày hiện tại
    date_folder = today.strftime("%Y/%m/%d")  # Tạo thư mục theo format YYYY/MM/DD

    # Tạo đường dẫn thư mục đầy đủ
    full_folder_path = os.path.join(BASE_VIDEO_DIR, date_folder)
    os.makedirs(full_folder_path, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

    # Tạo đường dẫn file video (relative path)
    video_filename = f"{schedule_id}_{student_id}.jpg"
    relative_video_path = os.path.join(full_folder_path, video_filename)

    return relative_video_path  # Trả về đường dẫn tương đối để lưu vào DB




# ================================================================
# PROCESS FRAME FUNCTION
def process_frame(frame, targets, names,schedule_id):
    """Detect faces, recognize, and draw bounding boxes & landmarks."""
    # print('targets',targets)
    print('names',names)
    try:
        bboxes, landmarks = create_mtcnn_net(frame, mini_face=20, device=device,
                                             p_model_path='MTCNN/weights/pnet_Weights',
                                             r_model_path='MTCNN/weights/rnet_Weights',
                                             o_model_path='MTCNN/weights/onet_Weights')
        # print('fond face',bboxes)
        if len(bboxes) > 0:
            # print('fond face2',bboxes)

            faces = [frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in bboxes]
            embs = [detect_model(test_transform(cv2.resize(face, (112, 112))).to(device).unsqueeze(0)) for face in faces]

            source_embs = torch.cat(embs)
            diff = source_embs.unsqueeze(-1) - targets.transpose(1, 0).unsqueeze(0)
            # print('diff',diff)
            dist = torch.sum(torch.pow(diff, 2), dim=1)
            min_idx = torch.argmin(dist, dim=1)
            index = min_idx.item()
            # print('min_idx',min_idx)

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("utils/times.ttf", 30)
            
            box = bboxes[0]
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="blue", width=3)
            draw.text((box[0], box[1] - 25), names[index][1], fill=(255, 255, 0), font=font)
            
            student_id = names[index][0]

            print('this iss student id :',student_id)
            face_image = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            _, buffer = cv2.imencode('.jpg', face_image)
            face_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result = {"id": student_id, "image": face_image_base64}
            processed_frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            print('detected_student_ids:',detected_student_ids)
            if student_id not in detected_student_ids:
                print('this iss student id :',student_id)

                image_path=create_evidence_image_url(schedule_id=schedule_id,student_id=student_id)
                print('this iss image path id :',image_path)
                if processed_frame is not None and image_path:
                    try:
                        cv2.imwrite(image_path, processed_frame)
                        print(f"📸 Ảnh lưu thành công tại: {image_path}")
                    except Exception as e:
                        print(f"❌ Lỗi khi lưu ảnh: {e}")

                attencace_student(schedule_id=schedule_id,student_id=student_id,image_path=image_path)
                detected_student_ids.add(student_id)

            return processed_frame, json.dumps(result)
    except Exception as e:
        print(f"❌ Error detecting face: {e}")

    return frame, "[]"

# ================================================================
# WEBSOCKET VIDEO STREAM FUNCTION
async def video_stream(websocket,path):

    path = websocket.path  # VD: /ws/101
    socekt_id = path.split("/")[-1]  # lấy ID lớp từ path

    classroom_info = classrooms_info.get(socekt_id)
    if not classroom_info:
        print(f"❌ Không tìm thấy lớp học {socekt_id}")
        await websocket.close()
        return

    print(f"✅ WebSocket đến lớp: {socekt_id} từ {websocket.remote_address}")

    schedule_id = classroom_info["schedule_id"]
    class_id = classroom_info["class_id"]
    camera_url = classroom_info["camera_url"]

    targets, names = get_pytorch_embedding(class_id)
    init_attendace(schedule_id)
    print(f"✅ New WebSocket connection from {websocket.remote_address} for camera: {camera_url}")
    # cap = cv2.VideoCapture('http://192.168.1.150:81/stream')
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(camera_url)
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video stream {camera_url}")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print(f"⚠️ Warning: Empty frame captured from {camera_url}; skipping")
                continue
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            processed_frame, json_result = process_frame(frame=rotated_frame, targets=targets, names=names,schedule_id=schedule_id)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            success, buffer = cv2.imencode(".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
            if not success:
                print(f"❌ Error: Failed to encode image from {camera_url}")
                continue

            encoded_frame = base64.b64encode(buffer).decode("utf-8")
            try:
                await websocket.send(encoded_frame)
            except websockets.exceptions.ConnectionClosed as e:
                print(f"⚠️ Connection closed: {e}")
                break

            await asyncio.sleep(0.03)  # approx 30 FPS

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"⚠️ Client disconnected with protocol error: {e}")
        await asyncio.sleep(2)  # Chờ 2 giây trước khi kết nối lại
        asyncio.create_task(video_stream(websocket, camera_url, targets, names, schedule_id))  # Thử lại
    finally:
        cap.release()
        print(f"🔻 Closing video stream from {camera_url}")

# ================================================================
# START MULTIPLE WEBSOCKET SERVERS FOR EACH CAMERA
    targets,names=get_pytorch_embedding(class_id)
    init_attendace(schedule_id)
    print("init_attendace successful")
    # Load facebank for the specific camera
    async with websockets.serve(partial(video_stream, camera_url=camera_url, targets=targets, names=names,schedule_id=schedule_id), "0.0.0.0", port):
        await asyncio.Future()  # Keep server running


async def main():
    print("🚀 Khởi động WebSocket server duy nhất trên ws://0.0.0.0:8765")
    async with serve(video_stream, "0.0.0.0", 11000):
        await asyncio.Future()

if __name__ == "__main__":
    try:

        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 Server stopped manually.")

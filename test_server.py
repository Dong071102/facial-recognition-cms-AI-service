import asyncio
import websockets
import base64
import cv2

PORT = 11000
IMG_PATH = "facebank/dong/2025-02-08-18-59-36.jpg"  # Đường dẫn ảnh test
DELAY = 0.3  # delay giữa các lần gửi ảnh

async def send_image(websocket):
    print(f"✅ ESP32-S3 connected: {websocket.remote_address}")

    while True:
        try:
            # Đọc ảnh
            img = cv2.imread(IMG_PATH)
            if img is None:
                print("❌ Không đọc được ảnh.")
                break

            # Resize về 240x320
            img = cv2.resize(img, (240, 320))

            # Mã hóa JPEG
            success, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if not success:
                print("❌ Không thể mã hóa JPEG.")
                break

            # Base64 encode
            encoded = base64.b64encode(buffer).decode('utf-8')

            # Gửi qua WebSocket
            await websocket.send(encoded)
            await asyncio.sleep(DELAY)

        except websockets.ConnectionClosed:
            print("🔌 ESP32-S3 đã ngắt kết nối.")
            break
        except Exception as e:
            print("❌ Lỗi khi gửi ảnh:", e)
            break

async def main():
    print(f"🚀 Đang mở WebSocket server tại ws://0.0.0.0:{PORT}")
    async with websockets.serve(send_image, "0.0.0.0", PORT):
        await asyncio.Future()  # giữ server chạy

if __name__ == "__main__":
    asyncio.run(main())

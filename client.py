import asyncio
import websockets
import cv2
import base64
import numpy as np

async def receive_video():
    uri = "ws://localhost:8000/ws/0"

    async with websockets.connect(uri) as websocket:
        while True:
            frame_base64 = await websocket.recv()
            frame_data = base64.b64decode(frame_base64)
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            cv2.imshow("Processed Video", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

asyncio.run(receive_video())

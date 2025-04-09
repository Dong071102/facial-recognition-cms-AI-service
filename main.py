from src import server

import os
import sys

# Thêm đường dẫn src/ vào sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from server import main 

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped manually.")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stream_manager import manager
import time

print("DEBUG: importing complete")

url = "https://www.youtube.com/playlist?list=PLRBp0Fe2GpgnIh0AiYKh7o7HnYA64-SZL"
print(f"DEBUG: Starting session with {url}")

sid = manager.start_session(url)
print(f"DEBUG: Session started: {sid}")

session = manager.get_session(sid)

print("DEBUG: Waiting for chunks...")
while True:
    if not session.chunks_queue.empty():
        chunk = session.chunks_queue.get()
        print(f"GOT CHUNK: {chunk}")
        if chunk is None:
            break
    time.sleep(1)
    print(f"DEBUG: Status: {session.status}")

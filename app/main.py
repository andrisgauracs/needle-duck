"""Needle Duck Studio: a native window around the local server.

    .venv/bin/python app/main.py            # the app window
    .venv/bin/python app/main.py --browser  # serve only, open http://127.0.0.1:8777 yourself
"""
import os, socket, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("NEEDLE_TELEMETRY", "0")
PORT = int(os.environ.get("DUCK_STUDIO_PORT", 8777))


def serve():
    import uvicorn
    from server import app
    uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")).run()


def wait_up(timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        try:
            socket.create_connection(("127.0.0.1", PORT), 0.3).close(); return True
        except OSError:
            time.sleep(0.2)
    return False


if __name__ == "__main__":
    if "--browser" in sys.argv:
        serve(); sys.exit()
    threading.Thread(target=serve, daemon=True).start()
    wait_up()
    import webview
    try:  # the duck icon in the Dock instead of Python's
        from AppKit import NSApplication, NSImage
        icon = Path(__file__).resolve().parent.parent / "Needle Duck Studio.app/Contents/Resources/icon.icns"
        NSApplication.sharedApplication().setApplicationIconImage_(NSImage.alloc().initWithContentsOfFile_(str(icon)))
    except Exception:
        pass
    webview.create_window("Needle Duck Studio", f"http://127.0.0.1:{PORT}/?launch={int(time.time())}", width=1600, height=960,
                          min_size=(1200, 760), background_color="#0b0d12")
    webview.start()
    try:  # stop the Needle worker processes too, so none outlive the app
        from server import brains
        for b in list(brains.values()):
            b.close()
    except Exception:
        pass
    os._exit(0)  # tear down the sim thread and the Needle workers with the window

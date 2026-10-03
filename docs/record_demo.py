"""Re-record docs/demo.gif: drive Needle Duck Studio in headless Chrome and capture every repaint.

    pip install playwright pillow                 # in any venv; uses your installed Chrome
    .venv/bin/python app/main.py --browser        # the app server, in another terminal (needs models/duck_20l.cact)
    python docs/record_demo.py /tmp/frames
    gifski --fps 15 --width 960 --quality 85 -o docs/demo.gif /tmp/frames/f_*.png
"""
import asyncio, base64, json, os, sys, time
from pathlib import Path
from playwright.async_api import async_playwright

URL = os.environ.get("STUDIO_URL", "http://127.0.0.1:8777") + "/?rec=1"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
W, H = 1280, 760

IDLE_JS = """async () => { for (let i = 0; i < 200; i++) { await new Promise(r => setTimeout(r, 150));
  const s = await fetch('/api/state').then(r => r.json()); if (s.sim.status === 'idle' && s.sim.queued === 0) return true; } return false; }"""


async def main():
    frames = []
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome", headless=True)
        page = await b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        await page.goto(URL)
        await page.wait_for_function("() => typeof setStep === 'function' && typeof S !== 'undefined' && S !== null")
        # clean starting state: Before step, empty conversation, duck at home, close front 3/4 camera
        await page.evaluate("""async () => {
            await fetch('/api/clear_history', {method: 'POST'}); await fetch('/api/reset', {method: 'POST'});
            Object.assign(cam, {azimuth: 158, elevation: -9, distance: 0.85}); pushCam(); setStep('before'); }""")
        await page.wait_for_timeout(3500)

        cdp = await page.context.new_cdp_session(page)

        def on_frame(ev):
            frames.append((time.time(), base64.b64decode(ev["data"])))
            asyncio.ensure_future(cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]}))
        cdp.on("Page.screencastFrame", on_frame)
        await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": W, "maxHeight": H, "everyNthFrame": 1})

        async def say(text, hold=0.6):
            await page.evaluate("t => send(t)", text)
            await page.wait_for_timeout(300)
            await page.evaluate(IDLE_JS)
            await page.wait_for_timeout(int(hold * 1000))

        await page.wait_for_timeout(1200)
        await say("walk forward", 0.9)                       # base Needle obeys without a please
        await page.evaluate("""() => { setStep('after'); const s = document.querySelector('#model-select');
                               s.value = 'duck_20l.cact'; s.onchange({target: s}); }""")
        await page.wait_for_timeout(1300)
        await say("walk forward", 0.6)                       # fine-tuned: shakes its head
        await say("walk forward please", 0.6)                # walks
        await say("the floor is lava!", 1.6)                 # chicken dance
        await cdp.send("Page.stopScreencast")
        await b.close()

    from PIL import Image
    import io
    t0, fps = frames[0][0], 15
    times = [t - t0 for t, _ in frames]
    j = k = 0
    while k / fps <= times[-1]:          # resample the uneven repaints to a steady frame rate
        while j + 1 < len(frames) and times[j + 1] <= k / fps:
            j += 1
        Image.open(io.BytesIO(frames[j][1])).save(OUT / f"f_{k:05d}.png")
        k += 1
    print(f"{k} frames at {fps} fps ({times[-1]:.1f}s) in {OUT}")


asyncio.run(main())

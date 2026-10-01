import os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
import mujoco, numpy as np

class Recorder:
    """Offscreen tracking camera. Attach with duck.frame_cb = rec; frames at ~25 fps."""
    def __init__(self, duck, w=640, h=480, azimuth=160, elevation=-15, distance=0.9, fps=25):
        duck.model.vis.global_.offwidth = max(duck.model.vis.global_.offwidth, w)
        duck.model.vis.global_.offheight = max(duck.model.vis.global_.offheight, h)
        self.r = mujoco.Renderer(duck.model, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = duck.model.body("trunk_assembly").id if _has_body(duck.model, "trunk_assembly") else 1
        self.cam.azimuth, self.cam.elevation, self.cam.distance = azimuth, elevation, distance
        self.frames, self.every, self.n = [], max(1, int(round(50 / fps))), 0
        self.caption = ""
    def __call__(self, duck):
        self.n += 1
        if self.n % self.every: return
        self.r.update_scene(duck.data, self.cam)
        self.frames.append((self.r.render().copy(), self.caption))
    def save_mp4(self, path, fps=25):
        import imageio.v2 as iio
        from PIL import Image, ImageDraw, ImageFont
        try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        except Exception: font = ImageFont.load_default()
        w = iio.get_writer(path, fps=fps, codec="libx264", quality=8)
        for img, cap in self.frames:
            if cap:
                im = Image.fromarray(img); d = ImageDraw.Draw(im)
                lines = cap.split("\n")
                bar = 12 + 30 * len(lines)
                d.rectangle([0, im.height - bar, im.width, im.height], fill=(0, 0, 0))
                for i, line in enumerate(lines):
                    d.text((14, im.height - bar + 8 + 30 * i), line, font=font,
                           fill=(255, 255, 255) if i == 0 else (255, 210, 80))
                img = np.asarray(im)
            w.append_data(img)
        w.close()

def _has_body(m, name):
    try: m.body(name); return True
    except KeyError: return False

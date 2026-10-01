"""The duck, rendered offscreen in real time and streamed to the app as JPEG frames.

One thread owns MuJoCo (the sim and the renderer). Everything else talks to it through
a queue of tool-call batches, so the window never blocks on the physics.
"""
import io, math, os, queue, sys, threading, time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

W, H, FPS = 1280, 720, 30

# Same scene as the Playground's flat terrain (robot + "home" keyframe untouched), restyled for camera:
# dark gradient sky, dark checker floor with a soft reflection, a key light that casts shadows.
STUDIO_VISUAL = """
    <visual>
        <headlight diffuse="0.45 0.45 0.45" ambient="0.35 0.35 0.38" specular="0.1 0.1 0.1"/>
        <rgba haze="0.09 0.1 0.14 1"/>
        <quality shadowsize="4096"/>
        <map znear="0.01" fogstart="3" fogend="9"/>
    </visual>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.17 0.19 0.25" rgb2="0.1 0.11 0.15" width="800" height="800"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.13 0.15 0.19" rgb2="0.11 0.125 0.16"
            markrgb="0.26 0.29 0.36" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="4 4" reflectance="0.12"/>
    </asset>
    <worldbody>
        <light name="key" pos="0.6 -0.8 2.2" dir="-0.25 0.35 -1" diffuse="0.75 0.72 0.68" specular="0.2 0.2 0.2" castshadow="true"/>
        <light name="rim" pos="-1 1 1.5" dir="0.5 -0.5 -0.6" diffuse="0.25 0.3 0.4" castshadow="false"/>
    </worldbody>
"""


def studio_scene():
    import re
    from duck_sim import _XMLS
    src = (_XMLS / "scene_flat_terrain.xml").read_text()
    src = re.sub(r"<!--.*?-->", "", src, flags=re.S)
    src = re.sub(r"<visual>.*?</visual>", "", src, count=1, flags=re.S)
    src = re.sub(r"<asset>.*?</asset>", STUDIO_VISUAL, src, count=1, flags=re.S)
    out = Path(__file__).resolve().parent / "scene_studio.xml"
    out.write_text(src)
    return out


class SimEngine:
    def __init__(self, log):
        self.log = log                      # log(kind, text)
        self.jobs = queue.Queue()
        self.frame = None                   # latest JPEG bytes
        self.frame_id = 0
        self.cond = threading.Condition()
        # azimuth is relative to the duck's heading (follow cam), so it stays framed after it turns
        self.cam_args = {"azimuth": 150.0, "elevation": -12.0, "distance": 1.0}
        self._cam_yaw = None
        self.status = "starting"
        self.current = None                 # tool call being performed
        self._reset = False
        self.ready = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="sim").start()

    # ---- called from other threads -------------------------------------------------
    def perform(self, calls, on_done=None):
        self.jobs.put((calls, on_done))

    def reset(self):
        self._reset = True

    def set_camera(self, **kw):
        for k, v in kw.items():
            if k in self.cam_args and v is not None:
                self.cam_args[k] = float(v)
        self.cam_args["elevation"] = max(-80.0, min(5.0, self.cam_args["elevation"]))
        self.cam_args["distance"] = max(0.4, min(3.0, self.cam_args["distance"]))

    def wait_frame(self, last_id, timeout=1.0):
        with self.cond:
            if self.frame_id == last_id:
                self.cond.wait(timeout)
            return self.frame_id, self.frame

    # ---- sim thread ----------------------------------------------------------------
    def _run(self):
        from duck_sim import Duck
        from duck_moves import perform, empty_reaction
        self._perform, self._empty = perform, empty_reaction
        self.duck = duck = Duck(scene=studio_scene())
        duck.model.vis.global_.offwidth = max(duck.model.vis.global_.offwidth, W)
        duck.model.vis.global_.offheight = max(duck.model.vis.global_.offheight, H)
        self.renderer = mujoco.Renderer(duck.model, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = duck.model.body("trunk_assembly").id
        self._wall0, self._sim0, self._last_render = time.time(), duck.t, 0.0
        duck.frame_cb = self._on_step
        self.status = "idle"
        self.ready.set()
        self.log("sim", "MuJoCo ready: Open Duck Mini v2 + pretrained walking policy")
        while True:
            if self._reset:
                self._do_reset()
            try:
                calls, on_done = self.jobs.get_nowait()
            except queue.Empty:
                duck.settle(0.05)
                continue
            try:
                results = self._act(calls)
            except _Interrupted:
                results = [{"interrupted": True}]
            self.current, self.status = None, "idle"
            if on_done:
                try: on_done(results)
                except Exception as e: self.log("error", f"on_done: {e}")

    def _act(self, calls):
        duck = self.duck
        self.status = "acting"
        if not calls:
            self.current = {"name": "(no call)", "arguments": {}}
            self._empty(duck)
            return []
        out = []
        for c in calls:
            self.current = c
            try:
                r = self._perform(duck, c)
            except Exception as e:
                r = {"error": str(e)}
                self.log("error", f"{c.get('name')}: {e}")
            if r.get("fell_over"):
                self.log("sim", "the duck fell over and got back up")
            out.append(r)
            if self._reset:
                break
        return out

    def _do_reset(self):
        self._reset = False
        d = self.duck
        mujoco.mj_resetDataKeyframe(d.model, d.data, d.model.keyframe("home").id)
        d.last_action[:] = 0; d.ll_action[:] = 0; d.lll_action[:] = 0
        d.motor_targets = d.default_actuator.copy(); d.prev_motor_targets = d.default_actuator.copy()
        d.commands[:] = 0; d.head_fn = None
        mujoco.mj_forward(d.model, d.data)
        self._wall0, self._sim0, self._cam_yaw = time.time(), d.t, None
        self.log("sim", "duck reset to the start position")

    def _on_step(self, duck):
        # pace the sim to wall-clock time and render at FPS
        ahead = (duck.t - self._sim0) - (time.time() - self._wall0)
        if ahead > 0:
            time.sleep(ahead)
        elif ahead < -0.25:   # fell behind (e.g. heavy render); resync instead of fast-forwarding
            self._wall0, self._sim0 = time.time(), duck.t
        if duck.t - self._last_render < 1.0 / FPS:
            return
        self._last_render = duck.t
        if self._reset and self.status == "acting":
            raise _Interrupted()
        c = self.cam
        yaw = duck.pose()["yaw_deg"]
        if self._cam_yaw is None:
            self._cam_yaw = yaw
        self._cam_yaw += ((yaw - self._cam_yaw + 180) % 360 - 180) * 0.06   # ease toward the heading
        c.azimuth = self.cam_args["azimuth"] + self._cam_yaw
        c.elevation, c.distance = self.cam_args["elevation"], self.cam_args["distance"]
        self.renderer.update_scene(duck.data, c)
        buf = io.BytesIO()
        Image.fromarray(self.renderer.render()).save(buf, "JPEG", quality=85)
        with self.cond:
            self.frame, self.frame_id = buf.getvalue(), self.frame_id + 1
            self.cond.notify_all()


class _Interrupted(BaseException):
    """Raised from the frame hook to abort a move on reset; BaseException skips the per-call except."""

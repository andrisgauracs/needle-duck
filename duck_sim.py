"""Headless-or-viewer Open Duck Mini v2 simulator with a high-level command API.

Legs are driven by the pretrained Open Duck walking policy (BEST_WALK_ONNX_2.onnx).
Head joints are overridden by scripted "emote" animations layered on top, so the
policy keeps the duck balanced while the head does silly things.
"""
import os, sys, types, math, time
from pathlib import Path
import numpy as np
import mujoco

_HERE = Path(__file__).parent
PLAYGROUND = Path(os.environ.get("OPEN_DUCK_PLAYGROUND", _HERE / "vendor" / "Open_Duck_Playground"))
sys.path.insert(0, str(PLAYGROUND))

# Stub out the jax-heavy base module; the inference base only needs get_assets().
_XMLS = PLAYGROUND / "playground/open_duck_mini_v2/xmls"
def _get_assets():
    assets = {}
    for p in _XMLS.glob("*.xml"):
        assets[p.name] = p.read_bytes()
    for p in (_XMLS / "assets").glob("*"):
        assets[p.name] = p.read_bytes()
    return assets
_stub = types.ModuleType("playground.open_duck_mini_v2.base")
_stub.get_assets = _get_assets
sys.modules["playground.open_duck_mini_v2.base"] = _stub

from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase  # noqa: E402
from playground.common.onnx_infer import OnnxInfer  # noqa: E402
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion  # noqa: E402

HEAD = slice(5, 9)  # neck_pitch, head_pitch, head_yaw, head_roll in actuator order
HERE = Path(__file__).parent


class Duck(MJInferBase):
    VX, VY, WZ = 0.15, 0.2, 1.0  # max forward, lateral (m/s), turn (rad/s)

    def __init__(self, scene="scene_flat_terrain.xml", onnx=HERE / "BEST_WALK_ONNX_2.onnx"):
        super().__init__(str(_XMLS / scene))
        self.policy = OnnxInfer(str(onnx), awd=True)
        self.PRM = PolyReferenceMotion(str(PLAYGROUND / "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"))
        n = self.num_dofs
        self.last_action = np.zeros(n); self.ll_action = np.zeros(n); self.lll_action = np.zeros(n)
        self.commands = np.zeros(7)
        self.imitation_i = 0.0
        self.phase = np.zeros(2)
        self.max_motor_velocity = 5.24
        self.head_fn = None      # t -> [neck_pitch, head_pitch, head_yaw, head_roll] offsets
        self.head_t = 0.0
        self.t = 0.0
        self.viewer = None
        self.frame_cb = None     # optional hook for recording
        self._trunk = self.model.body("trunk_assembly").id

    # --- low level -------------------------------------------------------
    def _obs(self):
        acc = self.get_accelerometer(self.data).copy(); acc[0] += 1.3
        return np.concatenate([
            self.get_gyro(self.data), acc, self.commands,
            self.get_actuator_joints_qpos(self.data.qpos) - self.default_actuator,
            self.get_actuator_joints_qvel(self.data.qvel) * 0.05,
            self.last_action, self.ll_action, self.lll_action,
            self.motor_targets, self.get_feet_contacts(self.data), self.phase,
        ])

    def _control_step(self):
        self.imitation_i = (self.imitation_i + 1.0) % self.PRM.nb_steps_in_period
        a = self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
        self.phase = np.array([np.cos(a), np.sin(a)])
        action = self.policy.infer(self._obs())
        self.lll_action, self.ll_action, self.last_action = self.ll_action.copy(), self.last_action.copy(), action.copy()
        targets = self.default_actuator + action * 0.25
        dt = self.sim_dt * self.decimation
        if self.head_fn is not None:
            targets[HEAD] = self.default_actuator[HEAD] + np.asarray(self.head_fn(self.head_t))
            self.head_t += dt
        targets = np.clip(targets, self.prev_motor_targets - self.max_motor_velocity * dt,
                          self.prev_motor_targets + self.max_motor_velocity * dt)
        self.prev_motor_targets = targets.copy()
        self.motor_targets = targets
        self.data.ctrl[:] = targets

    def simulate(self, seconds, realtime=None):
        realtime = self.viewer is not None if realtime is None else realtime
        steps = int(seconds / self.sim_dt)
        for i in range(steps):
            t0 = time.time()
            mujoco.mj_step(self.model, self.data)
            self.t += self.sim_dt
            if i % self.decimation == 0:
                self._control_step()
                if self.frame_cb: self.frame_cb(self)
            if self.viewer is not None:
                if i % 8 == 0: self.viewer.sync()
                if not self.viewer.is_running(): raise SystemExit
            if realtime:
                sleep = self.sim_dt - (time.time() - t0)
                if sleep > 0: time.sleep(sleep)

    # --- state ------------------------------------------------------------
    def pose(self):
        x, y, z = self.data.qpos[0:3]
        w, qx, qy, qz = self.data.qpos[3:7]
        yaw = math.degrees(math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        return {"x": round(float(x), 3), "y": round(float(y), 3), "z": round(float(z), 3), "yaw_deg": round(yaw, 1)}

    def fallen(self):
        # upstream get_gravity() reads the wrong sensor address, so use the trunk's z axis
        up_z = self.data.xmat[self._trunk].reshape(3, 3)[2, 2]
        return self.data.qpos[2] < 0.10 or up_z < 0.6

    def respawn(self):
        """Put the duck back on its feet where it is, facing the same way."""
        x, y = self.data.qpos[0:2].copy()
        yaw = math.radians(self.pose()["yaw_deg"])
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.keyframe("home").id)
        self.data.qpos[0:2] = [x, y]
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.last_action[:] = 0; self.ll_action[:] = 0; self.lll_action[:] = 0
        self.motor_targets = self.default_actuator.copy(); self.prev_motor_targets = self.default_actuator.copy()
        self.commands[:] = 0; self.head_fn = None
        mujoco.mj_forward(self.model, self.data)

    # --- high level primitives --------------------------------------------
    def move(self, vx=0.0, vy=0.0, wz=0.0, seconds=1.0, head=None):
        self.commands[:3] = [vx, vy, wz]
        self.head_fn, self.head_t = head, 0.0
        self.simulate(seconds)
        self.commands[:3] = 0.0
        self.head_fn = None

    def settle(self, seconds=0.6):
        self.move(seconds=seconds)

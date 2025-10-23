"""
Deterministic version of the CWL/SWAL lane-guidance + adaptive signal simulation.

Goals:
- For a given `random_seed`, ALL random outcomes (e.g., lane choices in SWAL 5/6, any other randomness) are reproducible.
- Remove timing/ordering nondeterminism caused by threads and wall-clock sleeps.
- Make signal updates, vehicle generation, and optimization triggers occur on a fixed, discrete timeline.
- Interact with `stage1_opt` deterministically by setting/restoring RNG state around calls.

Key changes from original code:
1) Seed handling: Use a dedicated RNG (`rng = random.Random(seed)`) and a dedicated NumPy Generator (`np_rng = np.random.default_rng(seed)`).
   - Avoid using global `random` and `numpy.random` in the main code path. When calling `stage1_opt` (which may use global RNG),
     we set + restore global RNG states to be deterministic, isolating side-effects.
2) Eliminate background threads and `time.sleep` in core logic.
   - Signals are updated once per simulated second in the main loop.
   - Vehicle generation from JSON is performed at the exact simulated second boundary in the main loop.
   - Optimization is executed synchronously at fixed steps (every `Step` local seconds), then the main loop resumes.
3) Fixed-step simulation clock based on frame counts, NOT wall-clock (`pygame.time.get_ticks`).
   - Each loop is one frame. We count frames; every `FPS` frames => +1 simulated second.
4) Deterministic iteration orders.
   - When building `vehicles_info`, iterate vehicles sorted by `vehicle.number`.
   - When multiple vehicles are scheduled at the same second from JSON, sort them by a stable key.

NOTE: File/asset paths (images, JSON) follow the original structure.
"""

import os
import math
import json
import random
import pygame
import numpy as np

import stage1_opt

# =========================
# --- Determinism Setup ---
# =========================

number_of_changed_vehs = 0

def setup_global_hash():
    # Optional: Hash randomization off for deterministic dict/set iteration in some cases.
    os.environ.setdefault("PYTHONHASHSEED", "0")


def make_rng(seed: int):
    """Create dedicated RNGs for Python and NumPy (do NOT use global random/numpy.random elsewhere)."""
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    return py_rng, np_rng


def stage1_opt_call_with_seed(seed: int, func, *args, **kwargs):
    """Call `func(*args, **kwargs)` with global RNGs set deterministically, then restore previous states.
    This isolates any randomness inside `stage1_opt`.
    """
    # Save states
    import numpy as _np
    py_state = random.getstate()
    np_state = _np.random.get_state()

    try:
        random.seed(seed)
        _np.random.seed(seed)
        return func(*args, **kwargs)
    finally:
        # Restore states
        random.setstate(py_state)
        _np.random.set_state(np_state)


# =========================
# --- Reaction Time Model --
# =========================

def compute_reaction_time(vehicle, lane_type: str, speed_mps: float) -> float:
    # Baseline by vehicle type
    tau = 1.2 if vehicle.type == 'car' else 1.6

    # Lateral space effect: SWAL is narrower => longer reaction
    if lane_type == 'SWAL':
        tau += 0.3   # tuneable

    # Speed effect: at higher speeds, drivers tend to react slightly faster
    if speed_mps > 8.0:  # ~28.8 km/h
        tau -= 0.1

    # Clamp to reasonable range
    return max(0.8, min(tau, 2.5))

# =========================
# --- Simulation Config ---
# =========================

# Geometry/visual
L1 = 100
FPS = 60  # Fixed frames per simulated second
SCREEN_W, SCREEN_H = 842, 842  # 421*2

# Speeds & sizes (pixels)
REAL_SPEED_KMH = 40
REAL_SPEED_MS = REAL_SPEED_KMH / 3.6
V_SPEED = round(REAL_SPEED_MS * 2 / FPS, 2)  # pixels per frame (2 px/m, FPS frames/s)
CAR_LEN, BUS_LEN = 10, 24  # (5m, 12m) * 2 px/m
STOP_GAP, MOVE_GAP = 4, 4

# Time horizon
SIM_SECONDS = 300  # 5 minutes
STEP = 5          # optimize every STEP local seconds

# Paths
JSON_PATH_TEMPLATE = r"E:\0 论文\3、CV+SWAL\code\scenarios\demand_{demand}\bus_{bus}\vehicles_{seed}.json"

# =========================
# --- Static Geometry ----
# =========================

def build_geometry():
    # Divided line between CWL and SWAL
    Divided_line = {i: 442 + 2 * L1 + 0.5 if i in [1, 4] else 400 - 2 * L1 - 0.5 for i in [1, 2, 3, 4]}

    # CWL lane centers
    x = {
        (1, 1): 424.5, (1, 2): 431.5, (1, 3): 438.5,
        (2, 1): 0,     (2, 2): 0,     (2, 3): 0,
        (3, 1): 417.5, (3, 2): 410.5, (3, 3): 403.5,
        (4, 1): 842,   (4, 2): 842,   (4, 3): 842,
    }
    y = {
        (1, 1): 842,   (1, 2): 842,   (1, 3): 842,
        (2, 1): 422.5, (2, 2): 430.5, (2, 3): 437.5,
        (3, 1): 0,     (3, 2): 0,     (3, 3): 0,
        (4, 1): 417.5, (4, 2): 410.5, (4, 3): 403.5,
    }

    # CWL + SWAL centers (per arm, lane 1-7, vehicle type)
    defaultLaneCentral = {
        # cars
        (1, 1, "car"): 424.5, (1, 2, "car"): 431.5, (1, 3, "car"): 438.5,
        (1, 4, "car"): 422.6, (1, 5, "car"): 427.8, (1, 6, "car"): 432.0, (1, 7, "car"): 437.2,
        (2, 1, "car"): 424.5, (2, 2, "car"): 431.5, (2, 3, "car"): 437.5,
        (2, 4, "car"): 422.6, (2, 5, "car"): 427.8, (2, 6, "car"): 432.0, (2, 7, "car"): 437.2,
        (3, 1, "car"): 416.5, (3, 2, "car"): 410.5, (3, 3, "car"): 403.5,
        (3, 4, "car"): 417.4, (3, 5, "car"): 412.2, (3, 6, "car"): 407.0, (3, 7, "car"): 401.7,
        (4, 1, "car"): 417.5, (4, 2, "car"): 410.5, (4, 3, "car"): 404.5,
        (4, 4, "car"): 416.4, (4, 5, "car"): 411.2, (4, 6, "car"): 406.0, (4, 7, "car"): 401.7,
        # buses
        (1, 1, "bus"): 424.5, (1, 2, "bus"): 431.5, (1, 3, "bus"): 438.5,
        (1, 4, "bus"): 422.6, (1, 5, "bus"): 427.8, (1, 6, "bus"): 433.0, (1, 7, "bus"): 438.2,
        (2, 1, "bus"): 421.5, (2, 2, "bus"): 426.5, (2, 3, "bus"): 434.5,
        (2, 4, "bus"): 423.6, (2, 5, "bus"): 428.8, (2, 6, "bus"): 434.0, (2, 7, "bus"): 439.2,
        (3, 1, "bus"): 415.5, (3, 2, "bus"): 408.5, (3, 3, "bus"): 401.5,
        (3, 4, "bus"): 412.4, (3, 5, "bus"): 407.2, (3, 6, "bus"): 401.0, (3, 7, "bus"): 396.7,
        (4, 1, "bus"): 417.5, (4, 2, "bus"): 410.5, (4, 3, "bus"): 403.5,
        (4, 4, "bus"): 413.4, (4, 5, "bus"): 407.2, (4, 6, "bus"): 402.0, (4, 7, "bus"): 397.7,
    }

    stopLines = {1: 221 * 2, 2: 200 * 2, 3: 200 * 2, 4: 221 * 2}
    defaultStop = {1: 222 * 2, 2: 199 * 2, 3: 199 * 2, 4: 222 * 2}

    return Divided_line, x, y, defaultLaneCentral, stopLines, defaultStop


# =========================
# --- Demand & Signals ----
# =========================

def build_default_signal():
    # (start, green_duration, amber)
    return {
        1:  (0.0, 6.0, 4),  2:  (0.0, 12.24, 4), 3:  (0.0, 12.24, 4),
        4:  (36.24, 6.0, 4), 5:  (36.24, 6.0, 4), 6:  (36.24, 6.0, 4),
        7:  (16.24, 6.0, 4), 8:  (10.0, 12.24, 4), 9:  (10.0, 12.24, 4),
        10: (26.24, 6.0, 4), 11: (26.24, 6.0, 4), 12: (26.24, 6.0, 4),
    }


def build_phases_map():
    # Map (arm, lane)-> phase index (0..12). 0 kept as CWL helper channel.
    Phases = {
        (1, 1): 0, (1, 2): 0, (1, 3): 0, (1, 4): 1,  (1, 5): 2,  (1, 6): 2,  (1, 7): 3,
        (2, 1): 0, (2, 2): 0, (2, 3): 0, (2, 4): 4,  (2, 5): 5,  (2, 6): 5,  (2, 7): 6,
        (3, 1): 0, (3, 2): 0, (3, 3): 0, (3, 4): 7,  (3, 5): 8,  (3, 6): 8,  (3, 7): 9,
        (4, 1): 0, (4, 2): 0, (4, 3): 0, (4, 4): 10, (4, 5): 11, (4, 6): 11, (4, 7): 12,
    }
    return Phases


# =========================
# --- TrafficSignal class --
# =========================

class TrafficSignal:
    def __init__(self, red_state: int, green_state: int):
        self.red_state = red_state
        self.green_state = green_state


# =========================
# --- Vehicle class --------
# =========================

class Vehicle(pygame.sprite.Sprite):
    def __init__(self, number, arm, lane, vehicle_type, movement, swal_lane,
                 lane_change_time, target_lane, generate_time,
                 x_ref, y_ref, defaultLaneCentral, defaultStop,
                 stopLines, Divided_line,
                 vehicles_container, simulation_group,
                 speeds):
        super().__init__()
        self.number = number
        self.arm = arm
        self.type = vehicle_type  # "car" or "bus"
        self.movement = movement  # 'left'/'straight'/'right'
        self.speed = speeds[vehicle_type]
        self.length = CAR_LEN if self.type == "car" else BUS_LEN
        self.image = pygame.image.load(f"images/{arm}/{vehicle_type}.png")
        self.generate_time = generate_time

        # Optimization-controlled
        self.lane = lane
        self.lane_change_time = lane_change_time
        self.swal_lane = swal_lane
        self.target_lane = target_lane

        # Initial position & free-flow distance
        self.x = (x_ref[arm, lane] - self.length) if self.arm == 2 else x_ref[arm, lane]
        self.y = (y_ref[arm, lane] - self.length) if self.arm == 3 else y_ref[arm, lane]
        if self.arm == 1:
            self.intitial_L = (self.y - SCREEN_H) / 2 + 200
        elif self.arm == 2:
            self.intitial_L = (-self.length - self.x) / 2 + 200
        elif self.arm == 3:
            self.intitial_L = (-self.length - self.y) / 2 + 200
        else:
            self.intitial_L = (self.x - SCREEN_W) / 2 + 200

        self.crossed = 0
        self.affected = []
        self.lane_change_complete = False
        self.is_moving = True
        self.restart_move_time = 4000
        self.restart_min_time = 0
        self.change_area_bool = False

        # Register into containers
        self.vehicles_container = vehicles_container
        self.simulation_group = simulation_group
        self.defaultLaneCentral = defaultLaneCentral
        self.defaultStop = defaultStop
        self.stopLines = stopLines
        self.Divided_line = Divided_line

        # Append to queue
        self.vehicles_container[arm][lane].append(self)
        self.index = len(self.vehicles_container[arm][lane]) - 1
        
        #==============新增换道有关==================#
        # --- Lane-change (bus-only) state ---
        self.lc_state = 'idle'          # 'idle' | 'executing'
        self.lc_t0 = 0.0                # 开始时刻（Local_floatTime）
        self.lc_t1 = 0.0                # 结束时刻（= t0 + duration）
        self.lc_from = lane
        self.lc_to = lane
        self.lc_x0, self.lc_y0 = self.x, self.y
        self.lc_x1, self.lc_y1 = self.x, self.y
        self.lc_overlay_in_target = False  # 目标车道 overlay 占用标记
        
        # 换道时降速（相对原速的倍率）
        self.lc_speed_factor = 0.9  # bus 更保守；如需更保守可取 0.85
        #==============新增换道有关==================#

        # Initial stop position (car-following target)
        def initial_stop(arm_, lane_, vehs):
            if len(vehs[arm_][lane_]) > 1 and vehs[arm_][lane_][-2].crossed == 0:
                last = vehs[arm_][lane_][-2]
                if arm_ in [1, 4]:
                    return last.stop + last.length + STOP_GAP
                else:
                    return last.stop - last.length - STOP_GAP
            else:
                return self.defaultStop[arm_]

        self.stop = initial_stop(self.arm, self.lane, self.vehicles_container)

        # Push next generation coordinate to avoid overlap
        delta = self.length + STOP_GAP
        if arm == 2:
            x_ref[arm, lane] -= delta
        elif arm == 4:
            x_ref[arm, lane] += delta
        elif arm == 3:
            y_ref[arm, lane] -= delta
        elif arm == 1:
            y_ref[arm, lane] += delta

        # Add to simulation
        self.simulation_group.add(self)

    # ===== Deterministic helpers =====
    def _phase_index(self, Phases):
        return Phases[(self.arm, self.lane)]

    def reset_lane_change(self, new_time):
        self.lane_change_time = new_time

    def reset_target_lane(self, new_target):
        self.target_lane = new_target
        if self.type == "bus":
            # Map CWL 1->SWAL4, 2->5, 3->6
            self.swal_lane = 4 if self.target_lane == 1 else 5 if self.target_lane == 2 else 6

    def reset_speed(self, Local_floatTime, speeds_map):
        # Speeds_map may contain temporary speed before lane change time
        if self.number in speeds_map and not self.lane_change_complete:
            if Local_floatTime <= self.lane_change_time:
                self.speed = speeds_map[self.number]

    # ========= Behavior =========

    #==============新增换道有关==================#
    def sample_bus_lane_change_duration(self, rng):
        if self.type != "bus":
            return 0
        # 正态采样（rng.random() -> Box-Muller 简版或用简单离散近似）为了简洁可用离散表近似：权重集中在 4–7
        candidates = [4,5,5,5,6,6,7]  # 粗略近似 N(5,1^2) 的离散样本
        return rng.choice(candidates)
    
    def _lane_centroid(self, arm, lane, defaultLaneCentral):
        if arm in [1, 3]:
            return defaultLaneCentral[(arm, lane, self.type)], None
        else:
            return None, defaultLaneCentral[(arm, lane, self.type)]

    def _smoothstep(self, s: float) -> float:
        # S曲线插值（0-1）
        return s*s*(3.0 - 2.0*s)
    
    def _apply_lateral_progress(self, Local_floatTime):
        if getattr(self, 'lc_state', 'idle') != 'executing':
            return
        dur = max(1e-6, self.lc_t1 - self.lc_t0)
        s = (Local_floatTime - self.lc_t0) / dur
        if s <= 0.0: s = 0.0
        elif s >= 1.0: s = 1.0
        # S 曲线
        w = s*s*(3.0 - 2.0*s)
    
        # 横向 + 纵向 都插值
        self.x = self.lc_x0 + (self.lc_x1 - self.lc_x0) * w
        self.y = self.lc_y0 + (self.lc_y1 - self.lc_y0) * w
    
        # 完成：切换 solid 列表（从原 lane 删、按纵向位置插入目标 lane）
        if s >= 1.0:
            # 从原 lane 的 solid 删除并维护 index
            if self in self.vehicles_container[self.arm][self.lc_from]:
                self.vehicles_container[self.arm][self.lc_from].remove(self)
                for v in self.vehicles_container[self.arm][self.lc_from]:
                    if v.index > self.index:
                        v.index -= 1
    
            # 插入目标 lane 的 solid（按纵向坐标插位）
            former, back = [], []
            for v in self.vehicles_container[self.arm][self.lc_to]:
                if self.arm == 1:
                    (former if v.y < self.y else back).append(v)
                elif self.arm == 2:
                    (former if v.x > self.x else back).append(v)
                elif self.arm == 3:
                    (former if v.y > self.y else back).append(v)
                else:
                    (former if v.x < self.x else back).append(v)
            for v in back:
                v.index += 1
            self.vehicles_container[self.arm][self.lc_to] = former + [self] + back
            self.index = len(former)
    
            # 状态复位
            self.lane = self.lc_to
            self.lane_change_complete = True
            self.lc_state = 'idle'
            self.lc_overlay_in_target = False



    
    def change_lane(self, Local_CurrentTime, defaultLaneCentral, Divided_line,vehicles_container, speeds, Phases, signals, rng=None,):
        if not (self.type == "bus" and self.lane == 2):
            return
        
        # 快到截至区间时，不换道
        d = Divided_line[self.arm]

        if (self.arm == 1 and (self.y < d + 60)) or (self.arm == 2 and (self.x + self.length > d - 60)) or (
                self.arm == 3 and (self.y + self.length > d - 60)) or (self.arm == 4 and (self.x < d + 60)):
            return
    
        # 触发：到达计划时间，且尚未执行
        if (self.lane_change_time == Local_CurrentTime) and (not self.lane_change_complete) and (self.lc_state == 'idle'):
            
            global number_of_changed_vehs
            number_of_changed_vehs += 1
            
            # 1) 纵向换道期速度（像素/帧）与时长（秒, int）
            duration_s = self.sample_bus_lane_change_duration(rng or random.Random(0))
            if duration_s <= 0:
                duration_s = 5  # 兜底
            v_long = speeds[self.type] * getattr(self, 'lc_speed_factor', 0.9)  # 执行期降速
    
            # 2) 记录起点
            self.lc_state = 'executing'
            self.lc_from = self.lane
            self.lc_to   = self.target_lane
            self.lc_t0   = float(Local_CurrentTime)
            self.lc_t1   = float(Local_CurrentTime + duration_s)
            self.lc_x0, self.lc_y0 = self.x, self.y
    
            # 3) 横向目标：目标车道中心线
            if self.arm in [1, 3]:
                self.lc_x1 = defaultLaneCentral[(self.arm, self.target_lane, self.type)]
                self.lc_y1 = self.y  # 先置为当前，下一步再计算纵向目标
            else:
                self.lc_y1 = defaultLaneCentral[(self.arm, self.target_lane, self.type)]
                self.lc_x1 = self.x
    
            # 4) 纵向目标：按臂方向 + 速度*时长，并钳制到 stop（若尚未越线）
            #    像素位移 = 像素/帧 * (FPS * 秒)
            delta = v_long * FPS * duration_s
    
            if self.arm == 1:
                # 向上，y 变小；若未越 stop，则 y >= stop
                target_y = self.y - delta
                if self.crossed == 0:
                    target_y = max(self.stop, target_y)
                self.lc_y1 = target_y
    
            elif self.arm == 2:
                # 向右，x 变大；若未越 stop，则 x + length <= stop
                target_x = self.x + delta
                if self.crossed == 0:
                    target_x = min(self.stop - self.length, target_x)
                self.lc_x1 = target_x
    
            elif self.arm == 3:
                # 向下，y 变大；若未越 stop，则 y + length <= stop
                target_y = self.y + delta
                if self.crossed == 0:
                    target_y = min(self.stop - self.length, target_y)
                self.lc_y1 = target_y
    
            else:  # arm == 4
                # 向左，x 变小；若未越 stop，则 x >= stop
                target_x = self.x - delta
                if self.crossed == 0:
                    target_x = max(self.stop, target_x)
                self.lc_x1 = target_x
    
            # 5) 执行期：目标车道 overlay 占用（让目标 lane 后车把我当障碍）
            self.lc_overlay_in_target = True
            for v in vehicles_container[self.arm][self.lc_to]:
                if self not in v.affected:
                    v.affected.append(self)
                if v not in self.affected:
                    self.affected.append(v)
    
            # 6) 执行期降速生效
            self.speed = v_long
        
        # 执行中：持续横移；注意完成后在 _apply_lateral_progress() 里做 solid 切换
        if self.lc_state == 'executing':
            # 只要在执行期，就确保目标 lane 车辆把我当作障碍
            if self.lc_overlay_in_target:
                for v in vehicles_container[self.arm][self.lc_to]:
                    if self not in v.affected:
                        v.affected.append(self)
            # 注意：速度维持降速，直到完成；完成后在 move_and_follow 里会恢复
        


    def change_area(self, defaultLaneCentral, vehicles_container, simulation, Divided_line):
        # Enter SWAL region if crossing divided line
        if (self.lane in [1, 2, 3] and (
            (self.arm == 1 and self.y <= Divided_line[self.arm]) or
            (self.arm == 2 and self.x + self.length >= Divided_line[self.arm]) or
            (self.arm == 3 and self.y + self.length >= Divided_line[self.arm]) or
            (self.arm == 4 and self.x <= Divided_line[self.arm])
        )):
            # For cars in lane 2, pick SWAL 5/6 based on which is emptier near the front
            if self.lane == 2 and self.type == "car":
                lane5_front = [v for v in vehicles_container[self.arm][5] if v in simulation]
                lane6_front = [v for v in vehicles_container[self.arm][6] if v in simulation]
                if self.arm == 1:
                    max5 = max((v.y + v.length) for v in lane5_front) if lane5_front else 0
                    max6 = max((v.y + v.length) for v in lane6_front) if lane6_front else 0
                    self.swal_lane = 5 if max5 <= max6 else 6
                elif self.arm == 2:
                    min5 = min((v.x) for v in lane5_front) if lane5_front else 1000
                    min6 = min((v.x) for v in lane6_front) if lane6_front else 1000
                    self.swal_lane = 5 if min5 >= min6 else 6
                elif self.arm == 3:
                    min5 = min((v.y) for v in lane5_front) if lane5_front else 1000
                    min6 = min((v.y) for v in lane6_front) if lane6_front else 1000
                    self.swal_lane = 5 if min5 >= min6 else 6
                else:
                    max5 = max((v.x + v.length) for v in lane5_front) if lane5_front else 0
                    max6 = max((v.x + v.length) for v in lane6_front) if lane6_front else 0
                    self.swal_lane = 5 if max5 <= max6 else 6

            # Compute affected set
            self.affected = []
            self.affected += vehicles_container[self.arm][self.swal_lane]
            if self.type == "bus":
                self.affected += vehicles_container[self.arm][self.swal_lane + 1]
            if self.swal_lane > 4:
                self.affected += [v for v in vehicles_container[self.arm][self.swal_lane - 1] if v.type == "bus"]

            # Check occupancy beyond divided line
            def beyond_line(v):
                if v.arm == 1:
                    return (v.y + v.length + MOVE_GAP) < Divided_line[v.arm]
                if v.arm == 4:
                    return (v.x + v.length + MOVE_GAP) < Divided_line[v.arm]
                if v.arm == 2:
                    return v.x > Divided_line[v.arm]
                return v.y > Divided_line[v.arm]

            if self.affected:
                self.change_area_bool = all(beyond_line(v) for v in self.affected)
            else:
                self.change_area_bool = True

            if self.change_area_bool:
                # Leave CWL lane and join SWAL lane tail (with bus-right-lane indexing consideration)
                vehicles_container[self.arm][self.lane].remove(self)
                for v in vehicles_container[self.arm][self.lane]:
                    v.index -= 1

                self.lane = self.swal_lane
                if self.arm in [1, 3]:
                    self.x = defaultLaneCentral[(self.arm, self.lane, self.type)]
                else:
                    self.y = defaultLaneCentral[(self.arm, self.lane, self.type)]

                vehicles_container[self.arm][self.lane].append(self)
                addon = []
                if self.lane > 4:
                    addon = [v for v in vehicles_container[self.arm][self.lane - 1] if v.type == "bus"]
                self.index = len(vehicles_container[self.arm][self.lane] + addon) - 1
                if self.type == "bus":
                    # paired index on the right lane
                    self.right_index = len(vehicles_container[self.arm][self.lane + 1] + [v for v in vehicles_container[self.arm][self.lane] if v.type == "bus"]) - 1

                if self.affected:
                    if self.arm in [1, 4]:
                        self.stop = max((v.stop + v.length + MOVE_GAP) if v.is_moving else (v.y + v.length + MOVE_GAP if self.arm == 1 else v.x + v.length + MOVE_GAP)
                                        for v in self.affected)
                    else:
                        self.stop = min((v.stop - v.length - MOVE_GAP) if v.is_moving else (v.x - MOVE_GAP if self.arm == 2 else v.y - MOVE_GAP)
                                        for v in self.affected)
                else:
                    self.stop = self.defaultStop[self.arm]
            else:
                self.stop = Divided_line[self.arm] - 0.5 if self.arm in [1, 4] else Divided_line[self.arm] + 0.5
                self.is_moving = False

    def move_and_follow(self, Local_CurrentTime, Local_floatTime, vehicles_container, signals, Phases, speeds):
        # Do not handle lane-change here; only motion and follow/stop logic
        # Mark crossed
        if (self.crossed == 0) and ((self.arm == 1 and self.y < self.stopLines[self.arm]) or
                                    (self.arm == 2 and self.x + self.length > self.stopLines[self.arm]) or
                                    (self.arm == 3 and self.y + self.length > self.stopLines[self.arm]) or
                                    (self.arm == 4 and self.x < self.stopLines[self.arm])):
            self.crossed = 1

        # Helper to set restart_min_time for follower(s)
        def propagate_start():
            if vehicles_container[self.arm][self.lane] and vehicles_container[self.arm][self.lane][-1].index >= self.index + 1:
                for v in vehicles_container[self.arm][self.lane]:
                    if v.index == self.index + 1:
                        tau = compute_reaction_time(v, lane_type=('SWAL' if v.lane in [4,5,6,7] else 'CWL'), speed_mps=(v.speed * FPS / 2))

                        v.restart_min_time = self.restart_move_time + tau
                if self.lane > 4:
                    for v in vehicles_container[self.arm][self.lane - 1]:
                        if v.type == "bus":
                            if getattr(v, 'right_index', -999) == self.index + 1:
                                tau = compute_reaction_time(v, lane_type=('SWAL' if v.lane in [4,5,6,7] else 'CWL'), speed_mps=(v.speed * FPS / 2))

                                v.restart_min_time = self.restart_move_time + tau
            if self.type == "bus" and len(vehicles_container[self.arm][self.lane + 1]) >= 1:
                if vehicles_container[self.arm][self.lane + 1][-1].index >= getattr(self, 'right_index', -1) + 1:
                    for v in vehicles_container[self.arm][self.lane + 1]:
                        if v.index == getattr(self, 'right_index', -1) + 1:
                            tau = compute_reaction_time(v, lane_type=('SWAL' if v.lane in [4,5,6,7] else 'CWL'), speed_mps=(v.speed * FPS / 2))

                            v.restart_min_time = self.restart_move_time + tau

        # Arm-specific movement with follow rules
        arm = self.arm
        lane = self.lane
        idx = self.index

        # Update stop based on leader or affected set
        if idx >= 1:
            if lane in [1, 2, 3]:
                pre = vehicles_container[arm][lane][idx - 1]
                if pre.is_moving:
                    self.stop = pre.stop + pre.length + MOVE_GAP if arm in [1, 4] else pre.stop - pre.length - MOVE_GAP
                else:
                    self.stop = (pre.y - pre.length - MOVE_GAP) if arm == 1 else (
                                 pre.x - MOVE_GAP if arm == 2 else (
                                 pre.y - MOVE_GAP if arm == 3 else (
                                 pre.x + pre.length + MOVE_GAP)))
            else:
                aff = [v for v in self.affected if v in self.simulation_group]
                if aff:
                    if arm in [1, 4]:
                        self.stop = max((v.stop + v.length + MOVE_GAP) if v.is_moving else ((v.y + v.length + MOVE_GAP) if arm == 1 else (v.x + v.length + MOVE_GAP)) for v in aff)
                    else:
                        self.stop = min((v.stop - v.length - MOVE_GAP) if v.is_moving else ((v.x - MOVE_GAP) if arm == 2 else (v.y - MOVE_GAP)) for v in aff)
                else:
                    self.stop = self.defaultStop[arm]
        elif idx <= 0 and lane in [4, 5, 6, 7]:
            self.stop = self.defaultStop[arm]
        elif idx <= 0 and lane in [1,2,3]:
            self.stop = self.defaultStop[arm]

        # Decide to move based on stop, green, etc.
        phase_idx = Phases[(arm, lane)]
        green_now = signals[phase_idx].green_state == 1

        def do_move(delta):
            # 在 can_move_forward() 之后、任何 do_move() 之前：
            if getattr(self, 'lc_state', 'idle') == 'executing':
                # 执行期由插值推进，不再调用 do_move 进行纵向位移
                # 但跟驰/stop 的计算仍照常，以确保 lc_y1/lc_x1 被正确钳制
                self._apply_lateral_progress(Local_floatTime)
                return

            if arm == 1:
                self.y -= delta
            elif arm == 2:
                self.x += delta
            elif arm == 3:
                self.y += delta
            else:
                self.x -= delta
            
            # 尾部：若不是执行期，这行无效；若是执行期（通过别的路径进入），也能更新可视化
            self._apply_lateral_progress(Local_floatTime)


        def can_move_forward():
            if arm == 1:
                return (self.y > self.stop) or self.crossed == 1 or (idx <= 0 and lane in [4, 5, 6, 7] and green_now)
            if arm == 2:
                return (self.x + self.length < self.stop) or self.crossed == 1 or (idx <= 0 and lane in [4, 5, 6, 7] and green_now)
            if arm == 3:
                return (self.y + self.length < self.stop) or self.crossed == 1 or (idx <= 0 and lane in [4, 5, 6, 7] and green_now)
            else:
                return (self.x > self.stop) or self.crossed == 1 or (idx <= 0 and lane in [4, 5, 6, 7] and green_now)

        if can_move_forward():
            if self.index <= 0 and lane in [4, 5, 6, 7]:
                self.restart_min_time = 0
                self.is_moving = True
                do_move(self.speed)
                
            if not self.is_moving and idx >= 1:
                # restart after waiting
                if Local_floatTime >= self.restart_min_time:
                    self.is_moving = True
                    do_move(self.speed)
                    self.restart_move_time = Local_floatTime
                    propagate_start()
            elif not self.is_moving and idx <= 0 and lane in [4, 5, 6, 7] and Local_floatTime >= self.restart_min_time:
                do_move(self.speed)
                self.is_moving = True
                self.restart_move_time = Local_floatTime
                propagate_start()
            elif self.is_moving:
                do_move(self.speed)
        else:
            self.is_moving = False
            self.restart_move_time = 4000
            
        # 纵向逻辑之后，推进横向
        prev_state = self.lc_state
        self._apply_lateral_progress(Local_floatTime)
        
        # 若刚刚完成，恢复速度
        if prev_state == 'executing' and self.lc_state == 'idle':
            self.speed = speeds[self.type]  # 恢复到常速


    def delete_if_passed(self, Total_currentTime, leaving_time, simulation, vehicles_container, defaultStop):
        if self.crossed == 1 and self in vehicles_container[self.arm][self.lane]:
            vehicles_container[self.arm][self.lane].remove(self)
            group = vehicles_container[self.arm][self.lane]
            right_group = vehicles_container[self.arm][self.lane + 1] if self.type == "bus" and (self.lane + 1) in vehicles_container[self.arm] else []

            for v in (group + right_group if self.type == "bus" else group):
                v.index -= 1
                if self in v.affected:
                    v.affected.remove(self)
                    if self.arm in [1, 4]:
                        v.stop = max((vv.stop + vv.length + MOVE_GAP) for vv in v.affected) if v.affected else defaultStop[v.arm]
                    else:
                        v.stop = min((vv.stop - vv.length - MOVE_GAP) for vv in v.affected) if v.affected else defaultStop[v.arm]

            leaving_time[self.number] = {
                "generate time": self.generate_time,
                "leave time": Total_currentTime,
                "free flow time": self.intitial_L / REAL_SPEED_MS,
                "movement": self.movement,
            }
            simulation.remove(self)


# =========================
# --- Main Simulation ------
# =========================

def run_single_simulation(bus_ratio: float, demand_factor: float, seed: int):
    setup_global_hash()
    
    global number_of_changed_vehs
    number_of_changed_vehs = 0
    
    # Dedicated RNGs for this run
    rng, np_rng = make_rng(seed)

    # Containers
    simulation = pygame.sprite.Group()
    
    global signals,vehicles,Local_CurrentTime,Local_floatTime
    signals = []  # 1-12

    
    vehicles = {
        1: {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 'crossed': 0},
        2: {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 'crossed': 0},
        3: {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 'crossed': 0},
        4: {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 'crossed': 0},
    }

    Divided_line, x_ref, y_ref, defaultLaneCentral, stopLines, defaultStop = build_geometry()
    Phases = build_phases_map()
    defaultSignal = build_default_signal()

    # TrafficSignal objects: 13 entries, 0..12. Keep index 0 as CWL helper (we set it to green always)
    ts0 = TrafficSignal(0, 1)  # CWL lanes helper
    signals.append(ts0)
    for _ in range(12):
        signals.append(TrafficSignal(1, 0))

    # Demand matrix (3 stages) scaled by demand_factor (kept for completeness)
    demand_matrix = {
        1: [[50, 200, 70], [90, 150, 30], [110, 220, 80], [50, 250, 100]],
        2: [[220, 400, 100], [150, 400, 120], [180, 480, 150], [100, 270, 100]],
        3: [[70, 200, 50], [90, 280, 130], [120, 260, 100], [90, 330, 170]],
    }
    for k in demand_matrix:
        demand_matrix[k] = (np.array(demand_matrix[k]) * demand_factor).tolist()

    # Speed map (pixels/frame)
    speeds = {"car": V_SPEED, "bus": V_SPEED}

    # Read vehicles JSON (deterministic processing order)
    global pending
    json_path = JSON_PATH_TEMPLATE.format(demand=str(demand_factor), bus=str(int(bus_ratio * 100)), seed=str(seed))
    pending = []
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            pending = json.load(f)
    else:
        print(f"JSON not found: {json_path}")

    # Sort pending deterministically: by 'time', then arm, then direction, then type
    # direction order map to guarantee stable order: left < through < right
    dir_rank = {"left": 0, "through": 1, "right": 2}
    pending.sort(key=lambda v: (v.get('time', 0), int(v.get('arm', 1)), dir_rank.get(v.get('direction', 'through'), 1), v.get('type', 'car')))

    # Prepare pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("SIMULATION - Deterministic")
    background = pygame.image.load('images/background_grass.png')
    redSignal_pic = pygame.image.load('images/signals/red_1.png')
    greenSignal_pic = pygame.image.load('images/signals/green_1.png')
    font = pygame.font.Font(None, 20)
    clock = pygame.time.Clock()

    # Time counters (deterministic)
    frame_count = 0
    Total_currentTime = 0
    Local_CurrentTime = 0
    Local_floatTime = 0.0
    ts_local_start_frame = 0  # frame index at local horizon start

    # Optimization bookkeeping
    objs = []
    leaving_time = {}

    # Helper: update signal states deterministically (called every simulated second)
    def update_signals(Local_CurrentTime):
        for i in range(1, 13):
            start, dur, amber = defaultSignal[i]
            if (Local_CurrentTime >= int(start)) and (Local_CurrentTime <= math.ceil(start + dur)):
                signals[i].green_state = 1
                signals[i].red_state = 0
            else:
                signals[i].green_state = 0
                signals[i].red_state = 1

    # Helper: compute horizon length from current defaultSignal
    def current_horizon_length():
        return max(defaultSignal[j][0] + defaultSignal[j][1] + defaultSignal[j][2] for j in range(1, 13))

    # Helper: generate vehicles scheduled at this second
    vehicle_number = 0
    
    def generate_vehicles_for_second(sec: int):
        nonlocal vehicle_number
        idx = 0
        EPS = 1e-6
    
        while idx < len(pending):
            t = float(pending[idx].get('time', 0))
            if t > sec + EPS:
                break  # 队首是未来的，先不生成
            # 走到这：t <= sec+EPS（本秒或已过期的小数秒/落后秒），要消费掉它
    
            item = pending[idx]
            d = (item.get('direction') or 'through').lower()
            if d == 'straight':
                d = 'through'
    
            direction = 0 if d == 'left' else 1 if d == 'through' else 2
            lane = 1 if direction == 0 else 2 if direction == 1 else 3
            vtype = item.get('type', 'car')
    
            # SWAL 选道（与你现有规则一致）
            if vtype == 'car':
                swal_lane = 4 if lane == 1 else (rng.choice([5, 6]) if lane == 2 else 7)
            else:
                swal_lane = 4 if lane == 1 else (5 if lane == 2 else 6)
    
            vehicle_number += 1
            Vehicle(
                number=vehicle_number, arm=int(item['arm']), lane=lane, vehicle_type=vtype,
                movement=['left','straight','right'][direction],
                swal_lane=swal_lane, lane_change_time=4000, target_lane=lane, generate_time=sec,
                x_ref=x_ref, y_ref=y_ref, defaultLaneCentral=defaultLaneCentral, defaultStop=defaultStop,
                stopLines=stopLines, Divided_line=Divided_line,
                vehicles_container=vehicles, simulation_group=simulation, speeds=speeds
            )
            pending.pop(idx)  # 消费队首，不要 idx += 1
    
            # If first item > sec, break early (since sorted by time)

    # Main deterministic loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fixed-step time: every FPS frames == 1 simulated second
        if frame_count % FPS == 0:
            # New simulated second starts
            Total_currentTime += 1

            # Determine if still within current horizon
            H_len = current_horizon_length()
            if Local_CurrentTime + 1 <= H_len:
                Local_CurrentTime += 1
                Local_floatTime = (frame_count - ts_local_start_frame) / FPS

                # Update signals once per second
                update_signals(Local_CurrentTime)
                
                # Generate vehicles scheduled exactly at this whole second (after signals/optimization)
                # (Ensures generation order is fixed per second)
                generate_vehicles_for_second(Total_currentTime)

                # Optimization trigger at deterministic steps
                if Local_CurrentTime % STEP == 0:
                    # Build vehicles_info deterministically (sorted by vehicle number)
                    vehicles_info = {}
                    for v in sorted(list(simulation), key=lambda vv: vv.number):
                        vehicles_info[v.number] = {
                            'i': v.arm,
                            'x': v.x + v.length if v.arm == 2 else v.x,
                            'y': v.y + v.length if v.arm == 3 else v.y,
                            'lane': v.lane,
                            'type': v.type,
                            'gen_time': -( (ts_local_start_frame // FPS) - v.generate_time ),  # local age
                            'free flow time': v.intitial_L / REAL_SPEED_MS,
                        }

                    phase_info = {}
                    for p in range(1, 13):
                        start, dur, amber = defaultSignal[p]
                        if Local_CurrentTime > start + dur:
                            phase_info[p] = 'ended'
                        elif Local_CurrentTime < start:
                            phase_info[p] = 'inactive'
                        else:
                            phase_info[p] = 'active'

                    # Call stage1_opt deterministically
                    try:
                        result = stage1_opt_call_with_seed(
                            seed,
                            stage1_opt.optimize_sequence,
                            L1, vehicles_info, phase_info, defaultSignal, Local_CurrentTime
                        )
                        #optimize_sequence(L1,vehicles_info, phase_info, LastSignal, t0)
                        obj, NewSignal, Target_lane, New_Lane_Change_Time, New_Speed = result
                        if obj is not None:
                            objs.append(obj)
                            defaultSignal = NewSignal

                        # Normalize returned maps
                        # lane-change time is local; convert to absolute local seconds
                        New_Lane_Change_Time = {vn: (New_Lane_Change_Time[vn] + Local_CurrentTime) for vn in New_Lane_Change_Time}
                        # Convert speed m/s to px/frame (assuming New_Speed provided in m/s like before)
                        New_Speed = {vn: round(New_Speed[vn] * 2 / FPS, 2) for vn in New_Speed}
                        # Map SWAL 5/6 target back to CWL 1/3
                        Target_lane = {vn: (1 if Target_lane[vn] == 5 else 3) for vn in Target_lane}

                        # Apply back to vehicles deterministically
                        by_id = {v.number: v for v in simulation}
                        for vn, t in New_Lane_Change_Time.items():
                            if vn in by_id:
                                by_id[vn].reset_lane_change(t)
                        for vn, tl in Target_lane.items():
                            if vn in by_id:
                                by_id[vn].reset_target_lane(tl)
                        # store temp speeds map for per-step reset usage
                        temp_speed_map = New_Speed.copy()
                    except Exception as e:
                        # If optimization fails, keep previous signal plan; empty temp speeds
                        temp_speed_map = {}
                    # End optimization step
                else:
                    temp_speed_map = {}

                

            else:
                # New horizon: reset local counters deterministically
                ts_local_start_frame = frame_count
                Local_CurrentTime = 0
                Local_floatTime = 0.0
                temp_speed_map = {}

        # Draw and update world (per frame)
        screen.blit(background, (0, 0))

        # Current Time display
        time_text = font.render(f"Current Time: {Total_currentTime:.0f} s", True, (0, 0, 0))
        screen.blit(time_text, (200, 300))
        local_time_text = font.render(f"Local Time: {Local_floatTime:.0f} s", True, (0, 0, 0))
        screen.blit(local_time_text, (200, 200))

        # Draw signals
        # Note: index 0 is helper; draw 1..12 at fixed coords
        signalCoods = [
            (211*2,219*2),(215*2,219*2),(219*2,219*2),
            (201*2,212*2),(201*2,215*2),(201*2,218*2),
            (208*2,201*2),(204*2,201*2),(200*2,201*2),
            (219*2,209*2),(219*2,205*2),(219*2,201*2),
        ]
        for i in range(1, 13):
            if signals[i].green_state == 1:
                screen.blit(greenSignal_pic, signalCoods[i-1])
            else:
                screen.blit(redSignal_pic, signalCoods[i-1])

        # Vehicle updates (deterministic order by vehicle.number)
        # 1) deletions; 2) draw; 3) apply temp speed; 4) change_lane; 5) change_area; 6) move
        for v in sorted(list(simulation), key=lambda vv: vv.number):
            v.delete_if_passed(Total_currentTime, leaving_time, simulation, vehicles, defaultStop)
        for v in sorted(list(simulation), key=lambda vv: vv.number):
            screen.blit(v.image, [v.x, v.y])
        for v in sorted(list(simulation), key=lambda vv: vv.number):
            v.reset_speed(Local_floatTime, temp_speed_map)
            v.change_lane(Local_CurrentTime, defaultLaneCentral, Divided_line, vehicles, 
                                                   {"car": V_SPEED, "bus": V_SPEED}, Phases, signals)
            v.change_area(defaultLaneCentral, vehicles, simulation, Divided_line)
            v.move_and_follow(Local_CurrentTime, Local_floatTime, vehicles, signals, Phases, {"car": V_SPEED, "bus": V_SPEED})

        pygame.display.update()

        # Advance deterministic frame counter
        frame_count += 1
        if Total_currentTime >= SIM_SECONDS:
            running = False
        clock.tick(FPS)  # keep UI smooth; logic timing is frame_count-based

    # At end, compute summary
    delays = []
    for vid, rec in leaving_time.items():
        delay = max(0.0, (rec["leave time"] - rec["generate time"]) - rec["free flow time"])
        delays.append(delay)
    avg_delay = (sum(delays) / len(delays)) if delays else 0.0

    delays_50 = []
    for vid, rec in leaving_time.items():
        if rec["generate time"] >= 50:
            delay = max(0.0, (rec["leave time"] - rec["generate time"]) - rec["free flow time"])
            delays_50.append(delay)
    avg_delay_50 = (sum(delays_50) / len(delays_50)) if delays_50 else 0.0

    mean_obj = (sum(objs) / len(objs)) if objs else 0.0
    throughput_per_hour = int(len(leaving_time) * 3600 / SIM_SECONDS) if SIM_SECONDS > 0 else 0

    out_name = f"proposed_{L1}_{bus_ratio}_{demand_factor}_{seed}.txt"
    with open(out_name, 'w', encoding='utf-8') as f:
        f.write(str(avg_delay) + "\n")
        f.write(str(avg_delay_50) + "\n")
        f.write(str(mean_obj) + "\n")
        f.write(str(throughput_per_hour) + "\n")

    print("demand_factor=", demand_factor)
    print("bus_ratio=", bus_ratio)
    print("Average delay (all)=", avg_delay, 's')
    print("Average delay (>=50s)=", avg_delay_50, 's')
    print("Mean objective=", mean_obj)
    print("Throughput (veh/h)=", throughput_per_hour)
    print("number_of_changed_vehs",number_of_changed_vehs)
    print("objs",objs)
    


# =========================
# --- Program Entry --------
# =========================
if __name__ == "__main__":
    demand_factor_list = [0.5, 1, 1.5, 2, 2.5]
    bus_ratio_list = [0,0.05,0.1,0.15,0.25]
    random_seed_list = [38, 40, 42, 44, 46]
    
 
    for RUN_DEMAND in [1]:
        for RUN_BUS_RATIO in [0.2]:
            for RUN_SEED in [44]:
                run_single_simulation(bus_ratio=RUN_BUS_RATIO, demand_factor=RUN_DEMAND, seed=RUN_SEED)
    
    

from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np
from typing import Optional, Sequence, Tuple, Dict, Any
import os
import json
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional
import time

import mss
from enum import Enum
import pyautogui
import pygetwindow as gw
import keyboard
import threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import FuncFormatter

@dataclass
class Lane:
    idx: int
    y0: int
    y1: int
    height: int
    lane_type: LaneType
    mean_rgb: Tuple[float, float, float]
    confidence: float = 0.0
@dataclass
class CameraPose:
    offset_y: float = 0.0
    scale_y: float = 1.0

@dataclass
class CaptureConfig:
    monitor: int = 1
    region: Optional[Tuple[int, int, int, int]] = None  

class ScreenCapture:
    def __init__(self, config: CaptureConfig):
        self.config = config
        self.sct = mss.mss()
        self._monitor = self._select_monitor()
    def _select_monitor(self):
        mons = self.sct.monitors
        idx = self.config.monitor if 0 <= self.config.monitor < len(mons) else 1
        mon = mons[idx]
        if self.config.region is not None:
            left, top, w, h = self.config.region
            mon = {"left": left, "top": top, "width": w, "height": h}
        return mon
    def next_frame(self) -> np.ndarray:
        img = np.array(self.sct.grab(self._monitor))        
        return img[:, :, :3]       

class StageTimer:
    def __init__(self):
        self.t0 = None
        self.ms = 0.0
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.ms = (time.perf_counter() - self.t0) * 1000.0

class FPSMeter:
    def __init__(self, window: int = 60):
        self.window = window
        self.times = deque(maxlen=window)
        self.last = None
    def tick(self) -> float:
        now = time.perf_counter()
        if self.last is not None:
            dt = now - self.last
            self.times.append(dt)
        self.last = now
        return self.fps
    @property
    def fps(self) -> float:
        if not self.times:
            return 0.0
        avg = sum(self.times) / len(self.times)
        return 1.0 / avg if avg > 0 else 0.0

@dataclass
class Border:
    id: int
    x: int
    y0: int
    y1: int
    height: int
    is_boundary: bool = False   

class LaneVelocityEstimator:
    def __init__(self, lane_h:int, history:int=30, min_samples:int=3,
                 trim_frac:float=0.2, ignore_abs_below:float=10.0, ignore_abs_above:float=600.0,
                 offset:int=0,
                 disp_history_seconds: float = 60.0):                                               
        self.lane_h = lane_h
        self.history = history
        self.min_samples = min_samples
        self.trim_frac = float(np.clip(trim_frac, 0.0, 0.49))
        self.ignore_abs_below = ignore_abs_below
        self.ignore_abs_above = ignore_abs_above
        self._last_xt = {}                                           
        self._lane_samples = defaultdict(lambda: deque(maxlen=self.history))                         
                                                                     
        self.offset = int(offset)
                                                                             
                                                                                                                   
        self._disp_store = defaultdict(lambda: {
            "t0": None,                                                  
            "x0": None,                              
            "t_abs": deque(),                                        
            "disp": deque(),                                                    
            "last_seen": 0.0,                                           
        })
        self._disp_window_s = float(disp_history_seconds)
                                                      
    def set_offset(self, offset:int):
        self.offset = int(offset)
    def _lane_index_for_border(self, b: "Border") -> int:
        y_mid = (b.y0 + b.y1) // 2
        return int((y_mid - self.offset) // self.lane_h)
    def update(self, borders_now: list["Border"], t_now: float, *, fps_hint: float|None=None):
        for b in borders_now:
            lane_idx = self._lane_index_for_border(b)
                                                                  
            if b.id in self._last_xt:
                x_prev, t_prev = self._last_xt[b.id]
                dt = max(1e-6, t_now - t_prev)
                vx = (float(b.x) - float(x_prev)) / dt        
                if self.ignore_abs_below <= abs(vx) <= self.ignore_abs_above:
                    self._lane_samples[lane_idx].append(vx)
            self._last_xt[b.id] = (float(b.x), float(t_now))
                                                            
            rec = self._disp_store[b.id]
            if rec["t0"] is None:
                rec["t0"] = float(t_now)
                rec["x0"] = float(b.x)
            disp = float(b.x) - float(rec["x0"])                                             
            rec["t_abs"].append(float(t_now))                                                     
            rec["disp"].append(disp)
            rec["last_seen"] = float(t_now)
                                                                    
            t_cut = float(t_now) - self._disp_window_s
            dq_t = rec["t_abs"]; dq_d = rec["disp"]
            while dq_t and dq_t[0] < t_cut:
                dq_t.popleft(); dq_d.popleft()
    def estimate_per_lane(self) -> dict[int, float]:
        lane_vx = {}
        for lane_idx, samples in self._lane_samples.items():
            if len(samples) < self.min_samples:
                continue
            lane_vx[lane_idx] = float(samples[-1])                  
        return lane_vx
    def assign_border_velocities(self, borders_now: list["Border"], lane_vx: dict[int, float]) -> dict[int, dict]:
        out = {}
        for b in borders_now:
            lane_idx = self._lane_index_for_border(b)
            if lane_idx in lane_vx:
                out[b.id] = {"vx": lane_vx[lane_idx]}
        return out
                                        
    def get_displacement_log(self) -> dict[int, dict]:
        out = {}
        for bid, rec in self._disp_store.items():
            t0 = rec["t0"]
            if t0 is None or not rec["t_abs"]:
                continue
                                                                  
            t_rel = [float(t) - float(t0) for t in rec["t_abs"]]
            out[bid] = {
                "t_rel": t_rel,
                "disp": list(rec["disp"]),
                "last_seen": float(rec["last_seen"]),
            }
        return out

class GameOverDetector:
    def __init__(self,
                 hsv_lo=(17, 160, 200),                                            
                 hsv_hi=(30, 255, 255),
                 min_frac: float = 0.0015,                       
                 consec_needed: int = 3):
        self.lo = np.array(hsv_lo, np.uint8)
        self.hi = np.array(hsv_hi, np.uint8)
        self.min_frac = float(min_frac)
        self.consec_needed = int(consec_needed)
        self._streak = 0
        self.triggered = False
                                         
        self._k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    def update(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lo, self.hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._k_open,  iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k_close, iterations=1)
        frac = cv2.countNonZero(mask) / mask.size
        seen = frac >= self.min_frac
        self._streak = (self._streak + 1) if seen else 0
        if self._streak >= self.consec_needed:
            self.triggered = True
        else:
            self.triggered = False
        return self.triggered, float(frac)

class BotStateMachine:
    def __init__(self, wait_time: float = 0.5):
        self.wait_time = wait_time
        self.reset()
    def reset(self):
        self.state = "start"
        self.last_change = time.perf_counter()
    def update(self, *, move_cmd: str):
        now = time.perf_counter()
        if self.state == "start":
            pyautogui.press('up')
            self.state = "moving"
            self.last_change = now
                                                                
        elif self.state == "moving":
                                                          
            if now - self.last_change >= self.wait_time:
                self.state = "static"
                self.last_change = now
        elif self.state == "static":
            if move_cmd and move_cmd != 'wait':
                pyautogui.press(move_cmd)
                self.state = "moving"
                self.last_change = now

@dataclass
class BorderSideFree:
    left_px: int
    right_px: int
    left_ok: bool
    right_ok: bool
    min_clear_px: int

class XTBandVelocity:
    def __init__(self,
                 max_seconds: float = 2.0,
                 resample_hz: float = 60.0,
                 x_blur_px: float = 1.0,
                 min_rows_for_fit: int = 12,
                 hough_thresh: int = 12,
                 min_seg_len: int = 10,
                 max_seg_gap: int = 5,
                 slope_abs_min: float = 0.10,             
                 slope_abs_max: float = 4.0,              
                 collinear_tol: float = 0.12,                                  
                 intercept_tol: float = 12.0,                                          
                 allow_wrap: bool = True,
                 close_time_gaps_k: int = 0                                                            
                 ):
                   
        self.resample_hz = float(resample_hz)
        self.dt_grid = 1.0 / self.resample_hz
        self.max_seconds = float(max_seconds)
        self.max_rows = int(round(self.max_seconds / self.dt_grid))
                          
        self.x_blur_px = float(x_blur_px)
        self.min_rows_for_fit = int(min_rows_for_fit)
        self.hough_thresh = int(hough_thresh)
        self.min_seg_len = int(min_seg_len)
        self.max_seg_gap = int(max_seg_gap)
        self.slope_abs_min = float(slope_abs_min)
        self.slope_abs_max = float(slope_abs_max)
        self.collinear_tol = float(collinear_tol)
        self.intercept_tol = float(intercept_tol)
        self.allow_wrap = bool(allow_wrap)
        self.close_time_gaps_k = int(close_time_gaps_k)
                        
        self._band_bufs: List[deque] = []                                              
        self._xt_imgs: List[Optional[np.ndarray]] = []                                            
        self._row_idx: List[int] = []                                                             
        self._filled_rows: List[int] = []                                                            
        self._widths: List[int] = []                           
        self._last_grid_time: List[Optional[float]] = []                                         
        self._last_sign: List[int] = []                                      
                            
        self.last_tracks: Dict[int, List[Tuple[Tuple[int,int],Tuple[int,int]]]] = {}
                                                  
    def push(self, occ_lines: List[np.ndarray], t_now: float) -> None:
        self._ensure_bands(len(occ_lines))
        for b, line in enumerate(occ_lines):
            if line is None:
                continue
            li = self._coerce_binary_1d(line)
            W = int(li.shape[0])
                                                     
            if self._xt_imgs[b] is None or self._widths[b] != W:
                self._xt_imgs[b] = np.zeros((self.max_rows, W), np.uint8)
                self._widths[b] = W
                self._row_idx[b] = -1
                self._filled_rows[b] = 0
                self._last_grid_time[b] = None
                                              
            self._advance_to_time(b, float(t_now))
                                                                                        
            ridx = self._row_idx[b]
            if ridx >= 0:
                self._xt_imgs[b][ridx] = np.maximum(self._xt_imgs[b][ridx], li * 255)
                                                                            
            self._band_bufs[b].append((float(t_now), li))
                                           
            t_cut = float(t_now) - self.max_seconds
            q = self._band_bufs[b]
            while q and q[0][0] < t_cut:
                q.popleft()

    def estimate(self,
                 direction_hint: Optional[List[int]] = None
                 ) -> Tuple[List[float], List[bool], Dict]:
        vx_list = [0.0] * len(self._xt_imgs)
        ready = [False] * len(self._xt_imgs)
        dbg = {"rows": [], "nz": [], "used": []}
        self.last_tracks.clear()
        for b in range(len(self._xt_imgs)):
            xt = self.get_xt(b, processed=True)                                           
            if xt is None:
                dbg["rows"].append(0); dbg["nz"].append(0); dbg["used"].append("empty")
                continue
            rows, W = xt.shape[:2]
            nz = int((xt > 0).sum())
            dbg["rows"].append(rows); dbg["nz"].append(nz)
            if rows < self.min_rows_for_fit or nz < 8:
                dbg["used"].append("not_ready")
                continue
                                        
            edges = cv2.Canny(xt, 20, 40, apertureSize=3)
            sign_hint = 0
            if direction_hint and b < len(direction_hint):
                sign_hint = 1 if direction_hint[b] == 1 else (-1 if direction_hint[b] == -1 else 0)
            segs = self._hough_segments(edges, sign_hint)
            tracks = self._merge_segments(segs, xt.shape[1])
            self.last_tracks[b] = tracks
            best_v = None
            used = "hough"
            if tracks:
                (x0, y0), (x1, y1) = tracks[0]
                m = self._slope((x0, y0), (x1, y1))
                if m is not None:
                    best_v = m / self.dt_grid
            if best_v is None:
                used = "centroid"
                a = self._centroid_tls_slope(xt)
                if a is not None:
                    best_v = a / self.dt_grid
            if best_v is not None:
                                            
                s_hint = sign_hint
                s_prev = self._last_sign[b]
                s_use = s_hint if s_hint in (+1, -1) else (s_prev if s_prev in (+1, -1) else 0)
                if s_use in (+1, -1):
                    best_v = float(s_use) * abs(float(best_v))
                    self._last_sign[b] = s_use
                else:
                    self._last_sign[b] = int(math.copysign(1, best_v)) if best_v != 0 else 0
                vx_list[b] = float(best_v)
                ready[b] = True
                dbg["used"].append(used)
            else:
                dbg["used"].append("none")
        return vx_list, ready, dbg
    
    def get_xt(self, band_idx: int, processed: bool = False) -> Optional[np.ndarray]:
        if band_idx < 0 or band_idx >= len(self._xt_imgs):
            return None
        xt = self._xt_imgs[band_idx]
        if xt is None or self._filled_rows[band_idx] == 0:
            return None
        ridx = self._row_idx[band_idx]
        rows, W = xt.shape[:2]                                 
                                         
        start = (ridx + 1) % rows if ridx >= 0 else 0
        xt_snap = np.roll(xt, shift=-(start), axis=0).copy()
                                                                            
        if processed:
            x = xt_snap
            if self.x_blur_px > 0:
                k = max(1, int(2 * round(self.x_blur_px) + 1))
                x = cv2.GaussianBlur(x, (k, 1), self.x_blur_px)
            if self.close_time_gaps_k and self.close_time_gaps_k >= 3 and self.close_time_gaps_k % 2 == 1:
                x = cv2.morphologyEx(x, cv2.MORPH_CLOSE,
                                     np.ones((self.close_time_gaps_k, 1), np.uint8))
            return x
        return xt_snap

    def render_xt_panel(self, band_idx: int,
                        show_tracks: bool = True,
                        show_velocity: bool = True,
                        font_scale: float = 0.5) -> np.ndarray:
        xt = self.get_xt(band_idx, processed=False)
        if xt is None:
            return np.zeros((40, 200, 3), np.uint8)
        rows, W = xt.shape[:2]
        vis = cv2.cvtColor(xt, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis, (0, 0), (W - 1, rows - 1), (90, 90, 90), 1)
                                                   
        tracks = self.last_tracks.get(band_idx, [])
        if show_tracks and tracks:
            for (x0, y0), (x1, y1) in tracks[:4]:
                cv2.line(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
        if show_velocity and tracks:
            (x0, y0), (x1, y1) = tracks[0]
            m = self._slope((x0, y0), (x1, y1))
            if m is not None:
                v = m / self.dt_grid
                cv2.putText(vis, f"v~{v:.1f} px/s", (6, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"dt={self.dt_grid*1000:.0f}ms", (6, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1, cv2.LINE_AA)
        return cv2.resize(vis, (W * 2, rows * 2), interpolation=cv2.INTER_NEAREST)
                                               
    def _ensure_bands(self, n_bands: int) -> None:
        grow = n_bands - len(self._xt_imgs)
        if grow <= 0:
            return
        self._band_bufs += [deque() for _ in range(grow)]
        self._xt_imgs += [None for _ in range(grow)]
        self._row_idx += [-1 for _ in range(grow)]
        self._filled_rows += [0 for _ in range(grow)]
        self._widths += [0 for _ in range(grow)]
        self._last_grid_time += [None for _ in range(grow)]
        self._last_sign += [0 for _ in range(grow)]

    @staticmethod
    def _coerce_binary_1d(line: np.ndarray) -> np.ndarray:
        li = np.asarray(line)
        if li.ndim != 1:
            li = li.reshape(-1)
        if li.dtype != np.uint8:
            li = li.astype(np.uint8)
        if li.max() > 1:
            li = (li > 0).astype(np.uint8)
        return li
    
    def _advance_to_time(self, band: int, t_now: float) -> None:
        last = self._last_grid_time[band]
        if last is None:
                                       
            self._row_idx[band] = 0
            self._filled_rows[band] = 1
            self._xt_imgs[band][0].fill(0)
            self._last_grid_time[band] = t_now
            return
        dt = t_now - last
        if dt < 0:
                                                             
            self._xt_imgs[band][self._row_idx[band]].fill(0)
            self._last_grid_time[band] = t_now
            return
        steps = int(math.floor(dt / self.dt_grid))
        if steps <= 0:
                                                                           
            return
                                                     
        if steps >= self.max_rows:
            self._xt_imgs[band].fill(0)
            self._row_idx[band] = 0
            self._filled_rows[band] = 1
            self._last_grid_time[band] = t_now
            return
                                                     
        R = self.max_rows
        ridx = self._row_idx[band]
        for _ in range(steps):
            ridx = (ridx + 1) % R
            self._xt_imgs[band][ridx].fill(0)
        
        self._row_idx[band] = ridx
        self._filled_rows[band] = min(R, self._filled_rows[band] + steps)
        self._last_grid_time[band] = last + steps * self.dt_grid

    @staticmethod
    def _slope(p1: Tuple[int, int], p2: Tuple[int, int]) -> Optional[float]:
        (x1, y1), (x2, y2) = p1, p2
        dy = (y2 - y1)
        if abs(dy) < 1e-6:
            return None
        return (x2 - x1) / dy
    
    def _hough_segments(self, edges: np.ndarray, sign_hint: int):
        rows, W = edges.shape[:2]
        segs = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                               threshold=max(self.hough_thresh, rows // 8),
                               minLineLength=self.min_seg_len,
                               maxLineGap=self.max_seg_gap)
        out = []
        if segs is None:
            return out
        for x1, y1, x2, y2 in segs[:, 0, :]:
            if y2 < y1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            m = self._slope((x1, y1), (x2, y2))
            if m is None:
                continue
            if not (self.slope_abs_min <= abs(m) <= self.slope_abs_max):
                continue
            sgn = 1 if m > 0 else -1
            if sign_hint in (+1, -1) and sgn != sign_hint:
                continue
            out.append(((x1, y1), (x2, y2), float(m)))
        out.sort(key=lambda s: s[0][1])              
        return out
    
    def _merge_segments(self, segs, width: int):
        if not segs:
            return []
        tracks = []
        for (p1, p2, m) in segs:
            b = p1[0] - m * p1[1]               
            placed = False
            for tr in tracks:
                if abs(m - tr['m']) > self.collinear_tol:
                    continue
                db = abs(b - tr['b'])
                if self.allow_wrap:
                    db = min(db, abs((b + width) - tr['b']), abs((b - width) - tr['b']))
                if db > self.intercept_tol:
                    continue
                y_gap = p1[1] - tr['y_max']
                if y_gap < -5 or y_gap > max(20, int(0.25 * width)):
                    continue
                tr['segs'].append((p1, p2))
                tr['y_max'] = max(tr['y_max'], p2[1])
                ys, xs = [], []
                for (q1, q2) in tr['segs']:
                    ys += [q1[1], q2[1]]
                    xs += [q1[0], q2[0]]
                A = np.vstack([np.array(ys, np.float32), np.ones(len(ys), np.float32)]).T
                a, b_ref = np.linalg.lstsq(A, np.array(xs, np.float32), rcond=None)[0]
                tr['m'] = float(a)
                tr['b'] = float(b_ref)
                placed = True
                break
            if not placed:
                tracks.append({'segs': [(p1, p2)], 'm': float(m), 'b': float(b), 'y_max': p2[1]})
        merged = []
        for tr in tracks:
            y0 = min(min(s[0][1], s[1][1]) for s in tr['segs'])
            y1 = max(max(s[0][1], s[1][1]) for s in tr['segs'])
            x0 = int(round(tr['m'] * y0 + tr['b']))
            x1 = int(round(tr['m'] * y1 + tr['b']))
            x0 = int(np.clip(x0, 0, width - 1))
            x1 = int(np.clip(x1, 0, width - 1))
            if (y1 - y0) >= self.min_seg_len:
                merged.append(((x0, y0), (x1, y1)))
        merged.sort(key=lambda e: (e[1][1] - e[0][1]), reverse=True)
        return merged
    
    @staticmethod
    def _centroid_tls_slope(xt: np.ndarray) -> Optional[float]:
        ys, xs = [], []
        rows = xt.shape[0]
        for r in range(rows):
            cols = np.where(xt[r] > 0)[0]
            if len(cols):
                ys.append(r)
                xs.append(float(cols.mean()))
        if len(xs) < 2:
            return None
        y = np.array(ys, np.float32)
        x = np.array(xs, np.float32)
        A = np.vstack([y, np.ones_like(y)]).T
        a, b = np.linalg.lstsq(A, x, rcond=None)[0]
        return float(a)

class SideFreePlanner:
    def __init__(self, movement_time: float = 0.40, *, occ_free_thresh: float = 0.80, band_step: int = 1, lane_h: int = 22, frame_w: int, frame_shape):
        self.movement_time = float(movement_time)
        self.occ_free_thresh = float(np.clip(occ_free_thresh, 0.0, 1.0))
        self.band_step = int(band_step)                                                                             
        self._occ_lines: Optional[List[np.ndarray]] = None
                                           
        self._proj_vx_min_abs      = 1.0
        self._proj_ignore_boundary = True
        self._proj_ignore_edge_px  = 2
        self._proj_thicken_px      = 0
                
        self._direction: str = "down"
        self._n_extra: int = 0
        self._lane_h: int = 22
        self._frame_w = frame_w
        self._frame_shape = frame_shape
        self.block: Dict = {}
        self._consensus_N: int = 5                                            
        self._decision_hist: deque[str] = deque(maxlen=self._consensus_N)
        self._last_cmd_raw: Optional[str] = None                               
        self._last_cmd: Optional[str] = None         

    def _update_decision_history(self, cmd: str) -> None:
        self._last_cmd_raw = cmd
        self._decision_hist.append(cmd)

    def _consensus_direction(self) -> Optional[str]:
        if len(self._decision_hist) < self._consensus_N:
            return None
        first = self._decision_hist[0]
        if first not in ("up", "down", "left", "right"):
            return None
        if all(c == first for c in self._decision_hist):
            return first
        return None
    
    def update_occ_lines(self, occ_lines: List[np.ndarray]):
        self._occ_lines = occ_lines
    @staticmethod
    def _band_index_for_border(b: "Border", bands: List[Tuple[int, int]]) -> Optional[int]:
        y_mid = (b.y0 + b.y1) * 0.5
        for i, (y0, y1) in enumerate(bands):
            if y0 <= y_mid <= (y1 - 1):
                return i
        return None
    @staticmethod
    def compute_block(
        bands: List[Tuple[int, int]],
        lane_h: int,
        char_band_idx: Optional[int],
        char_center: Optional[Tuple[int, int]],
        frame_shape: Tuple[int, int],
        x: int = 0,                                                     
        y: int = 1,                                                                
    ) -> Dict:
        y = -(1-y)
        H, W = frame_shape
        if char_center is None:
            return []
        cx, cy = map(int, char_center)
                                                          
        base_idx = None
        for i, (y0b, y1b) in enumerate(bands):
            if y0b <= cy <= (y1b - 1):
                base_idx = i
                break
                                                                            
        if base_idx is None:
            if char_band_idx is not None and 0 <= char_band_idx < len(bands):
                base_idx = int(char_band_idx)
            else:
                return []
                                                                    
        bi = base_idx + (int(y) - 1)
        if not (0 <= bi < len(bands)):
            return []
        y0_band, y1_band = bands[bi]
        if y1_band <= y0_band:
            return []
                                                                                             
        half = lane_h // 2
        x0 = (cx - half) + int(x) * lane_h
        x1 = x0 + lane_h - 1
                                        
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if x1 >= W:
            shift = x1 - (W - 1)
            x0 = max(0, x0 - shift)
            x1 = W - 1
        if x1 <= x0:                              
            x1 = min(W - 1, x0 + 1)
                                                                                    
        y_top = max(y0_band, y1_band - lane_h)
        y_bot = min(H, y_top + lane_h)
        if y_bot - 1 <= y_top:
            return []
        return {
            "bi": int(bi),
            "x0": int(x0),
            "x1": int(x1),
            "y_top": int(y_top),
            "y_bot": int(y_bot),
            "is_current": True,
        }
    
    def block_has_min_free_fraction(
        block: Dict,
        occ_lines: Sequence[Optional[np.ndarray]],
    ) -> bool:
                      
        if not block or not occ_lines:
            return False
        bi = int(block.get("bi", -1))
        if not (0 <= bi < len(occ_lines)):
            return False
        line = occ_lines[bi]
        if line is None:
            return False
        line = np.asarray(line)
        if line.ndim != 1 or line.size == 0:
            return False
                                                           
        W_disp = int(240)
        if W_disp <= 0:
            return False
        W_line = int(line.shape[0])
        try:
            x0d = int(block["x0"])
            x1d = int(block["x1"])
        except Exception:
            return False
        if x1d < x0d:
            x0d, x1d = x1d, x0d             
        if W_line == W_disp:
            x0 = max(0, min(W_line - 1, x0d))
            x1 = max(0, min(W_line - 1, x1d))
        else:
            scale = W_line / max(1, W_disp)
            x0 = int(np.clip(round(x0d * scale), 0, W_line - 1))
            x1 = int(np.clip(round(x1d * scale), 0, W_line - 1))
            if x1 < x0:
                x0, x1 = x1, x0
                        
        win = line[x0:x1 + 1]
        if win.size == 0:
            return False
                                                           
        free_frac = float((win == 0).sum()) / float(win.size)
        required = max(0.0, min(1.0, float(1.0)))                                    
        return free_frac >= required
    
    def block_is_clear_now_and_future(
        self,
        block: Dict,
        *,
        n: int = 1,
        bands: List[Tuple[int, int]],
        borders: List["Border"],
        vel_map: Dict[int, Dict[str, float]],
        lane_types: List["LaneType"],
    ) -> bool:
                      
        if not block:
            return False
        bi = int(block.get("bi", -1))
        if not (0 <= bi < len(bands)) or not (0 <= bi < len(lane_types)):
            return False
                                                                                  
        occ_lines_now = getattr(self, "_occ_lines", None)
        if occ_lines_now is None or not occ_lines_now:
            return (lane_types[bi] != LaneType.ROAD)
                              
        if lane_types[bi] != LaneType.ROAD and lane_types[bi] != LaneType.WATER_PLATFORM:
            return True
                                            
        n = max(0, int(n))
        dt_total = float(n) * float(self.movement_time)
        dt_mid   = 0.5 * dt_total                             
                                                     
        occ_lines_mid = project_occ_lines(
            occ_lines=occ_lines_now,
            bands=bands,
            borders=borders,
            vel_map=vel_map,
            frame_w=self._frame_w,
            movement_time=dt_mid,
            vx_min_abs=self._proj_vx_min_abs,
            ignore_boundary=self._proj_ignore_boundary,
            ignore_edge_px=self._proj_ignore_edge_px,
            thicken_px=self._proj_thicken_px
        )
        occ_lines_future = project_occ_lines(
            occ_lines=occ_lines_now,
            bands=bands,
            borders=borders,
            vel_map=vel_map,
            frame_w=self._frame_w,
            movement_time=dt_total,
            vx_min_abs=self._proj_vx_min_abs,
            ignore_boundary=self._proj_ignore_boundary,
            ignore_edge_px=self._proj_ignore_edge_px,
            thicken_px=self._proj_thicken_px
        )
                                  
        if not (0 <= bi < len(occ_lines_now)) or not (0 <= bi < len(occ_lines_mid)) or not (0 <= bi < len(occ_lines_future)):
            return False
        line_now    = occ_lines_now[bi]
        line_mid    = occ_lines_mid[bi]
        line_future = occ_lines_future[bi]
        if line_now is None or line_mid is None or line_future is None:
            return False
        if line_now.size == 0 or line_mid.size == 0 or line_future.size == 0:
            return False
                                                           
        W_disp = int(self._frame_w)
        W_line = int(line_now.shape[0])
        x0d, x1d = int(block["x0"]), int(block["x1"])
        if x1d < x0d:
            return False
        if W_line == W_disp:
            x0 = max(0, min(W_line - 1, x0d))
            x1 = max(0, min(W_line - 1, x1d))
        else:
            scale = W_line / max(1, W_disp)
            x0 = int(np.clip(round(x0d * scale), 0, W_line - 1))
            x1 = int(np.clip(round(x1d * scale), 0, W_line - 1))
            if x1 < x0:
                x0, x1 = x1, x0
                         
        win_now    = line_now[x0:x1+1]
        win_mid    = line_mid[x0:x1+1]
        win_future = line_future[x0:x1+1]
        if win_now.size == 0 or win_mid.size == 0 or win_future.size == 0:
            return False
                        
        free_now    = float((win_now    == 0).sum()) / float(win_now.size)
        free_mid    = float((win_mid    == 0).sum()) / float(win_mid.size)
        free_future = float((win_future == 0).sum()) / float(win_future.size)
                                         
        thresh = self.occ_free_thresh
        return (free_now >= thresh) and (free_mid >= thresh) and (free_future >= thresh)
    
    def plan_blocks(
        self,
        *,
        bands: List[Tuple[int, int]],
        borders: List["Border"],
        vel_map: Dict[int, Dict[str, float]],
        lane_types: List["LaneType"],
                                                
        char_band_idx: Optional[int],
        char_center: Optional[Tuple[int, int]],
    ) -> str:
        if not char_band_idx:
            self.block = None
            return
        self.block = self.compute_block(
            bands=bands,
            lane_h=self._lane_h,
            char_band_idx=char_band_idx,
            char_center=char_center,
            frame_shape=self._frame_shape,
            x=0,
            y=1,
        )
        can_go_up = self.block_is_clear_now_and_future(
            self.block,
            n=1,                                               
            bands=bands,
            borders=borders,
            vel_map=vel_map,
            lane_types=lane_types,
        )
        raw_cmd = 'wait'              
        if can_go_up:
            raw_cmd = 'up'                                     
        self._update_decision_history(raw_cmd)
        consensus = self._consensus_direction()                                            
        self._last_cmd = consensus if consensus is not None else 'wait'
        return self._last_cmd
    
class OccLinesMeanStabilizer:
    def __init__(self, n_bands: int, width: int, window: int = 8,
                 free_lo: float = 0.35, occ_hi: float = 0.65):
        self.N = max(1, int(window))
        self.free_lo = float(max(0.0, min(1.0, free_lo)))
        self.occ_hi  = float(max(0.0, min(1.0, occ_hi)))
        self._init_buffers(n_bands, width)

    def _init_buffers(self, n_bands: int, width: int):
        self.shape = (int(n_bands), int(width))
        nb, W = self.shape
        self.buf  = np.zeros((self.N, nb, W), np.uint8)                             
        self.sum  = np.zeros((nb, W), np.uint16)                       
        self.idx  = 0
        self.count = 0
        self.last = None      

    def reset(self):
        self._init_buffers(*self.shape)

    def update(self, lines: list[np.ndarray]) -> list[np.ndarray]:
        if not lines:
            return []
        nb = len(lines)
        W  = int(lines[0].shape[0])
                                 
        if self.shape != (nb, W):
            self._init_buffers(nb, W)
        mat = np.stack([(np.asarray(l, np.uint8) & 1).reshape(-1) for l in lines], axis=0)
        old = self.buf[self.idx]
        self.sum -= old
        self.buf[self.idx] = mat
        self.sum += mat
        self.idx = (self.idx + 1) % self.N
        if self.count < self.N:
            self.count += 1
        mean = self.sum.astype(np.float32) / float(self.count)
        if self.last is None:
            out = (mean >= 0.5).astype(np.uint8)
        else:
            out = self.last.copy()
            out[mean >= self.occ_hi] = 1
            out[mean <= self.free_lo] = 0
                                                 
        self.last = out
                                                    
        return [out[i].copy() for i in range(out.shape[0])]


K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

class LaneType(Enum):
    GREEN = 0
    ROAD = 1
    WATER_LILYPAD = 2
    WATER_PLATFORM = 3
    TRACK = 4
    UNKNOWN = 9

def draw_hud(frame_bgr, stats: Dict[str, float], origin=(10, 20)):
    x, y = origin
    for k, v in stats.items():
        if isinstance(v, float):
            text = f"{k}: {v:.1f}"
        else:
            text = f"{k}: {v}"
        cv2.putText(frame_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        y += 22
    return frame_bgr

def draw_lane_bands(frame_bgr, lanes: List[Lane], show_labels: bool = True):
    h, w = frame_bgr.shape[:2]
    for lane in lanes:
        color = (0, 200, 255)
        y0, y1 = int(lane.y0), int(lane.y1)
        cv2.rectangle(frame_bgr, (0, y0), (w-1, y1), color, 1)
        if show_labels:
            label = lane.lane_type.name
            cv2.putText(frame_bgr, f"{lane.idx}:{label}", (5, max(14, y0+14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    return frame_bgr
                               
def _box_density(mask01: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.blur(mask01.astype(np.float32), (ksize, ksize), borderType=cv2.BORDER_REPLICATE)

def fill_small_gaps_1d(mask: np.ndarray, max_gap: int) -> np.ndarray:
    m = mask.astype(bool).copy()
    W = m.shape[0]
    i = 0
    while i < W:
        if m[i]:
            i += 1
            continue
        j = i
        while j < W and not m[j]:
            j += 1
                         
        left_true  = (i - 1) >= 0 and m[i - 1]
        right_true = (j < W) and m[j] if j < W else False
        if left_true and right_true and (j - i) <= max_gap:
            m[i:j] = True
        i = j
    return m
                                                                        
def _lab_chroma(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    da = a.astype(np.int16) - 128
    db = b.astype(np.int16) - 128
    return cv2.sqrt((da*da + db*db).astype(np.float32))

def gray_density_map_hsv_from(hsv, lab, s_max=80, v_min=80, v_max=130, ksize=11):
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h_gate = cv2.inRange(H, 95, 135)
    s_gate = cv2.inRange(S, 0, s_max)
    v_gate = cv2.inRange(V, v_min, v_max)
    hsv_gate = cv2.bitwise_and(h_gate, cv2.bitwise_and(s_gate, v_gate))
    aL = lab[..., 1].astype(np.float32)
    bL = lab[..., 2].astype(np.float32)
    da = cv2.absdiff(aL, 128.0)
    db = cv2.absdiff(bL, 128.0)
    chroma = cv2.sqrt((aL - 128.0) * (aL - 128.0) + (bL - 128.0) * (bL - 128.0))
    lab_gate = ((da <= 10.0) & (db <= 12.0) & (chroma <= 16.0)).astype(np.uint8) * 255
    gray_255 = cv2.bitwise_and(hsv_gate, lab_gate)
    gray_255 = cv2.morphologyEx(gray_255, cv2.MORPH_OPEN, K3, iterations=1)
    gray_255 = cv2.morphologyEx(gray_255, cv2.MORPH_CLOSE, K3, iterations=1)
    gray01 = (gray_255 > 0).astype(np.uint8)
    den = _box_density(gray01, ksize)
    return gray01, den

def blue_density_map_hsv_from(hsv, lab,
                              h_lo=95, h_hi=110,
                              s_min=100, v_min=200,
                              ksize=11):
                              
    blue_hsv = cv2.inRange(hsv, (h_lo, s_min, v_min), (h_hi, 255, 255))
                                                         
    bL = lab[..., 2]
    lab_gate = (bL <= 125).astype(np.uint8) * 255
    blue_255 = cv2.bitwise_and(blue_hsv, lab_gate)
    blue01 = (blue_255 > 0).astype(np.uint8)
    den = _box_density(blue01, ksize)
    return blue01, den

def green_density_map_hsv_from(hsv, lab, ranges=None, ksize=11):
    if ranges is None:
        r1 = ((38, 100, 205), (46, 200, 255))
        r2 = ((36,  90, 185), (46, 200, 240))
        ranges = [r1, r2]
    mask_hsv = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask_hsv |= cv2.inRange(hsv, lo, hi)
    aL, bL = lab[..., 1], lab[..., 2]
    chroma = _lab_chroma(aL, bL)
    lab_gate = ((aL <= 120) & (bL >= 170) & (chroma >= 16.0)).astype(np.uint8) * 255
    water_band = cv2.inRange(hsv, (90, 40, 0), (110, 255, 255))
    not_water = cv2.bitwise_not(water_band)
    green_255 = cv2.bitwise_and(cv2.bitwise_and(mask_hsv, lab_gate), not_water)
    green_255 = cv2.morphologyEx(green_255, cv2.MORPH_OPEN, K3, iterations=1)
    green_255 = cv2.morphologyEx(green_255, cv2.MORPH_CLOSE, K3, iterations=1)
    green01 = (green_255 > 0).astype(np.uint8)
    den = _box_density(green01, ksize)
    return green01, den
                                                                            
def gray_density_map_hsv(bgr: np.ndarray, s_max: int = 80,
                         v_min: int = 80, v_max: int = 130, ksize: int = 11) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    return gray_density_map_hsv_from(hsv, lab, s_max, v_min, v_max, ksize)

def blue_density_map_hsv(bgr: np.ndarray,
                         h_lo: int = 96, h_hi: int = 101,
                         s_min: int = 105, v_min: int = 240,
                         ksize: int = 11) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    return blue_density_map_hsv_from(hsv, lab, h_lo, h_hi, s_min, v_min, ksize)

def green_density_map_hsv(bgr: np.ndarray,
                          ranges: list[tuple[tuple[int,int,int], tuple[int,int,int]]] = None,
                          ksize: int = 11) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    return green_density_map_hsv_from(hsv, lab, ranges, ksize)
                                                                    
def free_mask_gray(
    disp_bgr: np.ndarray,
    bands: list[tuple[int,int]],
    lane_types: list["LaneType"],
    *,
    gray_kernel: int = 11,
    gray_tau: float = 0.55,
    blue_kernel: int = 11,
    nonblue_tau: float = 0.55,
    green_kernel: int = 11,
    green_tau: float = 0.55,
    min_vert_frac: float = 0.75,
    consider_bottom_frac: float = 0.75,
    moving_edges: np.ndarray | None = None,
    gap_fill_px: int | None = None,
    maps: dict | None = None,                                                                 
    edge_slant_deg: float | None = None,                                              
    bidirectional_slant: bool = False, 
    erode_px: int = 0                 
) -> np.ndarray:
    H, W = disp_bgr.shape[:2]
    mask = np.ones((H, W), np.uint8)                              
    if maps is None:
                                
        _, gray_den  = gray_density_map_hsv(disp_bgr, ksize=gray_kernel)
        _, blue_den  = blue_density_map_hsv(disp_bgr, ksize=blue_kernel)
        nonblue_den  = 1.0 - blue_den
        _, green_den = green_density_map_hsv(disp_bgr, ksize=green_kernel)
    else:
        gray_den  = maps["gray_den"]
        blue_den  = maps["blue_den"]
        nonblue_den = 1.0 - blue_den
        green_den = maps["green_den"]
    for (y0, y1), lt in zip(bands, lane_types):
        if y1 <= y0:
            continue
        band_h = y1 - y0
        h_bot = max(1, int(round(consider_bottom_frac * band_h)))
        ys, ye = y1 - h_bot, y1
        if lt == LaneType.ROAD:
            den_bot = gray_den[ys:ye, :]
            free_bot = (den_bot >= gray_tau)
        elif lt in (LaneType.WATER_PLATFORM, LaneType.WATER_LILYPAD, LaneType.UNKNOWN):
            den_bot = nonblue_den[ys:ye, :]
            free_bot = (den_bot >= nonblue_tau)
        elif lt == LaneType.GREEN:
            den_bot = green_den[ys:ye, :]
            free_bot = (den_bot >= green_tau)
        else:
            continue
        need = int(min_vert_frac * h_bot)
        col_ok = (free_bot.sum(axis=0) >= need)
        if gap_fill_px is not None and gap_fill_px > 0:
                                      
            col_ok = fill_small_gaps_1d(col_ok, gap_fill_px + (10 if lt == LaneType.ROAD else 0))
        band_mask = np.ones((band_h, W), np.uint8)                  
        if col_ok.any():
            band_mask[-h_bot:, col_ok] = 0             
        mask[y0:y1, :] = band_mask
                                                             
        if erode_px > 0:
            free01 = (mask == 0).astype(np.uint8)            
                                                    
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_px*2+1, 1))
            free_eroded = cv2.erode(free01, k, iterations=1)
            mask = np.where(free_eroded > 0, 0, 1).astype(np.uint8)

    return mask

def highlight_free_and_safe_gray(
    disp_bgr: np.ndarray,
    free_mask: np.ndarray,
    *,
    alpha: float = 0.30,
    road_color=(0,140,255),
    unknown_color=(0,140,255),
    green_color=(0,140,255),
    water_color=(255,0,0),
    border_mask: np.ndarray | None = None,                                
    border_color=(255,0,255),                                    
    border_alpha: float = 0.85,
    borders: list["Border"] | None = None,                                      
    border_thickness: int = 2                                                                  
) -> np.ndarray:
    overlay = disp_bgr.copy()
    H, W = disp_bgr.shape[:2]
                                                                     
    colored = np.zeros((H, W, 3), np.uint8)
    colored[free_mask == 0] = road_color                                                            
    overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)
                                                         
    if borders is not None and len(borders) > 0:
        border_layer = np.zeros_like(overlay)
        for b in borders:
            x = int(np.clip(b.x, 0, W - 1))
            y0 = int(np.clip(b.y0, 0, H - 1))
            y1 = int(np.clip(b.y1, 0, H - 1))
            cv2.line(border_layer, (x, y0), (x, y1), border_color, border_thickness)
        overlay = cv2.addWeighted(overlay, 1.0, border_layer, border_alpha, 0)
                                                                   
    elif border_mask is not None:
        border_colored = np.zeros((H, W, 3), np.uint8)
        border_colored[border_mask > 0] = border_color
        overlay = cv2.addWeighted(overlay, 1.0, border_colored, border_alpha, 0)
    return overlay

def _stack_tiles(tiles, scale=1.0, pad=6, bg=(24,24,24)):
    if not isinstance(tiles[0], (list, tuple)):
        tiles = [tiles]
    def to_bgr_u8(im):
        im = np.asarray(im)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif im.ndim == 3 and im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        if im.dtype != np.uint8:
            im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return im
    cell_h = max(to_bgr_u8(img).shape[0] for row in tiles for img in row)
    cell_w = max(to_bgr_u8(img).shape[1] for row in tiles for img in row)
    cols   = max(len(row) for row in tiles)
    rows_bgr = []
    for row in tiles:
        row_imgs = []
        for j in range(cols):
            if j < len(row):
                im = to_bgr_u8(row[j])
            else:
                im = np.full((1, 1, 3), bg, np.uint8)
            dh, dw = cell_h - im.shape[0], cell_w - im.shape[1]
            im = cv2.copyMakeBorder(im, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=bg)
            im = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg)
            row_imgs.append(im)
        rows_bgr.append(cv2.hconcat(row_imgs))
    grid = cv2.vconcat(rows_bgr)
    if scale != 1.0:
        grid = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return grid

def _label_img_from_cc(labels: np.ndarray, best_label: int) -> np.ndarray:
    h, w = labels.shape
    rng = np.random.RandomState(42)
    max_lab = labels.max()
    colors = np.zeros((max_lab + 1, 3), np.uint8)
    for i in range(1, max_lab + 1):
        colors[i] = rng.randint(40, 220, size=3)
    lbl_bgr = colors[labels]
    if best_label > 0:
        chosen = (labels == best_label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(chosen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(lbl_bgr, contours, -1, (255,255,255), 2)
    return lbl_bgr

def detect_character(
    bgr: np.ndarray,
    *,
    bgr_target: Tuple[int,int,int] = (92,172,255),                           
    tol: int = 4,                                                                    
    min_area_frac: float = 0.0005,
    max_area_frac: float = 0.05,
    open_ksz: int = 3,
    close_ksz: int = 5,
    debug: bool = False,
) -> Tuple[Optional[Tuple[int,int]], Optional[Tuple[int,int,int,int]], float, Optional[Dict[str, Any]]]:
    H, W = bgr.shape[:2]
                                                             
    B0, G0, R0 = map(int, bgr_target)
    lo = (max(0, B0 - tol), max(0, G0 - tol), max(0, R0 - tol))
    hi = (min(255, B0 + tol), min(255, G0 + tol), min(255, R0 + tol))
    mask_raw = cv2.inRange(bgr, lo, hi)
                                                              
    n_on = int(cv2.countNonZero(mask_raw))
             
    K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    K53 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))                             
    K35 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))                              
                                                    
    path = "orig"
    if n_on <= 12:
                                                              
        path = "grow_close_strong"
        mask_boost = cv2.dilate(mask_raw, K3, iterations=2)
        mask_boost = cv2.morphologyEx(mask_boost, cv2.MORPH_CLOSE, K35, iterations=1)
                                            
        mask_boost = cv2.medianBlur(mask_boost, 3)
                                                              
        min_area_abs = 2
    elif n_on <= 80:
                                                                            
        path = "close_dilate_light"
        tmp = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, K3, iterations=1)
        mask_boost = cv2.dilate(tmp, K3, iterations=1)
                              
        min_area_abs = 8
    else:
                                                   
        path = "open_close_knobs"
        mask_open = mask_raw
        if open_ksz > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksz, open_ksz))
            mask_open = cv2.morphologyEx(mask_open, cv2.MORPH_OPEN, k, iterations=1)
        mask_boost = mask_open
        if close_ksz > 0:
            k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksz, close_ksz))
            mask_boost = cv2.morphologyEx(mask_boost, cv2.MORPH_CLOSE, k2, iterations=1)
                                     
        min_area_abs = int(max(1, min_area_frac * (H * W)))
                                              
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_boost, connectivity=8)
                                                                 
    frame_area = H * W
    max_area_abs = int(max_area_frac * frame_area)
    best_i, best_area = -1, -1
    for i in range(1, num):
        x, y, w, h, a = stats[i]
        if a < min_area_abs or a > max_area_abs:
            continue
                                                               
        if w <= 1 and h <= 1:
            continue
        if a > best_area:
            best_i, best_area = i, a
    center = None
    bbox = None
    score = 0.0
    orig_center = None
    if best_i > 0:
        x, y, w, h, a = stats[best_i]
        cx, cy = map(int, centroids[best_i])
                                                                                            
        score = float(cv2.countNonZero(mask_raw[y:y+h, x:x+w])) / float(max(1, w*h))
        cy_shifted = int(np.clip(cy + 11, 0, H - 1))
        orig_center = (cx, cy)
        center = (cx, cy_shifted)
        bbox = (x, y, w, h)
    else:
                                                                
        if n_on > 0:
            ys, xs = np.where(mask_raw > 0)
            cx = int(np.clip(round(xs.mean()), 0, W - 1))
            cy = int(np.clip(round(ys.mean()), 0, H - 1))
            pad = 6                    
            x0 = int(max(0, cx - pad)); y0 = int(max(0, cy - pad))
            x1 = int(min(W - 1, cx + pad)); y1 = int(min(H - 1, cy + pad))
            w = x1 - x0 + 1; h = y1 - y0 + 1
            score = float(cv2.countNonZero(mask_raw[y0:y1+1, x0:x1+1])) / float(max(1, w*h))
            cy_shifted = int(np.clip(cy + 11, 0, H - 1))
            orig_center = (cx, cy)
            center = (cx, cy_shifted)
            bbox = (x0, y0, w, h)
                           
    if not debug:
        return center, bbox, score, None
                                          
    labels_vis = _label_img_from_cc(labels if 'labels' in locals() else np.zeros_like(mask_raw), best_i if best_i > 0 else -1)
    overlay = bgr.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,255,0), 2)
        if center is not None:
            cv2.circle(overlay, center, 3, (0,0,255), -1)
        cv2.putText(overlay, f"score={score:.2f}", (x, max(0, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)
        if orig_center is not None:
            cv2.circle(overlay, orig_center, 2, (255,255,0), -1)
    def _with_title(img, title):
        im = img.copy()
        cv2.putText(im, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        return im
    target_chip = np.full((40, 120, 3), (B0,G0,R0), np.uint8)
    cv2.putText(target_chip, f"B:{B0} G:{G0} R:{R0} tol:{tol}", (4, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
    montage = _stack_tiles([
        [_with_title(bgr, "BGR frame"), _with_title(target_chip, "target"), _with_title(mask_raw, f"mask_raw n={n_on}")],
        [_with_title(mask_boost, f"mask_boost [{path}]"), _with_title(labels_vis, "CC labels"), np.zeros_like(bgr)],
        [_with_title(overlay, "overlay")]
    ], scale=1.0)
    debug_dict = {
        "mask_raw": mask_raw,
        "mask_boost": mask_boost,
        "labels_vis": labels_vis,
        "overlay": overlay,
        "montage": montage,
        "best_label": best_i,
        "stats": stats if 'stats' in locals() else None,
        "centroids": centroids if 'centroids' in locals() else None,
        "center_unshifted": orig_center,
        "center_shifted": center,
        "bgr_target": bgr_target,
        "tol": tol,
        "lower_bgr": lo,
        "upper_bgr": hi,
        "n_on": n_on,
        "post_path": path,
        "min_area_abs": int(min_area_abs),
        "max_area_abs": int(max_area_abs),
    }
    return center, bbox, score, debug_dict
                                                               
def deskew(frame_bgr, M=None, out_size=None):
    if M is None:
        return frame_bgr
    h_out, w_out = out_size
    return cv2.warpAffine(frame_bgr, M, (w_out, h_out), flags=cv2.INTER_LINEAR)

def find_uniform_bands_with_offset(H: int, h: int, o: int, margin_px: int = 2):
    bands = []
    y = o
    while y + h < H:
        y0 = max(0, y + margin_px)
        y1 = min(H, y + h - margin_px)
        bands.append((y0, y1))
        y += h
    return bands

def classify_line_by_masks_fast(
    y: int,
    maps: dict,
    *,
    margin_px: int = 0,
    tau_g: float = 0.55,                       
    tau_w: float = 0.55,                            
    tau_r: float = 0.55,                           
    min_cov: float = 0.18,
    debug: bool = False,
) -> LaneType | tuple[LaneType, dict]:
    H, W = maps["gray_den"].shape[:2]
    y = int(np.clip(y, 0, H - 1))
    xs = slice(margin_px, -margin_px or None)
    gden = maps["green_den"][y, xs]
    bden = maps["blue_den"][y, xs]
    rden = maps["gray_den"][y, xs]
    cov_g = float((gden >= tau_g).mean()) if gden.size else 0.0
    cov_w = float((bden >= tau_w).mean()) if bden.size else 0.0
    cov_r = float((rden >= tau_r).mean()) if rden.size else 0.0
    covs = [(cov_g, LaneType.GREEN), (cov_w, LaneType.WATER_PLATFORM), (cov_r, LaneType.ROAD)]
    covs.sort(key=lambda t: t[0], reverse=True)
    label = covs[0][1] if covs[0][0] >= min_cov else LaneType.UNKNOWN
    if not debug:
        return label
    return label, {"cov_g": cov_g, "cov_w": cov_w, "cov_r": cov_r, "min_cov": min_cov, "y": y}

def classify_band_top_bottom_lines(
    band: tuple[int,int],
    maps: dict,
    *,
    margin_px: int = 0,
    tau_g: float = 0.55,
    tau_w: float = 0.55,
    tau_r: float = 0.55,
    min_cov: float = 0.18,
    debug: bool = False,
):
    y0, y1 = int(band[0]), int(band[1])
    if y1 <= y0:
        if debug:
            return {"top": (LaneType.UNKNOWN, {"reason": "empty band"}), "bottom": (LaneType.UNKNOWN, {"reason": "empty band"})}
        return {"top": LaneType.UNKNOWN, "bottom": LaneType.UNKNOWN}
                                                                 
    y_top = int(np.clip(y0 + margin_px, y0, y1 - 1))
    y_bot = int(np.clip(y1 - 1 - margin_px, y0, y1 - 1))
    args = dict(margin_px=margin_px, tau_g=tau_g, tau_w=tau_w, tau_r=tau_r, min_cov=min_cov, debug=debug)
    top_res = classify_line_by_masks_fast(y_top, maps, **args)
    bot_res = classify_line_by_masks_fast(y_bot, maps, **args)
    return {"top": top_res, "bottom": bot_res}
                                                                          
def _color_from_id(bid: int, *, sat: float = 0.90, val: float = 0.95) -> tuple[int, int, int]:                                                       
    golden = 0.618033988749895
    h = (bid * golden) % 1.0        
    s = float(np.clip(sat, 0.0, 1.0))
    v = float(np.clip(val, 0.0, 1.0))
                                                  
    H = int(round(h * 179)) % 180
    S = int(round(s * 255))
    V = int(round(v * 255))
    hsv = np.uint8([[[H, S, V]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
    return tuple(int(c) for c in bgr)             

def draw_borders_with_velocity(
    image_bgr: np.ndarray,
    borders: list[Border],
    velocities: dict[int, dict],
    *,
    color_border=(255, 0, 255),                                    
    color_text=(255, 255, 255),
    boundary_color=(160, 160, 160),
    thickness: int = 2,
    use_id_colors: bool = True                                         
) -> np.ndarray:
    out = image_bgr.copy()
    H, W = out.shape[:2]
    for b in borders:
        x = int(np.clip(b.x, 0, W - 1))
        y0 = int(np.clip(b.y0, 0, H - 1))
        y1 = int(np.clip(b.y1, 0, H - 1))
                      
        if b.is_boundary:
            col = boundary_color
        else:
            col = _color_from_id(b.id) if use_id_colors else color_border
        cv2.line(out, (x, y0), (x, y1), col, thickness)
        if b.id in velocities:
            vx = velocities[b.id]["vx"]
            if abs(vx) >= 15:
                ty = max(0, y0 - 6)
                                                                 
                cv2.putText(out, f"{vx:+.0f}", (min(W-80, max(0, x - 40)), ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1, cv2.LINE_AA)
    return out

def band_index_for_y_lower(y: int, lane_h: int, offset: int, frame_h: int, n_bands: int) -> Optional[int]:                                              
    if y < offset:
        return None
    k = int(np.floor((y - offset) / float(lane_h)))
                                                                                   
    if 0 <= k < n_bands:
                                                               
        if offset + (k + 1) * lane_h <= frame_h:
            return k
    return None

def draw_character_annotation(
    img: np.ndarray,
    center: Optional[Tuple[int,int]],
    bbox: Optional[Tuple[int,int,int,int]],
    band_idx: Optional[int],
    lane_type: Optional[LaneType],
    score: float,
) -> np.ndarray:
    if center is None:
        return img
    out = img.copy()
                              
    cv2.circle(out, center, 4, (0, 0, 255), -1)
           
    if band_idx is not None and lane_type is not None:
        label = f"band #{band_idx:02d}"
    elif band_idx is not None:
        label = f"band #{band_idx:02d}"
    else:
        label = "band = n/a"
                                                         
    px, py = center
    ty = max(14, py - 8)
    tx = min(out.shape[1] - 60, px + 8)
    cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def _build_occ_lines_for_bands(
    free_mask: np.ndarray,
    bands: list[tuple[int,int]],
    lane_h: int,
    use_bottom_slice: bool = True,
    *,
    v_median_ksize: int = 1,                                                                        
    pad_x: int = 0                                                                    
) -> list[np.ndarray]:
    H, W = free_mask.shape[:2]
    m = (free_mask.astype(np.uint8) & 1)        
    occ_lines: list[np.ndarray] = []
    for (y0, y1) in bands:
        y0c = max(0, min(H, int(y0)))
        y1c = max(0, min(H, int(y1)))
        band_h = y1c - y0c
        if band_h <= 0:
            occ_lines.append(np.ones((W,), np.uint8))
            continue
        if use_bottom_slice:
            slice_h = max(1, min(lane_h, band_h))
            y_top, y_bot = y1c - slice_h, y1c
            y_mid = (y_top + y_bot - 1) // 2
        else:
            y_mid = (y0c + y1c - 1) // 2
                                                                                  
        if v_median_ksize > 1:
            r = v_median_ksize // 2
            y_lo = max(0, y_mid - r)
            y_hi = min(H, y_mid + r + 1)
            row = np.median(m[y_lo:y_hi, :], axis=0).astype(np.uint8)
        else:
            row = m[y_mid, :]
                                                             
        if pad_x > 0:
            k = 2 * int(pad_x) + 1
            row = cv2.dilate(row.reshape(1, -1), np.ones((1, k), np.uint8), iterations=1).reshape(-1)
        occ_lines.append(row.astype(np.uint8))
    return occ_lines

def draw_borders_with_velocity_and_sides(
    image_bgr: np.ndarray,
    borders: list[Border],
    velocities: dict[int, dict],
    side_free: dict[int, BorderSideFree],
    *,
    color_border=(255, 0, 255),
    color_text=(255, 255, 255),
    color_ok=(0, 255, 0),
    color_bad=(0, 0, 255),
    thickness: int = 2
) -> np.ndarray:
    out = image_bgr.copy()
    H, W = out.shape[:2]
    for b in borders:
        x = int(np.clip(b.x, 0, W - 1))
        y0 = int(np.clip(b.y0, 0, H - 1))
        y1 = int(np.clip(b.y1, 0, H - 1))
        mid_y = (y0 + y1) // 2
                              
        cv2.line(out, (x, y0), (x, y1), color_border, thickness)
                                    
        if b.id in velocities:
            vx = velocities[b.id]["vx"]
            if abs(vx) >= 15:
                cv2.putText(out, f"{vx:+.0f}", (min(W-80, max(0, x - 40)), max(0, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1, cv2.LINE_AA)
                       
        if b.id in side_free:
            info = side_free[b.id]
                                                     
            L = int(min(info.left_px, 40))
            R = int(min(info.right_px, 40))
            cv2.line(out, (x - L, mid_y), (x - 1, mid_y), color_ok if info.left_ok else color_bad, 2)
            cv2.line(out, (x + 1, mid_y), (x + R, mid_y), color_ok if info.right_ok else color_bad, 2)
                        
            txt = f"L{info.left_px}/{info.min_clear_px} R{info.right_px}/{info.min_clear_px}"
            cv2.putText(out, txt, (min(W-120, max(0, x - 60)), min(H-5, mid_y + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_text, 1, cv2.LINE_AA)
    return out
                                                         
def draw_vertical_move_footprint(
    disp_bgr: np.ndarray,
    *,
    occ_lines: list[np.ndarray],
    bands: list[tuple[int,int]],
    lane_h: int,
    char_band_idx: Optional[int],
    char_center: Optional[tuple[int,int]],
    char_bbox: Optional[tuple[int,int,int,int]],
    n_up: int = 2,
    alpha: float = 0.35,
    color_ok=(0, 200, 0),
    color_bad=(0, 0, 255)
) -> np.ndarray:
    if char_band_idx is None or char_center is None or char_bbox is None:
        return disp_bgr
    out = disp_bgr.copy()
    H, W = out.shape[:2]
    cx, cy = map(int, char_center)
    _, _, bw, _ = char_bbox
                                                       
    half_w = max(3, min(18, int(round(bw * 0.5))))
    x0 = max(0, cx - half_w)
    x1 = min(W - 1, cx + half_w)
    overlay = out.copy()
    for k in range(1, n_up + 1):
        bi = char_band_idx + k
        if bi < 0 or bi >= len(bands):
            continue
        y0, y1 = bands[bi]
                                                                              
        y_top = max(y0, y1 - lane_h)
        y_bot = y1
                                                                               
        occ_line = occ_lines[bi]
        window = occ_line[x0:x1 + 1]
        is_free = (window.size > 0) and (window.max() == 0)
        color = color_ok if is_free else color_bad
                                 
        cv2.rectangle(overlay, (x0, y_top), (x1, y_bot - 1), color, thickness=-1)
                                        
        cv2.rectangle(out, (x0, y_top), (x1, y_bot - 1), color, thickness=2)
                                                        
        tag = f"+{k} OK" if is_free else f"+{k} BLOCK"
        ty = max(12, y_top - 4)
        tx = max(2, x0 - 40)
        cv2.putText(out, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
                 
    out = cv2.addWeighted(out, 1.0, overlay, alpha, 0.0)
    return out

def draw_band_blocks(
    disp_bgr: np.ndarray,
    blocks: List[Dict],
    *,
    color_current=(0, 255, 255),
    color_other=(0, 200, 255),
    alpha: float = 0.45,
    outline_px: int = 2,
    show_tags: bool = False
) -> np.ndarray:
    out = disp_bgr.copy()
    overlay = out.copy()
    for b in blocks:
        x0, x1 = b["x0"], b["x1"]
        y_top, y_bot = b["y_top"], b["y_bot"]
        color = color_current if b.get("is_current", False) else color_other
        cv2.rectangle(overlay, (x0, y_top), (x1, y_bot - 1), color, thickness=-1)
        cv2.rectangle(out,     (x0, y_top), (x1, y_bot - 1), color, thickness=outline_px)
        if show_tags:
            tag = "C" if b.get("is_current", False) else f"b{b['bi']}"
            cv2.putText(out, tag, (max(2, x0 - 18), max(12, y_top - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return cv2.addWeighted(out, 1.0, overlay, alpha, 0.0)

def draw_planned_blocks(
    disp_bgr: np.ndarray,
    blocks: List[Dict],
    move_ok: List[bool],
    *,
    color_ok=(0, 255, 0),
    color_bad=(0, 0, 255),
    color_current=None,
    alpha: float = 0.40,
    outline_px: int = 2
) -> np.ndarray:
    out = disp_bgr.copy()
    overlay = out.copy()
    for blk, ok in zip(blocks, move_ok):
        x0, x1 = blk["x0"], blk["x1"]
        y0, y1 = blk["y_top"], blk["y_bot"] - 1
        col = color_ok if ok else color_bad
        cv2.rectangle(overlay, (x0, y0), (x1, y1), col, thickness=-1)
        cv2.rectangle(out,     (x0, y0), (x1, y1), col, thickness=outline_px)
    return cv2.addWeighted(out, 1.0, overlay, alpha, 0.0)

def make_free_mask_panel(
    free_mask: np.ndarray,
    *,
    scale: int = 2,                                                
    white_for_free: bool = True                                            
) -> np.ndarray:
                                             
    if white_for_free:
        vis = np.where(free_mask == 0, 255, 0).astype(np.uint8)
    else:
        vis = np.where(free_mask == 0, 0, 255).astype(np.uint8)
                                                                  
    if isinstance(scale, int) and scale > 1:
        H, W = vis.shape[:2]
        vis = cv2.resize(vis, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)                                     
                                                 
    return vis
                                                                                                                         
def borders_from_occ_lines(
    bands: list[tuple[int,int]],
    occ_lines: list[np.ndarray],
    *,
    id_start: int = 1000,
    add_edge_sentinels: bool = True
) -> list[Border]:
    borders: list[Border] = []
    bid = int(id_start)
    if not occ_lines or not bands:
        return borders
    W = int(occ_lines[0].shape[0])
    for i, (y0, y1) in enumerate(bands):
        if i >= len(occ_lines):
            break
        line = (occ_lines[i].astype(np.uint8).reshape(-1) & 1)
        if line.size == 0 or y1 <= y0:
            continue
  
        prev = int(line[0])
        for x in range(1, W):
            v = int(line[x])
            if v != prev:
                borders.append(Border(id=bid, x=int(x), y0=int(y0), y1=int(y1-1),
                                      height=int(y1-y0), is_boundary=False))
                bid += 1
                prev = v
        if add_edge_sentinels:
                       
            borders.append(Border(id=bid, x=0, y0=int(y0), y1=int(y1-1),
                                  height=int(y1-y0), is_boundary=True)); bid += 1
                 
            borders.append(Border(id=bid, x=W-1, y0=int(y0), y1=int(y1-1),
                                  height=int(y1-y0), is_boundary=True)); bid += 1
    return borders                                       
                                                                                 
def project_occ_lines(
    occ_lines: list[np.ndarray],
    bands: list[tuple[int,int]],
    borders: list["Border"],
    vel_map: dict[int, dict],
    *,
    frame_w: int,
    movement_time: float,
    vx_min_abs: float = 1.0,                               
    ignore_boundary: bool = True,                                          
    ignore_edge_px: int = 2,                                                                        
    thicken_px: int = 0                                                                 
) -> list[np.ndarray]:
    if not occ_lines:
        return []
    import statistics
    W_line = int(occ_lines[0].shape[0])
    sx = W_line / float(max(1, frame_w))                              
                                               
    by_band: dict[int, list["Border"]] = {}
    for b in borders:
        y_mid = (b.y0 + b.y1) * 0.5
        bi = -1
        for i, (y0, y1) in enumerate(bands):
            if y0 <= y_mid <= (y1 - 1):
                bi = i; break
        if bi >= 0:
            by_band.setdefault(bi, []).append(b)
    out: list[np.ndarray] = []
    for bi, line in enumerate(occ_lines):
        line_now = (np.asarray(line, np.uint8) & 1)
        if line_now.shape[0] != W_line:
                                     
            line_now = cv2.resize(line_now.reshape(1, -1), (W_line, 1),
                                  interpolation=cv2.INTER_NEAREST).reshape(-1)
                                                        
        vxs = []
        for b in by_band.get(bi, []):
            if ignore_boundary and getattr(b, "is_boundary", False):
                continue
            x_line = int(round(b.x * sx))
            if (x_line <= ignore_edge_px) or (x_line >= W_line - 1 - ignore_edge_px):
                continue
            vx = float(vel_map.get(b.id, {}).get("vx", 0.0))
            if abs(vx) >= vx_min_abs:
                vxs.append(vx)
        if not vxs:
            out.append(line_now.copy())
            continue
                                                                       
        vx_band = statistics.median(vxs)
        dx_cols = int(round(vx_band * movement_time * sx))
        if dx_cols == 0:
            future = line_now.copy()
        elif dx_cols > 0:
                                                                    
            future = np.empty_like(line_now)
            future[dx_cols:] = line_now[:-dx_cols]
            future[:dx_cols] = line_now[0]
        else:
                                              
            k = -dx_cols
            future = np.empty_like(line_now)
            future[:W_line - k] = line_now[k:]
            future[W_line - k:] = line_now[-1]
                                                      
        if thicken_px > 0:
            k = 2 * int(thicken_px) + 1
            future = cv2.dilate(future.reshape(1, -1), np.ones((1, k), np.uint8), iterations=1).reshape(-1)
        out.append(future.astype(np.uint8))
    return out

def make_occ_line_single_panel(
    occ_lines: list[np.ndarray],
    band_idx: int,
    *,
    col_scale: int = 8,
    band_h: int = 18,
    invert: bool = False
) -> np.ndarray:
    if not (0 <= band_idx < len(occ_lines)) or occ_lines[band_idx] is None:
        return np.zeros((band_h, col_scale, 3), np.uint8)
    line = (occ_lines[band_idx].astype(np.uint8) & 1).reshape(1, -1)         
    W = line.shape[1]
    vis = (line * 255).astype(np.uint8) if invert else np.where(line == 0, 255, 0).astype(np.uint8)
    vis = cv2.resize(vis, (W * col_scale, 1), interpolation=cv2.INTER_NEAREST)
    vis = np.repeat(vis, band_h, axis=0)                         
    panel = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                     
    cv2.putText(panel, f"band {band_idx}", (6, band_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
    return panel

def make_occ_lines(
    occ_lines: list[np.ndarray],
    band_idx: Optional[int] = None,                                   
    *,
    col_scale: int = 8,
    band_h: int = 18,
    invert: bool = False,
    show_labels: bool = True,              
    label_w: int = 80,                                             
    row_sep_px: int = 1,                                                       
    row_sep_color: tuple[int,int,int] = (50,50,50)
) -> np.ndarray:   
                                       
    def _render_row(idx: int) -> Optional[np.ndarray]:
        line_ = occ_lines[idx]
        if line_ is None:
            return None
        line = (line_.astype(np.uint8) & 1).reshape(1, -1)          
        W = line.shape[1]
        vis = (line * 255).astype(np.uint8) if invert else np.where(line == 0, 255, 0).astype(np.uint8)
        vis = cv2.resize(vis, (W * col_scale, 1), interpolation=cv2.INTER_NEAREST)
        vis = np.repeat(vis, band_h, axis=0)                         
        row = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        if show_labels:
            lab = np.full((band_h, label_w, 3), (24,24,24), np.uint8)
            cv2.putText(lab, f"band {idx}", (6, band_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
            row = cv2.hconcat([lab, row])
        return row
    if not occ_lines:
                                         
        w = label_w + col_scale
        return np.zeros((band_h, max(1, w), 3), np.uint8)
                                        
    if band_idx is not None and band_idx != -1:
        if not (0 <= band_idx < len(occ_lines)) or occ_lines[band_idx] is None:
            w = label_w + col_scale
            return np.zeros((band_h, max(1, w), 3), np.uint8)
        return _render_row(band_idx)
                    
    rows: list[np.ndarray] = []
    for i in range(6, min(20, len(occ_lines))):                
        r = _render_row(i)
        if r is not None:
            rows.append(r)
    if not rows:
        w = label_w + col_scale
        return np.zeros((band_h, max(1, w), 3), np.uint8)
    if row_sep_px > 0:
        width = rows[0].shape[1]
        sep = np.full((row_sep_px, width, 3), row_sep_color, np.uint8)
        out = rows[0]
        for r in rows[1:]:
            out = cv2.vconcat([out, sep, r])
        return out
    else:
        return cv2.vconcat(rows)
                                                            
def make_line_class_panel(
    top_bottom_line_labels: list[dict],
    *,
    band_h: int = 24,
    width: int = 500,
    bg_color=(24, 24, 24),
    text_color=(255, 255, 255)
) -> np.ndarray:
    n_bands = len(top_bottom_line_labels)
    H = max(1, n_bands * band_h)
    W = width
    panel = np.full((H, W, 3), bg_color, np.uint8)
    for i, tb in enumerate(top_bottom_line_labels):
        y = i * band_h + int(band_h * 0.7)
                                                   
        t_label = tb["top"][0] if isinstance(tb["top"], tuple) else tb["top"]
        b_label = tb["bottom"][0] if isinstance(tb["bottom"], tuple) else tb["bottom"]
        text = f"#{i:02d} T:{t_label.name:<6} B:{b_label.name:<6}"
        cv2.putText(panel, text, (4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
    return panel

def _line_cov_vector(
    y: int,
    maps: dict,
    *,
    margin_px: int = 0,
    tau_g: float = 0.55,
    tau_w: float = 0.55,
    tau_r: float = 0.55
) -> np.ndarray:
    H, W = maps["gray_den"].shape[:2]
    y = int(np.clip(y, 0, H - 1))
    xs = slice(margin_px, -margin_px or None)
    gden = maps["green_den"][y, xs]
    bden = maps["blue_den"][y, xs]
    rden = maps["gray_den"][y, xs]
    cov_g = float((gden >= tau_g).mean()) if gden.size else 0.0
    cov_w = float((bden >= tau_w).mean()) if bden.size else 0.0
    cov_r = float((rden >= tau_r).mean()) if rden.size else 0.0
    return np.array([cov_g, cov_w, cov_r], dtype=np.float32)

def _cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def best_offset_by_line_similarity(
    *,
    maps: dict,
    frame_h: int,
    lane_h: int = 22,
    margin_px: int = 1,
    sim_metric: str = "cosine",                        
    tau_g: float = 0.55,
    tau_w: float = 0.55,
    tau_r: float = 0.55,
    min_cov_for_label: float = 0.18,                     
    pair_weights: tuple[float,float,float] = (0.4, 0.4, 0.2),                   
) -> tuple[int, dict]:
    H = int(frame_h)
    w_tm, w_mb, w_tb = pair_weights
    w_sum = max(1e-6, (w_tm + w_mb + w_tb))
    scores: dict[int, float] = {}
    def _label_from_cov(v):
        cov_g, cov_w, cov_r = map(float, v)
        covs = [(cov_g, LaneType.GREEN), (cov_w, LaneType.WATER_PLATFORM), (cov_r, LaneType.ROAD)]
        covs.sort(key=lambda t: t[0], reverse=True)
        return covs[0][1] if covs[0][0] >= float(min_cov_for_label) else LaneType.UNKNOWN
    for o in range(0, max(1, lane_h)):
        bands = find_uniform_bands_with_offset(H, lane_h, o, margin_px=0)
        sims_band = []
        for (y0, y1) in bands:
            if y1 <= y0:
                continue
            y_top = int(np.clip(y0 + margin_px, y0, y1 - 1))
            y_bot = int(np.clip(y1 - 1 - margin_px, y0, y1 - 1))
            if y_bot < y_top:
                continue
            y_mid = int((y0 + y1 - 1) // 2)
            y_mid = int(np.clip(y_mid, y_top, y_bot))
            if sim_metric == "cosine":
                v_t = _line_cov_vector(y_top, maps, margin_px=margin_px, tau_g=tau_g, tau_w=tau_w, tau_r=tau_r)
                v_m = _line_cov_vector(y_mid, maps, margin_px=margin_px, tau_g=tau_g, tau_w=tau_w, tau_r=tau_r)
                v_b = _line_cov_vector(y_bot, maps, margin_px=margin_px, tau_g=tau_g, tau_w=tau_w, tau_r=tau_r)
                s_tm = _cosine_sim(v_t, v_m)
                s_mb = _cosine_sim(v_m, v_b)
                s_tb = _cosine_sim(v_t, v_b)
                s = (w_tm*s_tm + w_mb*s_mb + w_tb*s_tb) / w_sum
                sims_band.append(s)
            elif sim_metric == "label":
                lt = _label_from_cov(_line_cov_vector(y_top, maps, margin_px=margin_px, tau_g=tau_g, tau_w=tau_w, tau_r=tau_r))
                lm = _label_from_cov(_line_cov_vector(y_mid, maps, margin_px=margin_px, tau_g=tau_g, tau_w=tau_w, tau_r=tau_r))
                lb = _label_from_cov(_line_cov_vector(y_bot, maps, margin_px=margin_px, tau_g=tau_g, tau_w=tau_w, tau_r=tau_r))
                s_tm = 1.0 if lt == lm else 0.0
                s_mb = 1.0 if lm == lb else 0.0
                s_tb = 1.0 if lt == lb else 0.0
                s = (w_tm*s_tm + w_mb*s_mb + w_tb*s_tb) / w_sum
                sims_band.append(s)
            else:
                raise ValueError("sim_metric must be 'cosine' or 'label'")
        scores[o] = (float(np.mean(sims_band)) if sims_band else 0.0)
    best_o = max(sorted(scores.keys()), key=lambda k: scores[k]) if scores else 0
    return int(best_o), scores

def _pick_band_rows(y0: int, y1: int, *, margin_px: int, n_lines: int) -> list[int]:
    y_top = int(np.clip(y0 + margin_px, y0, y1 - 1))
    y_bot = int(np.clip(y1 - 1 - margin_px, y0, y1 - 1))
    if y_bot < y_top:
        return []
    n = max(2, int(n_lines))                         
    if (y_bot - y_top) < 1:
                                                     
        return []
    ys = np.linspace(y_top, y_bot, num=n, endpoint=True)
    ys = np.unique(np.round(ys).astype(int))
    ys = ys[(ys >= y_top) & (ys <= y_bot)]
    return ys.tolist() if ys.size >= 2 else []

def _pairwise_mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0

def best_offset_by_nline_similarity(
    *,
    maps: dict,
    frame_h: int,
    lane_h: int = 22,
    margin_px: int = 1,
    n_lines: int = 5,
    sim_metric: str = "cosine",                        
    tau_g: float = 0.55,
    tau_w: float = 0.55,
    tau_r: float = 0.55,
    min_cov_for_label: float = 0.18,                     
) -> tuple[int, dict]:
    H = int(frame_h)
    scores: dict[int, float] = {}
    def _label_from_cov(v):
        cov_g, cov_w, cov_r = map(float, v)
        covs = [(cov_g, LaneType.GREEN), (cov_w, LaneType.WATER_PLATFORM), (cov_r, LaneType.ROAD)]
        covs.sort(key=lambda t: t[0], reverse=True)
        return covs[0][1] if covs[0][0] >= float(min_cov_for_label) else LaneType.UNKNOWN
    for o in range(0, max(1, lane_h)):
        bands = find_uniform_bands_with_offset(H, lane_h, o, margin_px=0)
        per_band_scores = []
        for (y0, y1) in bands:
            if y1 <= y0:
                continue
            rows = _pick_band_rows(y0, y1, margin_px=margin_px, n_lines=n_lines)
            if len(rows) < 2:
                continue
            if sim_metric == "cosine":
                vecs = [
                    _line_cov_vector(y, maps, margin_px=margin_px, tau_g=tau_g, tau_w=tau_w, tau_r=tau_r)
                    for y in rows
                ]
                                                             
                sims = []
                for i in range(len(vecs)):
                    for j in range(i + 1, len(vecs)):
                        sims.append(_cosine_sim(vecs[i], vecs[j]))
                per_band_scores.append(_pairwise_mean(sims))
            elif sim_metric == "label":
                labels = [
                    _label_from_cov(_line_cov_vector(y, maps, margin_px=margin_px, tau_g=tau_g, tau_w=tau_w, tau_r=tau_r))
                    for y in rows
                ]
                                                  
                eq = []
                for i in range(len(labels)):
                    for j in range(i + 1, len(labels)):
                        eq.append(1.0 if labels[i] == labels[j] else 0.0)
                per_band_scores.append(_pairwise_mean(eq))
            else:
                raise ValueError("sim_metric must be 'cosine' or 'label'")
        scores[o] = float(np.mean(per_band_scores)) if per_band_scores else 0.0
    best_o = max(sorted(scores.keys()), key=lambda k: scores[k]) if scores else 0
    return int(best_o), scores

def precompute_row_cov(
    maps: dict,
    *,
    margin_px: int = 1,
    tau_g: float = 0.55,
    tau_w: float = 0.55,
    tau_r: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = maps["green_den"]; b = maps["blue_den"]; r = maps["gray_den"]
    H, W = g.shape[:2]
    x0, x1 = margin_px, max(margin_px, W - margin_px)
    if x1 <= x0:                                     
        x0, x1 = 0, W
                                                                  
    cov_g_row = (g[:, x0:x1] >= tau_g).mean(axis=1).astype(np.float32)
    cov_w_row = (b[:, x0:x1] >= tau_w).mean(axis=1).astype(np.float32)
    cov_r_row = (r[:, x0:x1] >= tau_r).mean(axis=1).astype(np.float32)
    return cov_g_row, cov_w_row, cov_r_row

def best_offset_by_nline_similarity_rows(
    *,
    cov_g_row: np.ndarray,
    cov_w_row: np.ndarray,
    cov_r_row: np.ndarray,
    frame_h: int,
    lane_h: int = 22,
    margin_px: int = 1,
    n_lines: int = 5,
    sim_metric: str = "cosine",                        
    min_cov_for_label: float = 0.18,
) -> tuple[int, dict]:
    H = int(frame_h)
    scores: dict[int, float] = {}
    def vec_at_y(y: int) -> np.ndarray:
        y = int(np.clip(y, 0, H - 1))
        return np.array([cov_g_row[y], cov_w_row[y], cov_r_row[y]], dtype=np.float32)
    
    def label_from_vec(v: np.ndarray):
        cov_g, cov_w, cov_r = map(float, v)
        best = max([(cov_g, LaneType.GREEN),
                    (cov_w, LaneType.WATER_PLATFORM),
                    (cov_r, LaneType.ROAD)], key=lambda t: t[0])
        return best[1] if best[0] >= float(min_cov_for_label) else LaneType.UNKNOWN
    
    def pick_rows(y0: int, y1: int) -> list[int]:
        y_top = int(np.clip(y0 + margin_px, y0, y1 - 1))
        y_bot = int(np.clip(y1 - 1 - margin_px, y0, y1 - 1))
        if y_bot < y_top: return []
        if y_bot - y_top < 1: return []
        ys = np.linspace(y_top, y_bot, num=max(2, int(n_lines)), endpoint=True)
        ys = np.unique(np.round(ys).astype(int))
        ys = ys[(ys >= y_top) & (ys <= y_bot)]
        return ys.tolist() if ys.size >= 2 else []
    
    def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
        na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
        if na < eps or nb < eps: return 0.0
        return float(np.dot(a, b) / (na * nb))
    for o in range(0, max(1, lane_h)):                       
                                                          
        bands = find_uniform_bands_with_offset(H, lane_h, o, margin_px=0)
        per_band_scores = []
        for (y0, y1) in bands:
            if y1 <= y0: continue
            rows = pick_rows(y0, y1)
            if len(rows) < 2: continue
            if sim_metric == "cosine":
                vecs = [vec_at_y(y) for y in rows]
                sims = []
                for i in range(len(vecs)):
                    for j in range(i + 1, len(vecs)):
                        sims.append(cosine(vecs[i], vecs[j]))
                per_band_scores.append(float(np.mean(sims)) if sims else 0.0)
            elif sim_metric == "label":
                labels = [label_from_vec(vec_at_y(y)) for y in rows]
                eq = []
                for i in range(len(labels)):
                    for j in range(i + 1, len(labels)):
                        eq.append(1.0 if labels[i] == labels[j] else 0.0)
                per_band_scores.append(float(np.mean(eq)) if eq else 0.0)
            else:
                raise ValueError("sim_metric must be 'cosine' or 'label'")
        scores[o] = float(np.mean(per_band_scores)) if per_band_scores else 0.0
    best_o = max(sorted(scores.keys()), key=lambda k: scores[k]) if scores else 0
    return int(best_o), scores

def _label_from_vec(v: np.ndarray, *, min_cov_for_label: float = 0.18) -> LaneType:
    cov_g, cov_w, cov_r = map(float, v)
    best = max([(cov_g, LaneType.GREEN),
                (cov_w, LaneType.WATER_PLATFORM),
                (cov_r, LaneType.ROAD)], key=lambda t: t[0])
    return best[1] if best[0] >= float(min_cov_for_label) else LaneType.UNKNOWN

def _pick_band_rows(y0: int, y1: int, *, margin_px: int, n_lines: int) -> list[int]:
    y_top = int(np.clip(y0 + margin_px, y0, y1 - 1))
    y_bot = int(np.clip(y1 - 1 - margin_px, y0, y1 - 1))
    if y_bot < y_top or (y_bot - y_top) < 1:
        return []
    ys = np.linspace(y_top, y_bot, num=max(2, int(n_lines)), endpoint=True)
    ys = np.unique(np.round(ys).astype(int))
    ys = ys[(ys >= y_top) & (ys <= y_bot)]
    return ys.tolist()

def classify_bands_from_rowcov(
    bands: list[tuple[int,int]],
    *,
    cov_g_row: np.ndarray,
    cov_w_row: np.ndarray,
    cov_r_row: np.ndarray,
    margin_px: int = 1,
    n_lines: int = 5,
    mode: str = "mean",                                                                 
    min_cov_for_label: float = 0.18,
    return_debug: bool = False,
) -> list[LaneType] | tuple[list[LaneType], list[dict]]:
    H = cov_g_row.shape[0]
    labels: list[LaneType] = []
    debugs: list[dict] = []
    for (y0, y1) in bands:
        if y1 <= y0:
            labels.append(LaneType.UNKNOWN)
            if return_debug: debugs.append({"rows":[],"vec_mean":None,"label":"UNKNOWN"})
            continue
        rows = _pick_band_rows(y0, y1, margin_px=margin_px, n_lines=n_lines)
        if len(rows) < 2:
            labels.append(LaneType.UNKNOWN)
            if return_debug: debugs.append({"rows":rows,"vec_mean":None,"label":"UNKNOWN"})
            continue
        if mode == "mean":
            vecs = np.stack([np.array([cov_g_row[y], cov_w_row[y], cov_r_row[y]], np.float32) for y in rows], axis=0)
            vec_mean = vecs.mean(axis=0)
            lab = _label_from_vec(vec_mean, min_cov_for_label=min_cov_for_label)
            labels.append(lab)
            if return_debug:
                debugs.append({
                    "rows": rows,
                    "vec_mean": vec_mean.tolist(),
                    "label": lab.name,
                })
        elif mode == "vote":
            row_labels = [
                _label_from_vec(np.array([cov_g_row[y], cov_w_row[y], cov_r_row[y]], np.float32),
                                min_cov_for_label=min_cov_for_label)
                for y in rows
            ]
                                                              
            counts = {}
            for L in row_labels:
                if L == LaneType.UNKNOWN: continue
                counts[L] = counts.get(L, 0) + 1
            lab = max(counts.items(), key=lambda kv: kv[1])[0] if counts else LaneType.UNKNOWN
            labels.append(lab)
            if return_debug:
                debugs.append({
                    "rows": rows,
                    "row_labels": [l.name for l in row_labels],
                    "label": lab.name,
                })
        else:
            raise ValueError("mode must be 'mean' or 'vote'")
    return (labels, debugs) if return_debug else labels

def shift_band_state_by_one(*, occ_stab, lane_est, lane_kf: dict, lane_kf_stale: dict, use_kalman: bool):                                               
    if occ_stab is not None and getattr(occ_stab, "last", None) is not None:
                                                                                         
        occ_stab.buf = np.roll(occ_stab.buf, shift=1, axis=1)
        occ_stab.buf[:, 0, :] = 0
                                   
        occ_stab.sum = np.roll(occ_stab.sum, shift=1, axis=0)
        occ_stab.sum[0, :] = 0
                             
        occ_stab.last = np.roll(occ_stab.last, shift=1, axis=0)
        occ_stab.last[0, :] = 0
                                                                 
    if lane_est is not None and hasattr(lane_est, "_lane_samples"):
        old = lane_est._lane_samples
        new = defaultdict(lambda: deque(maxlen=lane_est.history))
        for idx, dq in old.items():
            new[idx + 1] = dq                                      
        lane_est._lane_samples = new
                                                                                       
                                                     
    if use_kalman and lane_kf is not None and lane_kf_stale is not None:
        kf_new = {}
        stale_new = defaultdict(int)
        for idx, kf in lane_kf.items():
            kf_new[idx + 1] = kf
        for idx, age in lane_kf_stale.items():
            stale_new[idx + 1] = age
        lane_kf.clear(); lane_kf.update(kf_new)
        lane_kf_stale.clear(); lane_kf_stale.update(stale_new)

def xor_occ_lines(
    curr: list[np.ndarray],
    prev: list[np.ndarray] | None
) -> list[np.ndarray]:
    if not curr:
        return []
    out: list[np.ndarray] = []
    if prev is None or len(prev) != len(curr):
                                        
        for line in curr:
            w = int(line.shape[0]) if line is not None else 0
            out.append(np.zeros((w,), np.uint8))
        return out
    for lc, lp in zip(curr, prev):
        if lc is None:
            out.append(np.zeros((0,), np.uint8))
            continue
        lc = (np.asarray(lc, np.uint8).reshape(-1) & 1)
        if lp is None or lp.size == 0:
            out.append(np.zeros_like(lc))
            continue
                                                        
        lp = (np.asarray(lp, np.uint8).reshape(-1) & 1)
        if lp.shape[0] != lc.shape[0]:
            lp = cv2.resize(lp.reshape(1, -1), (lc.shape[0], 1),
                            interpolation=cv2.INTER_NEAREST).reshape(-1)
        out.append((lc ^ lp).astype(np.uint8))
    return out

def _best_shift_by_xor(prev_line: np.ndarray,
                       curr_line: np.ndarray,
                       max_shift: int = 50) -> tuple[int, int, float]:
    p = (np.asarray(prev_line, np.uint8).reshape(-1) & 1)
    c = (np.asarray(curr_line, np.uint8).reshape(-1) & 1)
    Wp, Wc = p.size, c.size
    if Wp == 0 or Wc == 0:
        return 0, 0, 1.0
                                                                                
    if Wp != Wc:
        p = cv2.resize(p.reshape(1, -1), (Wc, 1), interpolation=cv2.INTER_NEAREST).reshape(-1)
    W = c.size
    best_s = 0
    best_cost = 1.0
    best_overlap = 0
    for s in range(-max_shift, max_shift + 1):
        if s == 0:
            p_seg, c_seg = p, c
        elif s > 0:
                                                          
            if s >= W: 
                continue
            p_seg, c_seg = p[s:], c[:-s]
        else:         
            k = -s
            if k >= W:
                continue
            p_seg, c_seg = p[:-k], c[k:]
        n = c_seg.size
        if n <= 0:
            continue
        cost = float(np.count_nonzero(np.bitwise_xor(p_seg, c_seg))) / float(n)
        if cost < best_cost:
            best_cost = cost
            best_s = s
            best_overlap = n
    return best_s, best_overlap, best_cost

def estimate_lane_vx_from_xor(prev_lines: list[np.ndarray] | None,
                              curr_lines: list[np.ndarray],
                              dt: float,
                              *,
                              max_shift_cols: int = 60,
                              min_overlap_cols: int = 20,
                              vx_min_abs: float = 1.0,
                              vx_max_abs: float = 800.0) -> dict[int, float]:
    out: dict[int, float] = {}
    if dt <= 1e-6 or not curr_lines:
        return out
    if prev_lines is None or len(prev_lines) != len(curr_lines):
        return out
    for i, (prev_line, curr_line) in enumerate(zip(prev_lines, curr_lines)):
        if prev_line is None or curr_line is None:
            continue
        s, overlap, cost = _best_shift_by_xor(prev_line, curr_line, max_shift=max_shift_cols)
        if overlap < min_overlap_cols:
            continue
                                                        
        vx = float(s) / float(dt)
        if abs(vx) < vx_min_abs or abs(vx) > vx_max_abs:
            continue
        out[i] = vx
    return out
                                                         
def _theil_sen_slope(tt: np.ndarray, yy: np.ndarray) -> tuple[float, float, int]:
    n = tt.size
    if n < 3:
        return 0.0, float('nan'), 0
    i, j = np.triu_indices(n, k=1)
    denom = tt[j] - tt[i]
    m = np.abs(denom) > 1e-9
    if not np.any(m):
        return 0.0, float('nan'), 0
    slopes = (yy[j] - yy[i]) / denom
    slopes = slopes[m]
    if slopes.size == 0:
        return 0.0, float('nan'), 0
    v = float(np.median(slopes))
    mad = float(np.median(np.abs(slopes - v)))
    return v, mad, int(slopes.size)

def velocities_from_disp_history(
    disp_log: dict[int, dict],
    *,
    window_s: float = 1.3,
    min_points: int = 5,
    min_time_span: float = 0.20,                                       
    max_pairs: int = 4000
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for bid, rec in disp_log.items():
        t = np.asarray(rec.get("t_rel", []), dtype=float)
        y = np.asarray(rec.get("disp", []), dtype=float)
        if t.size != y.size or t.size < min_points:
            continue
        tmax = float(t[-1])
        keep = t >= (tmax - window_s)
        tt, yy = t[keep], y[keep]
        if tt.size < min_points or (tt[-1] - tt[0]) < min_time_span:
            continue
                                                       
        n = tt.size
        if n * (n - 1) // 2 > max_pairs:
            stride = int(np.ceil(n / np.sqrt(2 * max_pairs)))
            tt = tt[::stride]; yy = yy[::stride]
        v, mad, _ = _theil_sen_slope(tt, yy)
        out[bid] = {"vx": v, "mad": mad, "n": int(tt.size)}
    return out
                                                       
def _parabolic_subpixel(y: np.ndarray, k: int) -> float:
    if k <= 0 or k >= len(y)-1:
        return float(k)
    a, b, c = float(y[k-1]), float(y[k]), float(y[k+1])
    denom = (a - 2.0*b + c)
    if abs(denom) < 1e-12:
        return float(k)
    return k + 0.5*(a - c)/denom  
                    
def _tukey_window(n: int, alpha: float = 0.25) -> np.ndarray:
    if n <= 1: 
        return np.ones(n, np.float32)
    w = np.zeros(n, np.float32)
    m = n - 1
    for i in range(n):
        x = i / m
        if x < alpha/2:
            w[i] = 0.5*(1 + np.cos(np.pi*(2*x/alpha - 1)))
        elif x <= 1 - alpha/2:
            w[i] = 1.0
        else:
            w[i] = 0.5*(1 + np.cos(np.pi*(2*x/alpha - 2/alpha + 1)))
    return w

def _phase_corr_shift_1d(f: np.ndarray, g: np.ndarray, *, sigma: float = 0.8, use_window: bool = True
                         ) -> tuple[float, float]:
    f = np.asarray(f, np.float32).reshape(-1)
    g = np.asarray(g, np.float32).reshape(-1)
    n = min(f.size, g.size)
    if n == 0:
        return 0.0, 0.0
    f = f[:n]; g = g[:n]
                                               
    if sigma and sigma > 0:
                                                                            
        f_blur = cv2.GaussianBlur(f.reshape(1, -1), (0, 1), sigmaX=sigma, sigmaY=0).reshape(-1)
        g_blur = cv2.GaussianBlur(g.reshape(1, -1), (0, 1), sigmaX=sigma, sigmaY=0).reshape(-1)
    else:
        f_blur, g_blur = f, g
                                                      
    fz = f_blur - float(np.mean(f_blur))
    gz = g_blur - float(np.mean(g_blur))
    if use_window:
        w = _tukey_window(n, alpha=0.25)
        fz *= w; gz *= w
                          
    F = np.fft.rfft(fz)
    G = np.fft.rfft(gz)
    R = F * np.conj(G)
    mag = np.abs(R)
    R /= (mag + 1e-12)
                          
    r = np.fft.irfft(R, n=n).real
    k = int(np.argmax(r))
    k_sub = _parabolic_subpixel(r, k)
                                         
    shift = k_sub
    if shift > n/2:
        shift -= n
                                                           
    peak = float(r[int(round(k))])
    conf = peak / (np.sum(np.abs(r)) + 1e-12)
    return float(shift), float(conf)

def estimate_lane_vx_from_phasecorr(prev_lines: list[np.ndarray] | None,
                                    curr_lines: list[np.ndarray],
                                    dt: float,
                                    *,
                                    sigma: float = 0.8,
                                    use_window: bool = True,
                                    vx_min_abs: float = 1.0,
                                    vx_max_abs: float = 800.0,
                                    conf_min: float = 0.005
                                    ) -> dict[int, float]:
    out: dict[int, float] = {}
    if dt <= 1e-6 or prev_lines is None or not curr_lines or len(prev_lines) != len(curr_lines):
        return out
    for i, (p, c) in enumerate(zip(prev_lines, curr_lines)):
        if p is None or c is None:
            continue
                                            
        p1 = (np.asarray(p).reshape(-1) & 1).astype(np.float32)
        c1 = (np.asarray(c).reshape(-1) & 1).astype(np.float32)
        shift_cols, conf = _phase_corr_shift_1d(p1, c1, sigma=sigma, use_window=use_window)
        if conf < conf_min:
            continue
        vx = float(shift_cols) / float(dt)
        if abs(vx) < vx_min_abs or abs(vx) > vx_max_abs:
            continue
        out[i] = vx
    return out                                                                   

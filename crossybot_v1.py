from __future__ import annotations
from dataclasses import dataclass
import cv2
from typing import Optional
from collections import defaultdict, deque
from typing import Optional
import time

import pyautogui
import pygetwindow as gw
import utils

TARGET_W, TARGET_H = 240, 480
CHAR_TUNER_WIN = "CharTuner (BGR)"
CHAR_DEBUG_WIN = "CHAR DEBUG"
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
K_VERT_MINRUN = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))                             
K_H_THICK     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))    
frame_w = TARGET_W
occ_free_thres = 1.0              
                                         
                                             
cv2.useOptimized()

def main():                                                                                                                   
    SHOW_CHAR_DEBUG = False                   
    velocity_log: list[dict] = []
    angle_deg = 14.5                                                           
    cfg = utils.CaptureConfig(monitor=1, region=(0, 27, 247, 449))
    cap = utils.ScreenCapture(cfg)
    target_fps = 120
    target_dt = 1.0 / target_fps                                              
    offset = 0
    last_moving = None
    lane_h = 22
    margin_px = 1
    frame_i = 0
                                             
    M = None
    raw_w = raw_h = None
    param = 0
    lane_est = utils.LaneVelocityEstimator(
        lane_h=lane_h,
        history=60,
        min_samples=5,
        trim_frac=0.4,
        ignore_abs_below=15.0,
        ignore_abs_above=500.0,
        offset=0,
    )
    USE_KALMAN = False
    gameover = utils.GameOverDetector(
        hsv_lo=(17, 160, 200),                
        hsv_hi=(30, 255, 255),                
        min_frac=0.015,
        consec_needed=3                                                         
    )
    state_machine = utils.BotStateMachine(wait_time=0.2)
    DISP_WIN_FRAMES = 30                                                      
    DISP_SUM_THRESH_PX = 10                                                     
                                                             
    if not hasattr(main, "_border_last_x"):
        main._border_last_x = {}                       
    if not hasattr(main, "_border_dx_hist"):
        main._border_dx_hist = defaultdict(lambda: deque(maxlen=DISP_WIN_FRAMES))
    planner = None
    if not hasattr(main, "_last_offset0"):
        main._last_offset0 = None
    last_ts = time.perf_counter()
    last_offset_sign = -1
    if not hasattr(main, "_prev_occ_lines"):
        main._prev_occ_lines = None                                    
    if not hasattr(main, "_xt_vel"):
        main._xt_vel = utils.XTBandVelocity(
            max_seconds=2.0,
            resample_hz=60.0,                         
            x_blur_px=1.0,
            close_time_gaps_k=0                                                                          
        )

    while True:
        frame_start = time.perf_counter()
        dt = max(1e-6, frame_start - last_ts)
                           
        frame_bgr = cap.next_frame()
        last_ts = frame_start     
                                      
        if M is None:
            raw_h, raw_w = frame_bgr.shape[:2]
            center = (raw_w // 2, raw_h // 2)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
                                                
        frame_bgr = utils.deskew(frame_bgr, M, out_size=(raw_h, raw_w))
        proc_bgr = cv2.resize(frame_bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
        triggered, go_frac = gameover.update(proc_bgr)
        if triggered:
            win = gw.getWindowsWithTitle("MSI App Player")[0]
            pyautogui.click(150, 430)
            print("GAME OVER")
                                                    
            lane_est = utils.LaneVelocityEstimator(
                lane_h=lane_h,
                history=60,
                min_samples=5,
                trim_frac=0.4,
                ignore_abs_below=15.0,
                ignore_abs_above=500.0,
                offset=0,
            )
                                                             
            main._border_last_x.clear()
            main._border_dx_hist.clear()
            main._border_last_x = {}
            main._border_dx_hist = defaultdict(lambda: deque(maxlen=DISP_WIN_FRAMES))
            state_machine.reset()
            time.sleep(3)
            continue
        center, bbox, score, dbg = utils.detect_character(
            proc_bgr,
            bgr_target=(92,172,255),
            tol=4,
            debug=True
        )
        if SHOW_CHAR_DEBUG and dbg is not None and "montage" in dbg:
            cv2.imshow(CHAR_DEBUG_WIN, dbg["montage"])
            cv2.moveWindow(CHAR_DEBUG_WIN, 760, 0)
                                                                                    
        hsv = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2Lab)
        _, gray_den = utils.gray_density_map_hsv_from(hsv, lab, ksize=11)
        _, blue_den = utils.blue_density_map_hsv_from(hsv, lab, ksize=11)
        _, green_den = utils.green_density_map_hsv_from(hsv, lab, ksize=11)
        maps = {"hsv": hsv, "lab": lab, "gray_den": gray_den, "blue_den": blue_den, "green_den": green_den}
        cov_g_row, cov_w_row, cov_r_row = utils.precompute_row_cov(
            maps, margin_px=1, tau_g=0.55, tau_w=0.55, tau_r=0.55
        )
        offset = 7
        offset_sign = round((offset-11.5)/11.5)
        if offset_sign == -1 and last_offset_sign == 1:                                                                                       
            occ_stab = getattr(main, "_occ_stab", None)                               
            utils.shift_band_state_by_one(
                occ_stab=occ_stab,
                lane_est=lane_est,
                lane_kf=None,
                lane_kf_stale=None,
                use_kalman=USE_KALMAN,
            )
        last_offset_sign = offset_sign                    
        lane_est.set_offset(offset)                                                          
        H, W = proc_bgr.shape[:2]
        bands = utils.find_uniform_bands_with_offset(H, lane_h, offset, margin_px=margin_px)
        lane_types = []
        band_debugs = []
        lane_types = utils.classify_bands_from_rowcov(
            bands,
            cov_g_row=cov_g_row,
            cov_w_row=cov_w_row,
            cov_r_row=cov_r_row,
            margin_px=1,
            n_lines=5,
            mode="mean",                                                     
            min_cov_for_label=0.18,
        )                                                        
        char_band_idx: Optional[int] = None
        if center is not None and bbox is not None:                                 
            y_exact = center[1]
            char_band_idx = utils.band_index_for_y_lower(
                y=y_exact,
                lane_h=lane_h,
                offset=offset,
                frame_h=H,
                n_bands=len(bands)
            )
        frame_i += 1
        disp = proc_bgr.copy()                        
        top_bottom_line_labels = []                                                                    
        for i, b in enumerate(bands):
            tb = utils.classify_band_top_bottom_lines(
                b, maps,
                margin_px=margin_px,
                tau_g=0.55, tau_w=0.55, tau_r=0.55,
                min_cov=0.18,
                debug=True                                         
            )
            top_bottom_line_labels.append(tb)
            
        utils.draw_lane_bands(disp, [
            utils.Lane(idx=i, y0=y0, y1=y1, height=(y1 - y0), lane_type=lt,
                 mean_rgb=(0, 0, 0), confidence=1.0)
            for i, ((y0, y1), lt) in enumerate(zip(bands, lane_types))
        ], show_labels=True)
                                           
        free_mask = utils.free_mask_gray(
            proc_bgr, bands, lane_types,
            gray_kernel=11,
            gray_tau=0.55,
            min_vert_frac=0.75,
            consider_bottom_frac=0.75,
            moving_edges=last_moving,
            gap_fill_px=19,
            maps=maps,
            erode_px=0
        )
                                                                       
        occ_lines = utils._build_occ_lines_for_bands(free_mask, bands, lane_h, use_bottom_slice=True)
        ts_now = time.perf_counter()
        main._xt_vel.push(occ_lines, ts_now)                                                                    
        borders_now = utils.borders_from_occ_lines(bands, occ_lines, id_start=2000, add_edge_sentinels=False)                                  
        for b in borders_now:
            lastx = main._border_last_x.get(b.id)
            if lastx is not None:
                dx = abs(int(b.x) - int(lastx))
                main._border_dx_hist[b.id].append(dx)
            main._border_last_x[b.id] = int(b.x)
                                             
        present_ids = {b.id for b in borders_now}
        for bid in list(main._border_last_x.keys()):
            if bid not in present_ids:
                main._border_last_x.pop(bid, None)
                main._border_dx_hist.pop(bid, None)
        
        disp_log = lane_est.get_displacement_log()
        borders_all = list(borders_now)                                                         
                                              
        now_mid = time.perf_counter()
        dt = max(1e-6, now_mid - last_ts)
        inst_fps = 1.0 / dt                                                                       
        vel_map_lane = {}
                                              
        prev_vel_map_lane = dict(vel_map_lane)
        for b in borders_now:
            vx = vel_map_lane.get(b.id, {"vx": 0.0})["vx"]
            velocity_log.append({
                "frame": frame_i,
                "border_id": b.id,
                "x": int(b.x),
                "y0": int(b.y0),
                "y1": int(b.y1),
                "height": int(b.height),
                "vx": float(vx)
            })
                                                                 
        for b in borders_all:
                                                          
            if getattr(b, "is_boundary", False):
                vel_map_lane[b.id] = {"vx": 0.0}
                continue
            total_disp = sum(main._border_dx_hist.get(b.id, []))
            if total_disp < DISP_SUM_THRESH_PX:
                if b.id in vel_map_lane:
                    vel_map_lane[b.id]["vx"] = 0.0
                else:
                    vel_map_lane[b.id] = {"vx": 0.0}
        if not planner: 
            planner = utils.SideFreePlanner(
            movement_time=state_machine.wait_time, 
            occ_free_thresh=0.95, 
            lane_h=lane_h, 
            frame_w=disp.shape[1],
            frame_shape=disp.shape[:2],
        )  
                                                      
        planner.update_occ_lines(occ_lines)                    
                               
        move_cmd = planner.plan_blocks(
            bands=bands,
            borders=borders_all,
            vel_map=vel_map_lane,
            lane_types=lane_types,
            char_band_idx=char_band_idx,
            char_center=center,
        )
                                                    
        if planner.block:
            disp = utils.draw_planned_blocks(
                disp,
                [planner.block],
                [True],
                color_ok=(0, 255, 0),
                color_bad=(0, 0, 255),
                color_current=None,
                alpha=0.40,
                outline_px=2
            )
        disp = utils.draw_borders_with_velocity(
            disp,
            borders_now,
            vel_map_lane,
            color_border=(255, 0, 255),
            color_text=(255, 255, 255),
            thickness=2
        )
        if not triggered:
            state_machine.update(move_cmd=move_cmd)
        hud_info = {
            "fps": f"{inst_fps:.1f}",
            "param": param,
        }
        utils.draw_hud(disp, hud_info)
        occ_lines_future = utils.project_occ_lines(
            occ_lines=occ_lines,                                    
            bands=bands,
            borders=borders_all,                                       
            vel_map=vel_map_lane,                                    
            frame_w=disp.shape[1],
            movement_time=0.3,
            thicken_px=1                                                                    
        )
        occ_panel_now = utils.make_occ_lines(
            occ_lines,
            col_scale=1, band_h=18, show_labels=True, label_w=100
        )
        occ_panel_future = utils.make_occ_lines(
            getattr(main, "_prev_occ_lines", None),
            col_scale=1, band_h=18, show_labels=True, label_w=100
        )
                              
        cv2.imshow("occ_lines NOW (0=free,1=occ)", occ_panel_now)
        cv2.moveWindow("occ_lines NOW (0=free,1=occ)", 760, 0)
        cv2.imshow("occ_lines +dt (0=free,1=occ)", occ_panel_future)
        cv2.moveWindow("occ_lines +dt (0=free,1=occ)", 760, occ_panel_now.shape[0] + 20)
                              
        main._prev_occ_lines = [l.copy() if l is not None else None for l in occ_lines]
        cv2.imshow("CrossyBot", disp)
        cv2.moveWindow("CrossyBot", 650, 0)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            cv2.destroyAllWindows()                                               
            return
        if key == ord(']'):
            param += 1
        if key == ord('['):
            param -= 1
        work_time = time.perf_counter() - frame_start
        sleep_s = max(0.0, target_dt - work_time)
        if sleep_s > 0:
            time.sleep(sleep_s)
            work_time = target_dt
        inst_fps = 1.0 / work_time
       
if __name__ == "__main__":
    main()
 
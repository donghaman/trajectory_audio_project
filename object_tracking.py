import csv
import json
import os
import cv2
import numpy as np


TRACK_COLUMNS = [
    'Frame', 'Bounding box', 'Bounding box center', 'Bounding box bottom center',
    'Mask area', 'Confidence', 'Visible'
]
SCREEN_COLUMNS = ['frame', 'screen_x', 'screen_y', 'visible']


def create_initial_mask(frame, x, y):
    h, w = frame.shape[:2]
    radius = max(12, int(min(w, h) * 0.025))
    x0 = max(0, int(x) - radius)
    y0 = max(0, int(y) - radius)
    x1 = min(w, int(x) + radius)
    y1 = min(h, int(y) + radius)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    mask[y0:y1, x0:x1] = cv2.GC_PR_BGD
    inner = radius // 2
    cx, cy = int(x), int(y)
    cv2.ellipse(mask, (cx, cy), (inner, inner), 0, 0, 360, cv2.GC_FGD, -1)

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(frame, mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        result = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(result, (cx, cy), (radius, radius), 0, 0, 360, 255, -1)
    return result


def mask_bbox(mask):
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    if w <= 0 or h <= 0:
        return None
    return int(x), int(y), int(w), int(h)


def bbox_mask(frame, bbox):
    x, y, w, h = [int(v) for v in bbox]
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    return mask


def refine_mask(frame, bbox):
    x, y, w, h = [int(v) for v in bbox]
    h_img, w_img = frame.shape[:2]
    if w <= 0 or h <= 0:
        return np.zeros((h_img, w_img), dtype=np.uint8)

    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    if w <= 1 or h <= 1:
        return np.zeros((h_img, w_img), dtype=np.uint8)

    rect = (x, y, w, h)
    mask = np.full((h_img, w_img), cv2.GC_BGD, dtype=np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(frame, mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
        return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        return bbox_mask(frame, rect)


def center_and_bottom(bbox):
    x, y, w, h = bbox
    return (float(x + w / 2.0), float(y + h / 2.0), float(x + w / 2.0), float(y + h))


def track_video(video_path, start_frame, initial_bbox, initial_mask,
                progress_callback=None, stop_event=None):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError('추적할 영상을 열 수 없습니다.')

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    results = {}

    templates = {}

    def read_frame(index):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        return frame if ok else None

    def track_from(template, template_bbox, source_index, direction, frame_count):
        if template is None:
            return
        th, tw = template.shape[:2]
        index = source_index + direction
        previous_bbox = tuple(template_bbox)

        while 0 <= index < frame_count:
            if stop_event is not None and stop_event.is_set():
                break
            frame = read_frame(index)
            if frame is None:
                results[index] = {'bbox': None, 'mask': None, 'confidence': 0.0, 'visible': False}
                index += direction
                continue

            px, py, pw, ph = previous_bbox
            margin = max(24, int(max(pw, ph) * 0.35))
            sx0 = max(0, px - margin)
            sy0 = max(0, py - margin)
            sx1 = min(width, px + pw + margin)
            sy1 = min(height, py + ph + margin)
            search = frame[sy0:sy1, sx0:sx1]

            score = 0.0
            found = None
            if search.shape[0] >= th and search.shape[1] >= tw:
                tws = min(tw, search.shape[1])
                ths = min(th, search.shape[0])
                current_template = template
                if current_template.shape[1] != tws or current_template.shape[0] != ths:
                    current_template = cv2.resize(current_template, (tws, ths))
                gray_search = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
                gray_template = cv2.cvtColor(current_template, cv2.COLOR_BGR2GRAY)
                result = cv2.matchTemplate(gray_search, gray_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                score = float(max_val)
                if score >= 0.35:
                    found = (sx0 + max_loc[0], sy0 + max_loc[1], tws, ths)

            if found is None:
                found = previous_bbox
                score = min(score, 0.20)

            visible = score >= 0.35
            if visible:
                mask = refine_mask(frame, found)
                refined_bbox = mask_bbox(mask)
                if refined_bbox is not None:
                    found = refined_bbox
                else:
                    mask = bbox_mask(frame, found)
            else:
                mask = np.zeros((height, width), dtype=np.uint8)

            results[index] = {
                'bbox': found if visible else None,
                'mask': mask,
                'confidence': score if visible else 0.0,
                'visible': visible,
            }
            if visible:
                previous_bbox = found
                if index % 20 == 0:
                    templates[index] = frame[max(0, found[1]):min(height, found[1] + found[3]),
                                              max(0, found[0]):min(width, found[0] + found[2])].copy()

            processed = abs(index - start_frame) + 1
            if progress_callback:
                progress_callback(min(99, int(processed * 100 / max(frame_count, 1))))
            index += direction

    start_frame = max(0, min(int(start_frame), frame_count - 1))
    start_frame_image = read_frame(start_frame)
    if start_frame_image is None:
        capture.release()
        raise ValueError('선택 프레임을 읽을 수 없습니다.')

    refined_bbox = mask_bbox(initial_mask) if initial_mask is not None else initial_bbox
    if refined_bbox is None:
        refined_bbox = tuple(int(v) for v in initial_bbox)
    refined_mask = refine_mask(start_frame_image, refined_bbox)
    refined_bbox = mask_bbox(refined_mask) or refined_bbox
    results[start_frame] = {
        'bbox': refined_bbox,
        'mask': refined_mask,
        'confidence': 1.0,
        'visible': True,
    }

    x, y, w, h = refined_bbox
    template = start_frame_image[max(0, y):min(height, y+h), max(0, x):min(width, x+w)].copy()

    track_from(template, refined_bbox, start_frame, 1, frame_count)
    if start_frame > 0:
        track_from(template, refined_bbox, start_frame, -1, frame_count)

    ordered = {}
    for index in range(frame_count):
        item = results.get(index)
        if item is None:
            ordered[index] = {'bbox': None, 'mask': None, 'confidence': 0.0, 'visible': False}
        else:
            ordered[index] = item

    capture.release()
    if progress_callback:
        progress_callback(100)
    return ordered


def save_tracking_csv(path, track_name, results):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Target', track_name])
        writer.writerow(TRACK_COLUMNS)
        for frame in sorted(results):
            item = results[frame]
            bbox = item.get('bbox')
            mask = item.get('mask')
            visible = bool(item.get('visible'))
            if bbox is None:
                bbox_text = ''
                center_text = ''
                bottom_text = ''
            else:
                x, y, w, h = bbox
                cx, cy, bx, by = center_and_bottom(bbox)
                bbox_text = f'{x},{y},{w},{h}'
                center_text = f'{cx:.1f},{cy:.1f}'
                bottom_text = f'{bx:.1f},{by:.1f}'
            area = int(cv2.countNonZero(mask)) if mask is not None else 0
            confidence = float(item.get('confidence', 0.0))
            writer.writerow([frame, bbox_text, center_text, bottom_text, area,
                             f'{confidence:.4f}', 1 if visible else 0])


def save_screen_trajectory_csv(path, results):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SCREEN_COLUMNS)
        writer.writeheader()
        for frame in sorted(results):
            item = results[frame]
            bbox = item.get('bbox')
            visible = bool(item.get('visible') and bbox is not None)
            if visible:
                cx, cy, _bx, _by = center_and_bottom(bbox)
                screen_x = cx
                screen_y = cy
            else:
                screen_x = ''
                screen_y = ''
            writer.writerow({'frame': frame, 'screen_x': screen_x,
                             'screen_y': screen_y, 'visible': 1 if visible else 0})


def save_tracking_json(path, target_name, video_path, common_settings, results):
    data = {
        'target_name': target_name,
        'video_path': video_path,
        'common_settings': common_settings,
        'frames': [],
    }
    for frame in sorted(results):
        item = results[frame]
        bbox = item.get('bbox')
        mask = item.get('mask')
        data['frames'].append({
            'frame': frame,
            'bbox': list(bbox) if bbox is not None else None,
            'mask_area': int(cv2.countNonZero(mask)) if mask is not None else 0,
            'confidence': float(item.get('confidence', 0.0)),
            'visible': bool(item.get('visible') and bbox is not None),
        })
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

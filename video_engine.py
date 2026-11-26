import cv2


class VideoEngine:
    def __init__(self, path=None):
        self.capture = None
        self.path = None
        self.width = 0
        self.height = 0
        self.fps = 30.0
        self.frame_count = 0
        self.duration = 0.0
        if path:
            self.open(path)

    def open(self, path):
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            raise ValueError(f'영상 파일을 열 수 없습니다: {path}')

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
            capture.release()
            raise ValueError('영상의 해상도, 프레임레이트 또는 프레임 수가 올바르지 않습니다.')

        if self.capture:
            self.capture.release()
        self.capture = capture
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = frame_count
        self.duration = frame_count / fps

    def close(self):
        if self.capture:
            self.capture.release()
        self.capture = None

    def read_frame(self, frame_number):
        if not self.capture:
            raise RuntimeError('열린 영상이 없습니다.')
        frame_number = max(0, min(int(frame_number), self.frame_count - 1))
        if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number):
            raise RuntimeError('프레임 위치를 변경할 수 없습니다.')
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError(f'{frame_number}번 프레임을 읽을 수 없습니다.')
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def metadata(self):
        return {
            'path': self.path,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'duration': self.duration,
        }

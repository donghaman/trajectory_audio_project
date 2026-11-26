import os


class AudioTrack:
    def __init__(self, track_id, path, category='FX', delivery='Channel',
                 start_frame=0, end_frame=None, volume_db=0.0,
                 trajectory_path='', fx_subtype='Foley'):
        self.id = int(track_id)
        self.path = str(path)
        self.name = os.path.basename(path)
        self.category = category
        self.delivery = delivery
        self.start_frame = int(start_frame)
        self.end_frame = end_frame
        self.volume_db = float(volume_db)
        self.trajectory_path = trajectory_path
        self.fx_subtype = fx_subtype

    def validate(self, frame_count):
        if not os.path.isfile(self.path):
            raise ValueError(f'음원 파일을 찾을 수 없습니다: {self.path}')
        if self.category not in ('DX', 'FX', 'MX'):
            raise ValueError('음원 유형은 DX, FX, MX 중 하나여야 합니다.')
        if self.delivery not in ('Channel', 'Object'):
            raise ValueError('전달 방식은 Channel 또는 Object여야 합니다.')
        if self.start_frame < 0:
            raise ValueError('시작 프레임은 0 이상이어야 합니다.')
        if self.end_frame is not None:
            if self.end_frame > frame_count:
                raise ValueError('종료 프레임은 영상 전체 프레임 수를 초과할 수 없습니다.')
            if self.end_frame < self.start_frame:
                raise ValueError('종료 프레임은 시작 프레임보다 빠를 수 없습니다.')

    def to_dict(self):
        return {
            'id': self.id,
            'path': self.path,
            'name': self.name,
            'category': self.category,
            'delivery': self.delivery,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'volume_db': self.volume_db,
            'trajectory_path': self.trajectory_path,
            'fx_subtype': self.fx_subtype,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            track_id=data['id'], path=data['path'], category=data.get('category', 'FX'),
            delivery=data.get('delivery', 'Channel'), start_frame=data.get('start_frame', 0),
            end_frame=data.get('end_frame'), volume_db=data.get('volume_db', 0.0),
            trajectory_path=data.get('trajectory_path', ''),
            fx_subtype=data.get('fx_subtype', 'Foley'))

import json
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from audio_engine import AudioEngine
from audio_tracks import AudioTrack
from conventions import azimuth_to_xy, screen_to_direction, vertical_fov
from path_io import read_trajectory, save_stage3_trajectory, read_stage3_trajectory
from renderer import mix_tracks, export_four_channel_wav, mix_tracks_stereo, export_stereo_wav
from trajectory import Trajectory
from video_engine import VideoEngine
from object_tracking import (create_initial_mask, mask_bbox, track_video,
                             save_tracking_csv, save_screen_trajectory_csv,
                             save_tracking_json, center_and_bottom)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_ROOT = os.path.join(PROJECT_ROOT, 'Sample')


def first_existing(*paths):
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return ''


def find_sample(extension, preferred_names=()):
    matches=[]
    for root, _dirs, files in os.walk(SAMPLE_ROOT):
        for name in files:
            if name.lower().endswith(extension.lower()):
                matches.append(os.path.join(root,name))
    matches.sort()
    for preferred in preferred_names:
        for path in matches:
            if preferred.lower() in os.path.basename(path).lower():
                return path
    return matches[0] if matches else ''


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('궤적 기반 4채널 실시간 렌더링')
        self.geometry('1180x760')
        self.engine = AudioEngine()
        self.video_engine = VideoEngine()
        self.trajectory = None
        self.frames = []
        self.settings = self.default_settings()
        self.settings.update(self.load_config())

        default_csv = first_existing(
            os.path.join(SAMPLE_ROOT, '동선', 'object_path_1.csv'),
            os.path.join(SAMPLE_ROOT, '동선', 'object_path_2.csv'))
        default_source = find_sample('.wav', ('moving_sound',))
        default_test = find_sample('.wav', ('channel', '테스트', 'tone'))

        self.csv_path = tk.StringVar(value=default_csv)
        self.source_path = tk.StringVar(value=default_source)
        self.test_path = tk.StringVar(value=default_test)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)
        self.output_tab = ttk.Frame(self.notebook)
        self.play_tab = ttk.Frame(self.notebook)
        self.cal_tab = ttk.Frame(self.notebook)
        self.video_tab = ttk.Frame(self.notebook)
        self.render_tab = ttk.Frame(self.notebook)
        self.audio_tracks=[]
        self.next_track_id=1
        self.notebook.add(self.output_tab, text='출력 확인')
        self.notebook.add(self.play_tab, text='재생')
        self.notebook.add(self.cal_tab, text='청감 보정')
        self.notebook.add(self.video_tab, text='영상 및 방향 계산')
        self.notebook.add(self.render_tab, text='음원 및 렌더링')
        self.build_output_tab()
        self.build_play_tab()
        self.build_cal_tab()
        self.build_video_tab()
        self.build_render_tab()
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        if self.csv_path.get():
            try:
                self.frames = read_trajectory(self.csv_path.get())
                self.draw_trajectory()
            except Exception as exc:
                messagebox.showerror('샘플 CSV 오류', str(exc))
        self.refresh_devices()
        self.status_after = self.after(100, self.update_status)
        self.last_output_mode = self.mode_var.get()

        self.tracking_targets=[]
        self.active_tracking_target_id=None
        self.tracking_thread=None
        self.tracking_stop_event=None
        self.tracking_progress=0

    def default_settings(self):
        return {
            'mapping': {'FL':1,'FR':2,'RL':3,'RR':4},
            'speaker_db': {'FL':-12,'FR':-12,'RL':-12,'RR':-12},
            'master_db':-24,
            'mode':'speaker','renderer_method':'binaural','panning_law':'-3','elevation_strength':1.0,
            'r0':1.0,'distance_exponent':1.0,'highpass_strength':1.0,
            'reverb_amount':0.0,'doppler_strength':0.0,'doppler_enabled':False,
            'blocksize':1024,'latency':'high','output_device':None,
            'horizontal_fov':90.0,
            'person_height_m':1.75,'camera_height_m':1.60,
            'helicopter_length_m':12.00,'close_helicopter_distance_m':6.00,
            'distance_reference_m':10.00
        }

    def load_config(self):
        if not os.path.exists(CONFIG_PATH):
            return {}
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def save_config(self):
        self.collect_settings()
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.settings, f, ensure_ascii=False, indent=2)

    def collect_settings(self):
        s=self.settings
        s['mode']=self.mode_var.get()
        s['renderer_method']=self.renderer_method_var.get()
        s['mapping']={name:self.mapping_vars[name].get() for name in ('FL','FR','RL','RR')}
        s['speaker_db']={name:int(self.speaker_vars[name].get()) for name in ('FL','FR','RL','RR')}
        s['master_db']=int(self.master_var.get())
        s['panning_law']=self.panning_var.get()
        s['elevation_strength']=float(self.elevation_var.get())
        s['r0']=float(self.r0_var.get())
        s['distance_exponent']=float(self.distance_k_var.get())
        s['highpass_strength']=float(self.highpass_var.get())
        s['reverb_amount']=float(self.reverb_var.get())
        s['doppler_strength']=float(self.doppler_var.get()) if self.doppler_on_var.get() else 0.0
        s['doppler_enabled']=bool(self.doppler_on_var.get())
        s['blocksize']=int(self.blocksize_var.get())
        s['latency']=self.latency_var.get()
        s['output_device']=self.selected_device()
        if hasattr(self,'horizontal_fov_var'):
            s['horizontal_fov']=float(self.horizontal_fov_var.get())
        s.setdefault('person_height_m',1.75)
        s.setdefault('camera_height_m',1.60)
        s.setdefault('helicopter_length_m',12.00)
        s.setdefault('close_helicopter_distance_m',6.00)
        s.setdefault('distance_reference_m',10.00)

    def refresh_devices(self):
        try:
            devices=self.engine.devices()
        except Exception as exc:
            messagebox.showerror('오디오 장치 오류', str(exc)); return
        self.device_items={f'{i}: {name} ({ch}ch)': i for i,name,ch,_host in devices}
        self.device_info={i: ch for i,_name,ch,_host in devices}
        self.device_combo['values']=list(self.device_items)
        preferred=self.settings.get('output_device')
        index=0
        if preferred is not None:
            for label, device_index in self.device_items.items():
                if device_index == preferred:
                    index=list(self.device_items).index(label)
                    break
        if self.device_items:
            self.device_combo.current(index)
            self.update_device_channel_label()

    def build_output_tab(self):
        p=ttk.Frame(self.output_tab, padding=12); p.pack(fill='both', expand=True)
        ttk.Label(p,text='출력 장치').grid(row=0,column=0,sticky='w')
        self.device_combo=ttk.Combobox(p,state='readonly',width=70)
        self.device_combo.grid(row=0,column=1,sticky='w')
        self.device_combo.bind('<<ComboboxSelected>>',lambda _e:self.update_device_channel_label())
        ttk.Button(p,text='목록 새로고침',command=self.refresh_devices).grid(row=0,column=2,padx=6)

        self.device_channel_var=tk.StringVar(value='최대 출력 채널: -')
        ttk.Label(p,textvariable=self.device_channel_var).grid(row=0,column=3,padx=10)

        fileframe=ttk.LabelFrame(p,text='출력 확인 음원',padding=8)
        fileframe.grid(row=1,column=0,columnspan=4,sticky='ew',pady=8)
        ttk.Entry(fileframe,textvariable=self.test_path,width=70).grid(row=0,column=0)
        ttk.Button(fileframe,text='선택',command=lambda:self.pick_file(self.test_path,[('WAV','*.wav')])).grid(row=0,column=1,padx=5)
        ttk.Button(fileframe,text='채널 1→2→3→4 순차 테스트',command=self.test_channels).grid(row=1,column=0,pady=8,sticky='w')
        ttk.Button(fileframe,text='테스트 정지',command=self.stop_test).grid(row=1,column=1,padx=5)
        self.test_channel_var=tk.StringVar(value='테스트 대기')
        ttk.Label(fileframe,textvariable=self.test_channel_var).grid(row=1,column=2,padx=12)

        mapframe=ttk.LabelFrame(p,text='스피커 실제 출력 채널',padding=8)
        mapframe.grid(row=2,column=0,columnspan=4,sticky='ew')
        self.mapping_vars={}
        for i,name in enumerate(('FL','FR','RL','RR')):
            ttk.Label(mapframe,text=name).grid(row=0,column=i*2)
            v=tk.IntVar(value=self.settings['mapping'][name]); self.mapping_vars[name]=v
            ttk.Spinbox(mapframe,from_=1,to=4,textvariable=v,width=4).grid(row=0,column=i*2+1)

        vframe=ttk.LabelFrame(p,text='볼륨',padding=8)
        vframe.grid(row=3,column=0,columnspan=4,sticky='ew',pady=8)
        self.speaker_vars={}
        for i,name in enumerate(('FL','FR','RL','RR')):
            ttk.Label(vframe,text=f'{name} dB').grid(row=i//2,column=(i%2)*2)
            v=tk.IntVar(value=self.settings['speaker_db'][name]); self.speaker_vars[name]=v
            ttk.Scale(vframe,from_= -24,to=6,variable=v,orient='horizontal',length=160).grid(row=i//2,column=(i%2)*2+1)
        ttk.Label(vframe,text='전체 dB').grid(row=2,column=0)
        self.master_var=tk.IntVar(value=self.settings['master_db'])
        ttk.Scale(vframe,from_= -60,to=0,variable=self.master_var,orient='horizontal',length=160).grid(row=2,column=1)

        modeframe=ttk.LabelFrame(p,text='출력 모드',padding=8)
        modeframe.grid(row=4,column=0,columnspan=4,sticky='ew')
        self.mode_var=tk.StringVar(value=self.settings['mode'])
        ttk.Radiobutton(modeframe,text='4채널 스피커',variable=self.mode_var,value='speaker',command=self.on_output_mode_change).grid(row=0,column=0)
        ttk.Radiobutton(modeframe,text='2채널 이어폰/헤드폰',variable=self.mode_var,value='earphone',command=self.on_output_mode_change).grid(row=0,column=1,padx=10)
        ttk.Label(modeframe,text='렌더링 방식').grid(row=1,column=0,sticky='w',pady=(8,0))
        self.renderer_method_var=tk.StringVar(value=self.settings.get('renderer_method','binaural'))
        ttk.Radiobutton(modeframe,text='HRTF-style Binaural Rendering',variable=self.renderer_method_var,value='binaural').grid(row=1,column=1,sticky='w',pady=(8,0))
        ttk.Radiobutton(modeframe,text='Stereo Downmix',variable=self.renderer_method_var,value='downmix').grid(row=1,column=2,sticky='w',pady=(8,0),padx=8)

        streamframe=ttk.LabelFrame(p,text='스트림 설정',padding=8)
        streamframe.grid(row=5,column=0,columnspan=4,sticky='ew',pady=8)
        ttk.Label(streamframe,text='blocksize').grid(row=0,column=0)
        self.blocksize_var=tk.StringVar(value=str(self.settings['blocksize']))
        ttk.Combobox(streamframe,textvariable=self.blocksize_var,values=['256','512','1024','2048'],width=8).grid(row=0,column=1)
        ttk.Label(streamframe,text='latency').grid(row=0,column=2,padx=10)
        self.latency_var=tk.StringVar(value=self.settings['latency'])
        ttk.Combobox(streamframe,textvariable=self.latency_var,values=['low','high'],width=8).grid(row=0,column=3)
        ttk.Button(streamframe,text='설정 저장',command=self.save_config).grid(row=0,column=4,padx=15)

    def build_play_tab(self):
        p=ttk.Frame(self.play_tab,padding=8); p.pack(fill='both',expand=True)
        top=ttk.Frame(p); top.pack(fill='x')
        ttk.Label(top,text='CSV').grid(row=0,column=0)
        ttk.Entry(top,textvariable=self.csv_path,width=55).grid(row=0,column=1)
        ttk.Button(top,text='선택',command=lambda:self.pick_csv()).grid(row=0,column=2,padx=4)
        ttk.Label(top,text='음원').grid(row=1,column=0)
        ttk.Entry(top,textvariable=self.source_path,width=55).grid(row=1,column=1)
        ttk.Button(top,text='선택',command=lambda:self.pick_file(self.source_path,[('Audio','*.wav *.flac *.ogg')])).grid(row=1,column=2,padx=4)

        controls=ttk.Frame(p); controls.pack(fill='x',pady=6)
        ttk.Button(controls,text='재생',command=self.start_playback).pack(side='left')
        ttk.Button(controls,text='정지',command=self.stop_playback).pack(side='left',padx=4)
        ttk.Button(controls,text='처음으로',command=self.reset_playback).pack(side='left')
        self.status_var=tk.StringVar(value='대기'); ttk.Label(controls,textvariable=self.status_var).pack(side='left',padx=14)

        body=ttk.Frame(p); body.pack(fill='both',expand=True)
        plotframe=ttk.Frame(body); plotframe.pack(side='left',fill='both',expand=True)
        self.fig,self.ax=plt.subplots(figsize=(6,5)); self.canvas=FigureCanvasTkAgg(self.fig,master=plotframe); self.canvas.get_tk_widget().pack(fill='both',expand=True)
        self.init_plot()

        info=ttk.LabelFrame(body,text='재생 상태',padding=10); info.pack(side='right',fill='y',padx=8)
        self.info_vars={k:tk.StringVar(value='-') for k in ('time','az','el','dist','FL','FR','RL','RR')}
        labels={'time':'시간','az':'방위각','el':'고도','dist':'거리','FL':'FL 게인','FR':'FR 게인','RL':'RL 게인','RR':'RR 게인'}
        for i,(k,label) in enumerate(labels.items()):
            ttk.Label(info,text=label).grid(row=i,column=0,sticky='w'); ttk.Label(info,textvariable=self.info_vars[k]).grid(row=i,column=1,sticky='w')

    def init_plot(self):
        self.ax.clear(); self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('X (오른쪽 +)'); self.ax.set_ylabel('Y (정면 +)')
        for name,(az) in {'FL':-30,'FR':30,'RL':-135,'RR':135}.items():
            x,y=azimuth_to_xy(az,0,1)
            self.ax.scatter([x],[y],s=80); self.ax.annotate(name,(x,y),xytext=(5,5),textcoords='offset points')
        self.ax.arrow(0,-0.15,0,0.25,head_width=0.06,length_includes_head=True)
        self.ax.set_xlim(-2,2); self.ax.set_ylim(-2,2)
        self.current_dot=self.ax.scatter([0],[0],c='red',s=80)
        self.canvas.draw()

    def build_cal_tab(self):
        p=ttk.Frame(self.cal_tab,padding=12); p.pack(fill='both',expand=True)
        self.panning_var=tk.StringVar(value=self.settings['panning_law'])
        ttk.Label(p,text='패닝 법칙').grid(row=0,column=0,sticky='w')
        ttk.Combobox(p,textvariable=self.panning_var,values=['-3','-4.5','-6'],width=10,state='readonly').grid(row=0,column=1)
        self.elevation_var=tk.DoubleVar(value=self.settings['elevation_strength'])
        self.r0_var=tk.DoubleVar(value=self.settings['r0'])
        self.distance_k_var=tk.DoubleVar(value=self.settings['distance_exponent'])
        self.highpass_var=tk.DoubleVar(value=self.settings['highpass_strength'])
        self.reverb_var=tk.DoubleVar(value=self.settings['reverb_amount'])
        self.doppler_var=tk.DoubleVar(value=self.settings['doppler_strength'])
        self.doppler_on_var=tk.BooleanVar(value=self.settings['doppler_enabled'])
        rows=[('고도 분배 강도',self.elevation_var,0,1.5),('최소 거리 r0 (m)',self.r0_var,0.5,10),('거리 감쇠 지수',self.distance_k_var,0,2),('고역 감쇠 강도',self.highpass_var,0,1),('잔향 양',self.reverb_var,0,1),('도플러 강도',self.doppler_var,0,1)]
        for i,(label,var,a,b) in enumerate(rows,1):
            ttk.Label(p,text=label).grid(row=i,column=0,sticky='w',pady=4)
            scale=ttk.Scale(p,from_=a,to=b,variable=var,orient='horizontal',length=280); scale.grid(row=i,column=1)
            ttk.Entry(p,textvariable=var,width=8).grid(row=i,column=2,padx=8)
        ttk.Checkbutton(p,text='도플러 사용',variable=self.doppler_on_var).grid(row=7,column=0,sticky='w')
        ttk.Button(p,text='설정 저장',command=self.save_config).grid(row=8,column=0,pady=12,sticky='w')

    def build_video_tab(self):
        p=ttk.Frame(self.video_tab,padding=8); p.pack(fill='both',expand=True)

        top=ttk.Frame(p); top.pack(fill='x')
        ttk.Label(top,text='영상').grid(row=0,column=0)
        self.video_path=tk.StringVar(value='')
        ttk.Entry(top,textvariable=self.video_path,width=65).grid(row=0,column=1)
        ttk.Button(top,text='영상 열기',command=self.open_video).grid(row=0,column=2,padx=5)

        controls=ttk.Frame(p); controls.pack(fill='x',pady=6)
        ttk.Button(controls,text='이전 프레임',command=lambda:self.step_video(-1)).pack(side='left')
        ttk.Button(controls,text='다음 프레임',command=lambda:self.step_video(1)).pack(side='left',padx=4)
        ttk.Label(controls,text='프레임').pack(side='left',padx=(12,4))
        self.video_frame_var=tk.IntVar(value=0)
        self.video_slider=ttk.Scale(controls,from_=0,to=1,variable=self.video_frame_var,orient='horizontal',length=520)
        self.video_slider.pack(side='left')
        self.video_slider.bind('<ButtonRelease-1>',lambda _e:self.show_video_frame())

        fov=ttk.LabelFrame(p,text='화면 방향 설정',padding=8); fov.pack(fill='x')
        ttk.Label(fov,text='가로 화각 (°)').grid(row=0,column=0)
        self.horizontal_fov_var=tk.DoubleVar(value=self.settings.get('horizontal_fov',90.0))
        ttk.Entry(fov,textvariable=self.horizontal_fov_var,width=8).grid(row=0,column=1,padx=5)
        ttk.Button(fov,text='화각 적용',command=self.update_vertical_fov).grid(row=0,column=2,padx=5)
        self.vertical_fov_var=tk.StringVar(value='세로 화각: -')
        ttk.Label(fov,textvariable=self.vertical_fov_var).grid(row=0,column=3,padx=10)

        tracking=ttk.LabelFrame(p,text='대상 추적 — ISAD 3회차 1단계',padding=8); tracking.pack(fill='x',pady=4)
        ttk.Radiobutton(tracking,text='추적 대상 선택',variable=self.tracking_mode,value='target').grid(row=0,column=0)
        ttk.Button(tracking,text='전체 프레임 추적 시작',command=self.start_target_tracking).grid(row=0,column=1,padx=5)
        ttk.Button(tracking,text='추적 정지',command=self.stop_target_tracking).grid(row=0,column=2,padx=5)
        ttk.Button(tracking,text='추적 데이터 CSV 저장',command=self.save_tracking_data_csv).grid(row=0,column=3,padx=5)
        ttk.Button(tracking,text='화면 동선 CSV 저장',command=self.save_tracking_screen_csv).grid(row=0,column=4,padx=5)
        ttk.Button(tracking,text='추적 JSON 저장',command=self.save_tracking_json).grid(row=0,column=5,padx=5)
        self.tracking_progress_var=tk.DoubleVar(value=0)
        ttk.Progressbar(tracking,variable=self.tracking_progress_var,maximum=100,length=280).grid(row=1,column=0,columnspan=4,sticky='ew',pady=5)
        self.target_tracking_status=tk.StringVar(value='영상 프레임에서 헬기를 클릭하여 대상을 선택하세요.')
        ttk.Label(tracking,textvariable=self.target_tracking_status).grid(row=1,column=4,columnspan=2,sticky='w')
        self.target_tree=ttk.Treeview(tracking,columns=('id','name','frame','area','confidence'),show='headings',height=3)
        for col,label,width in (('id','대상',50),('name','이름',100),('frame','선택프레임',80),('area','마스크면적',90),('confidence','신뢰도',80)):
            self.target_tree.heading(col,text=label); self.target_tree.column(col,width=width)
        self.target_tree.grid(row=2,column=0,columnspan=6,sticky='ew',pady=4)
        self.target_tree.bind('<<TreeviewSelect>>',self.on_target_selected)

    def open_video(self):
        path=self.video_path.get()
        if not path:
            path=filedialog.askopenfilename(filetypes=[('Video',['*.mp4','*.mov','*.avi','*.mkv'])])
            if not path:return
            self.video_path.set(path)
        try:
            self.video_engine.open(path)
            meta=self.video_engine.metadata()
            self.video_slider.configure(to=max(0,meta['frame_count']-1))
            self.video_frame_var.set(0)
            self.video_info['resolution'].set(f"해상도: {meta['width']} × {meta['height']}")
            self.video_info['fps'].set(f"FPS: {meta['fps']:.3f}")
            self.video_info['duration'].set(f"길이: {meta['duration']:.3f} s")
            self.trajectory = Trajectory(meta['frame_count'], meta['fps'], meta['width'], meta['height'])
            self.update_vertical_fov()
            self.show_video_frame()
        except Exception as exc:
            messagebox.showerror('영상 오류',str(exc))

    def show_video_frame(self):
        if not self.video_engine.capture:
            return
        frame_number=int(self.video_frame_var.get())
        try:
            image=self.video_engine.read_frame(frame_number)
        except Exception as exc:
            messagebox.showerror('프레임 오류',str(exc));return
        self.video_ax.clear();self.video_ax.set_axis_off()
        self.video_ax.imshow(image)
        if self.trajectory:
            if self.trajectory.origin_px:
                ox,oy=self.trajectory.origin_px
                self.video_ax.scatter([ox],[oy],c='yellow',s=120,marker='+',linewidths=2,label='기준 위치')
            keyframes=sorted(self.trajectory.keyframes)
            if keyframes:
                kx=[self.trajectory.keyframes[f][0] for f in keyframes]
                ky=[self.trajectory.keyframes[f][1] for f in keyframes]
                kv=[self.trajectory.keyframes[f][2] for f in keyframes]
                self.video_ax.scatter(kx,ky,c=['lime' if v else 'gray' for v in kv],s=55,marker='o',label='키프레임')
            rows=self.trajectory.build_trajectory()
            visible=[r for r in rows if r['visible']==1 and r['x_px'] is not None]
            if len(visible)>1:
                sx=[r['x_px'] for r in visible]; sy=[r['y_px'] for r in visible]
                self.video_ax.plot(sx,sy,c='cyan',linewidth=1.5,alpha=0.8,label='보간 동선')
            current=self.trajectory.interpolate_position(frame_number)
            if current:
                cx,cy=current
                self.video_ax.scatter([cx],[cy],c='red',s=90,marker='x',linewidths=2,label='현재 위치')
        if self.video_marker is not None:
            if len(self.video_marker)==3:
                _kind,mx,my=self.video_marker
            else:
                mx,my=self.video_marker
            self.video_ax.scatter([mx],[my],c='red',s=100,marker='x',linewidths=2)
        active=self.get_active_tracking_target()
        if active:
            results=active.get('results')
            if results:
                points=[]
                for point_frame,item in sorted(results.items()):
                    if item.get('visible') and item.get('bbox'):
                        cx,cy,_bx,_by=center_and_bottom(item['bbox'])
                        points.append((cx,cy))
                if points:
                    sx=[p[0] for p in points]; sy=[p[1] for p in points]
                    self.video_ax.plot(sx,sy,c='yellow',linewidth=1.2,alpha=0.85,label='추적 동선')
            item=(results or {}).get(frame_number)
            if item:
                mask=item.get('mask')
                bbox=item.get('bbox')
                if mask is not None:
                    colored=np.zeros((*mask.shape,4),dtype=np.float32)
                    colored[...,1]=mask.astype(np.float32)/255.0
                    colored[...,3]=(mask.astype(np.float32)/255.0)*0.28
                    self.video_ax.imshow(colored)
                if bbox:
                    x,y,w,h=bbox
                    from matplotlib.patches import Rectangle
                    self.video_ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='lime',linewidth=1.5))
                    cx,cy,bx,by=center_and_bottom(bbox)
                    self.video_ax.scatter([cx],[cy],c='cyan',s=35,marker='o')
                    self.video_ax.scatter([bx],[by],c='red',s=35,marker='s')
        self.canvas.draw_idle()
        meta=self.video_engine.metadata()
        self.video_info['frame'].set(f'프레임: {frame_number}')
        self.video_info['time'].set(f'시간: {frame_number/meta["fps"]:.3f} s')

    def step_video(self,step):
        if not self.video_engine.capture:
            return
        frame=max(0,min(int(self.video_frame_var.get())+int(step),self.video_engine.frame_count-1))
        self.video_frame_var.set(frame)
        self.show_video_frame()

    def update_vertical_fov(self):
        if not hasattr(self,'vertical_fov_var'):
            return
        try:
            hfov=float(self.horizontal_fov_var.get())
            if not 0<hfov<180:
                raise ValueError
            if self.video_engine.capture:
                vfov=vertical_fov(hfov,self.video_engine.width,self.video_engine.height)
                self.vertical_fov_var.set(f'세로 화각: {vfov:.2f}°')
            else:
                self.vertical_fov_var.set('세로 화각: 영상 필요')
        except Exception:
            messagebox.showerror('화각 오류','가로 화각은 0보다 크고 180보다 작아야 합니다.')

    def on_video_click(self,event):
        if event.inaxes != self.video_ax or not self.video_engine.capture:
            return
        if event.xdata is None or event.ydata is None:
            return
        x=float(event.xdata); y=float(event.ydata)
        az,el,nx,ny=screen_to_direction(x,y,self.video_engine.width,self.video_engine.height,float(self.horizontal_fov_var.get()))
        frame=int(self.video_frame_var.get())

        if hasattr(self,'tracking_mode') and self.tracking_mode.get()=='origin':
            if self.trajectory:
                self.trajectory.set_origin(x,y)
            self.video_marker=('origin',x,y)
            self.trajectory_status.set(f'기준 위치 설정: ({x:.1f}, {y:.1f}) px')
        else:
            if self.trajectory:
                self.trajectory.set_keyframe(frame,x,y,1)
                self.trajectory.set_front_angle(float(self.front_angle_var.get()))
            self.video_marker=('object',x,y)

        if hasattr(self,'tracking_mode') and self.tracking_mode.get()=='target':
            self.select_tracking_target(x,y,frame)
            self.video_info['pixel'].set(f'클릭 픽셀: ({x:.1f}, {y:.1f})')
            self.video_info['normalized'].set(f'정규화: ({nx:.3f}, {ny:.3f})')
            self.video_info['azimuth'].set(f'방위각: {az:.2f}°')
            self.video_info['elevation'].set(f'고도: {el:.2f}°')
            self.video_info['frame'].set(f'프레임: {frame}')
            return

        self.video_info['pixel'].set(f'클릭 픽셀: ({x:.1f}, {y:.1f})')
        self.video_info['normalized'].set(f'정규화: ({nx:.3f}, {ny:.3f})')
        self.video_info['azimuth'].set(f'방위각: {az:.2f}°')
        self.video_info['elevation'].set(f'고도: {el:.2f}°')
        self.video_info['frame'].set(f'프레임: {frame}')
        self.refresh_trajectory_overlay()

    def set_tracking_mode(self,mode):
        self.tracking_mode.set(mode)

    def delete_selected_keyframe(self):
        if not self.trajectory:
            return
        selected=self.keyframe_tree.selection()
        if not selected:
            messagebox.showwarning('키프레임','삭제할 키프레임을 선택하세요.')
            return
        frame=int(self.keyframe_tree.item(selected[0],'values')[0])
        self.trajectory.delete_keyframe(frame)
        self.refresh_trajectory_overlay()

    def set_invisible_range(self,visible):
        if not self.trajectory:
            return
        start=int(self.invisible_start.get()); end=int(self.invisible_end.get())
        self.trajectory.set_visible_range(start,end,visible)
        self.refresh_trajectory_overlay()

    def refresh_keyframe_list(self):
        if not hasattr(self,'keyframe_tree'):
            return
        for item in self.keyframe_tree.get_children():
            self.keyframe_tree.delete(item)
        if not self.trajectory:
            return
        for frame in sorted(self.trajectory.keyframes):
            x,y,v=self.trajectory.keyframes[frame]
            self.keyframe_tree.insert('', 'end', values=(frame,f'{x:.1f}',f'{y:.1f}',v))

    def refresh_trajectory_overlay(self):
        self.refresh_keyframe_list()
        if not self.trajectory:
            return
        frame=int(self.video_frame_var.get())
        try:
            self.trajectory.set_front_angle(float(self.front_angle_var.get()))
            relative=self.trajectory.calculate_relative_position(frame)
            if relative:
                self.trajectory_status.set(f"프레임 {frame}: visible=1, 방위각={relative['azimuth_deg']:.2f}°, 거리={relative['distance_px']:.1f}px")
            else:
                self.trajectory_status.set(f'프레임 {frame}: visible=0')
        except Exception as exc:
            self.trajectory_status.set(str(exc))
        self.show_video_frame()

    def pick_file(self,var,filetypes):
        path=filedialog.askopenfilename(filetypes=filetypes)
        if path: var.set(path)

    def pick_csv(self):
        path=filedialog.askopenfilename(filetypes=[('CSV','*.csv')])
        if not path:return
        try:
            self.frames=read_trajectory(path)
        except Exception as exc:
            messagebox.showerror('CSV 오류',str(exc)); return
        self.csv_path.set(path)
        self.draw_trajectory()

    def draw_trajectory(self):
        if not self.frames:return
        xs=[]; ys=[]; cs=[]; sizes=[]
        max_r=0
        for r in self.frames:
            x,y=azimuth_to_xy(r['azimuth_deg'],r['elevation_deg'],r['distance_m'])
            xs.append(x);ys.append(y);cs.append(r['time_s']);sizes.append(20+max(0,r['elevation_deg']/90)*40)
            max_r=max(max_r,r['distance_m'])
        self.ax.clear();self.ax.set_aspect('equal',adjustable='box')
        self.ax.plot(xs,ys,color='gray',alpha=0.35)
        sc=self.ax.scatter(xs,ys,c=cs,cmap='viridis',s=sizes)
        for name,az in {'FL':-30,'FR':30,'RL':-135,'RR':135}.items():
            x,y=azimuth_to_xy(az,0,1);self.ax.scatter([x],[y],s=80);self.ax.annotate(name,(x,y),xytext=(5,5),textcoords='offset points')
        lim=max(2.0,max_r*1.15)
        self.ax.set_xlim(-lim,lim);self.ax.set_ylim(-lim,lim)
        self.ax.set_xlabel('X (오른쪽 +)');self.ax.set_ylabel('Y (정면 +)')
        self.ax.arrow(0,-lim*0.08,0,lim*0.18,head_width=lim*0.025,length_includes_head=True)
        self.current_dot=self.ax.scatter([0],[0],c='red',s=100)
        self.fig.colorbar(sc,ax=self.ax,label='시간 (s)')
        self.canvas.draw()

    def selected_device(self):
        label=self.device_combo.get()
        return self.device_items.get(label)

    def test_channels(self):
        if not self.test_path.get():
            messagebox.showwarning('음원 선택','채널 테스트용 WAV를 선택하세요.')
            return
        device=self.selected_device()
        channels=self.device_info.get(device) if hasattr(self,'device_info') else None
        mode=self.mode_var.get()
        required=2 if mode=='earphone' else 4
        if channels is not None and channels < required:
            messagebox.showerror('출력 장치 오류',f'선택한 장치는 최대 {channels}채널만 지원합니다. 이 모드에는 {required}채널이 필요합니다.')
            return
        if not messagebox.askokcancel('볼륨 확인','출력 볼륨을 낮게 설정한 상태로 테스트를 시작하시겠습니까?'):
            return
        try:
            self.collect_settings()
            self.settings['mode']=mode
            self.engine.update_settings(self.settings)
            if mode=='speaker':
                self.engine.validate_mapping()
            self.engine.set_test_source(self.test_path.get())
            self.engine.start_test(device,int(self.blocksize_var.get()),self.latency_var.get())
            self.test_channel_var.set('테스트 중')
        except Exception as exc:
            messagebox.showerror('오디오 오류',str(exc))

    def stop_test(self):
        if self.engine.mode_run == 'test':
            self.engine.stop()
        self.test_channel_var.set('테스트 정지')

    def start_playback(self):
        if not self.frames:
            if self.csv_path.get():
                try:self.frames=read_trajectory(self.csv_path.get())
                except Exception as exc:messagebox.showerror('CSV 오류',str(exc));return
            else:
                messagebox.showwarning('파일 선택','궤적 CSV를 선택하세요.');return
        if not self.source_path.get():
            messagebox.showwarning('파일 선택','이동시킬 음원을 선택하세요.');return
        try:
            self.collect_settings()
            self.settings['doppler_strength']=self.settings['doppler_strength'] if self.doppler_on_var.get() else 0.0
            self.engine.update_settings(self.settings)
            self.engine.set_frames(self.frames)
            self.engine.set_source(self.source_path.get())
            self.engine.start_trajectory(self.selected_device(),int(self.blocksize_var.get()),self.latency_var.get())
        except Exception as exc:
            messagebox.showerror('오디오 오류',str(exc))

    def stop_playback(self):
        self.engine.stop();self.status_var.set('정지')

    def reset_playback(self):
        self.engine.reset();self.status_var.set('처음으로')

    def update_status(self):
        self.collect_settings()
        self.engine.update_settings(self.settings)
        if self.engine.mode_run == 'test' and self.engine.test_channel:
            if self.mode_var.get() == 'earphone':
                side='왼쪽' if self.engine.test_channel == 1 else '오른쪽'
                self.test_channel_var.set(f'현재: {side} 출력 채널')
            else:
                names=('FL','FR','RL','RR')
                index=min(max(self.engine.test_channel-1,0),3)
                name=names[index]
                physical=self.engine.mapping[name]
                self.test_channel_var.set(f'현재: {name} → 출력 채널 {physical}')
        current_mode = self.mode_var.get()
        if current_mode != self.last_output_mode:
            if self.engine.mode_run == 'trajectory':
                try:
                    self.engine.restart_output_mode(
                        self.selected_device(), int(self.blocksize_var.get()), self.latency_var.get())
                except Exception as exc:
                    messagebox.showerror('오디오 오류', str(exc))
            self.last_output_mode = current_mode
        pos=self.engine.get_position()
        if pos:
            sec,frame,row=pos
            self.info_vars['time'].set(f'{sec:.3f} s');self.info_vars['az'].set(f'{row["azimuth_deg"]:.1f}°')
            self.info_vars['el'].set(f'{row["elevation_deg"]:.1f}°');self.info_vars['dist'].set(f'{row["distance_m"]:.2f} m')
            if self.engine.frame_gains:
                g=self.engine.frame_gains[min(frame,len(self.engine.frame_gains)-1)]
                for name in ('FL','FR','RL','RR'):self.info_vars[name].set(f'{g[name]:.3f}')
            x,y=azimuth_to_xy(row['azimuth_deg'],row['elevation_deg'],row['distance_m'])
            if hasattr(self,'current_dot'):
                self.current_dot.set_offsets([[x,y]])
                self.canvas.draw_idle()
        if self.engine.clipped:
            self.status_var.set('경고: 출력이 ±1을 초과하여 클리핑됨')
        elif self.engine.underflow:
            self.status_var.set('경고: 오디오 언더런 발생')
        elif self.engine.mode_run!='stop':
            self.status_var.set('재생 중')
        self.status_after = self.after(33, self.update_status)

    def update_device_channel_label(self):
        device=self.selected_device()
        channels=self.device_info.get(device) if hasattr(self,'device_info') else None
        method=getattr(self,'renderer_method_var',None)
        method_text='HRTF-style Binaural Rendering' if (method and method.get()=='binaural') else 'Stereo Downmix'
        if channels is None:
            self.device_channel_var.set(f'최대 출력 채널: -  |  렌더링: {method_text}')
        else:
            self.device_channel_var.set(f'최대 출력 채널: {channels}  |  렌더링: {method_text}')

    def on_close(self):
        self.engine.stop()
        self.video_engine.close()
        if hasattr(self, 'status_after'):
            self.after_cancel(self.status_after)
        self.destroy()

    def save_trajectory_csv(self):
        if not self.trajectory:
            messagebox.showwarning('동선','열린 영상이 없습니다.')
            return
        path=filedialog.asksaveasfilename(defaultextension='.csv',filetypes=[('CSV','*.csv')])
        if not path:
            return
        try:
            rows=self.trajectory.build_trajectory()
            save_stage3_trajectory(path,rows)
            self.trajectory_status.set(f'동선 저장 완료: {path}')
        except Exception as exc:
            messagebox.showerror('동선 저장 오류',str(exc))

    def load_trajectory_csv(self):
        if not self.trajectory:
            messagebox.showwarning('동선','열린 영상이 없습니다.')
            return
        path=filedialog.askopenfilename(filetypes=[('CSV','*.csv')])
        if not path:
            return
        try:
            rows=read_stage3_trajectory(path)
            if not rows:
                raise ValueError('동선 데이터가 없습니다.')
            for row in rows:
                frame=int(row['frame'])
                visible=int(row.get('visible',0))
                if visible and row.get('x_px') is not None and row.get('y_px') is not None:
                    self.trajectory.set_keyframe(frame,row['x_px'],row['y_px'],1)
                else:
                    self.trajectory.set_visible_range(frame,frame,False)
            self.refresh_trajectory_overlay()
            self.trajectory_status.set(f'동선 불러오기 완료: {len(rows)}프레임')
        except Exception as exc:
            messagebox.showerror('동선 불러오기 오류',str(exc))

    def preview_trajectory(self):
        if not self.trajectory:
            messagebox.showwarning('시험 재생','동선이 없습니다.')
            return
        frame=int(self.video_frame_var.get())
        relative=self.trajectory.calculate_relative_position(frame) if self.trajectory.origin_px else None
        if relative:
            self.trajectory_status.set(f"시험 재생 위치: 프레임 {frame}, 방위각 {relative['azimuth_deg']:.2f}°, visible=1")
        else:
            self.trajectory_status.set(f'시험 재생 위치: 프레임 {frame}, visible=0 또는 기준 위치 미설정')

    def build_render_tab(self):
        p=ttk.Frame(self.render_tab,padding=8); p.pack(fill='both',expand=True)

        top=ttk.Frame(p); top.pack(fill='x')
        ttk.Button(top,text='음원 추가',command=self.add_audio_track).pack(side='left')
        ttk.Button(top,text='선택 음원 삭제',command=self.delete_audio_track).pack(side='left',padx=5)
        ttk.Button(top,text='미리 듣기',command=self.preview_audio_track).pack(side='left')

        self.track_tree=ttk.Treeview(p,columns=('id','name','type','delivery','start','end','volume','trajectory'),show='headings',height=8)
        headings={'id':'ID','name':'음원','type':'유형','delivery':'전달','start':'시작','end':'종료','volume':'볼륨(dB)','trajectory':'동선 CSV'}
        widths={'id':45,'name':180,'type':65,'delivery':75,'start':65,'end':65,'volume':75,'trajectory':180}
        for col,label in headings.items():
            self.track_tree.heading(col,text=label); self.track_tree.column(col,width=widths[col])
        self.track_tree.pack(fill='x',pady=6)
        self.track_tree.bind('<<TreeviewSelect>>',self.load_track_settings)

        settings=ttk.LabelFrame(p,text='선택 음원 설정',padding=8); settings.pack(fill='x')
        ttk.Label(settings,text='유형').grid(row=0,column=0)
        self.track_category=tk.StringVar(value='FX')
        ttk.Combobox(settings,textvariable=self.track_category,values=['DX','FX','MX'],state='readonly',width=8).grid(row=0,column=1)
        ttk.Label(settings,text='전달').grid(row=0,column=2,padx=(8,4))
        self.track_delivery=tk.StringVar(value='Channel')
        ttk.Combobox(settings,textvariable=self.track_delivery,values=['Channel','Object'],state='readonly',width=10).grid(row=0,column=3)
        ttk.Label(settings,text='FX 세부').grid(row=0,column=4,padx=(8,4))
        self.track_fx_subtype=tk.StringVar(value='Foley')
        ttk.Combobox(settings,textvariable=self.track_fx_subtype,values=['Foley','Ambience'],state='readonly',width=12).grid(row=0,column=5)

        ttk.Label(settings,text='시작 프레임').grid(row=1,column=0)
        self.track_start=tk.IntVar(value=0); ttk.Entry(settings,textvariable=self.track_start,width=8).grid(row=1,column=1)
        ttk.Label(settings,text='종료 프레임').grid(row=1,column=2)
        self.track_end=tk.IntVar(value=0); ttk.Entry(settings,textvariable=self.track_end,width=8).grid(row=1,column=3)
        ttk.Label(settings,text='볼륨(dB)').grid(row=1,column=4)
        self.track_volume=tk.DoubleVar(value=0.0); ttk.Entry(settings,textvariable=self.track_volume,width=8).grid(row=1,column=5)

        ttk.Button(settings,text='현재 프레임을 시작',command=lambda:self.set_track_frame('start')).grid(row=2,column=0,columnspan=2,pady=4)
        ttk.Button(settings,text='현재 프레임을 종료',command=lambda:self.set_track_frame('end')).grid(row=2,column=2,columnspan=2)
        ttk.Button(settings,text='설정 적용',command=self.apply_track_settings).grid(row=2,column=4,columnspan=2)
        ttk.Button(settings,text='동선 CSV 선택',command=self.select_track_trajectory).grid(row=3,column=0,columnspan=2,pady=4)
        self.track_trajectory=tk.StringVar(value='')
        ttk.Label(settings,textvariable=self.track_trajectory).grid(row=3,column=2,columnspan=4,sticky='w')

        timeline=ttk.LabelFrame(p,text='타임라인',padding=8); timeline.pack(fill='both',expand=True,pady=6)
        self.timeline_canvas=tk.Canvas(timeline,height=180,bg='white'); self.timeline_canvas.pack(fill='both',expand=True)

        action=ttk.Frame(p); action.pack(fill='x',pady=4)
        ttk.Button(action,text='JSON 저장',command=self.save_project_json).pack(side='left')
        ttk.Button(action,text='JSON 불러오기',command=self.load_project_json).pack(side='left',padx=5)
        ttk.Button(action,text='4채널 WAV 렌더링',command=self.render_four_channel).pack(side='left')
        ttk.Button(action,text='스테레오 WAV 렌더링',command=self.render_stereo).pack(side='left',padx=5)
        self.render_status=tk.StringVar(value='렌더링 대기')
        ttk.Label(action,textvariable=self.render_status).pack(side='left',padx=12)

    def add_audio_track(self):
        paths=filedialog.askopenfilenames(filetypes=[('Audio',['*.wav','*.flac','*.ogg'])])
        for path in paths:
            track=AudioTrack(self.next_track_id,path)
            if self.video_engine.capture:
                track.end_frame=self.video_engine.frame_count
            self.next_track_id+=1
            self.audio_tracks.append(track)
        self.refresh_track_tree()

    def delete_audio_track(self):
        selected=self.track_tree.selection()
        if not selected:
            messagebox.showwarning('음원','삭제할 음원을 선택하세요.')
            return
        track_id=int(self.track_tree.item(selected[0],'values')[0])
        self.audio_tracks=[track for track in self.audio_tracks if track.id!=track_id]
        self.refresh_track_tree()

    def refresh_track_tree(self):
        for item in self.track_tree.get_children():
            self.track_tree.delete(item)
        for track in self.audio_tracks:
            end='' if track.end_frame is None else track.end_frame
            self.track_tree.insert('', 'end', values=(track.id,track.name,track.category,track.delivery,track.start_frame,end,track.volume_db,track.trajectory_path))
        self.refresh_timeline()

    def load_track_settings(self,_event=None):
        selected=self.track_tree.selection()
        if not selected:
            return
        track_id=int(self.track_tree.item(selected[0],'values')[0])
        track=next((item for item in self.audio_tracks if item.id==track_id),None)
        if not track:
            return
        self.track_category.set(track.category)
        self.track_delivery.set(track.delivery)
        self.track_fx_subtype.set(track.fx_subtype)
        self.track_start.set(track.start_frame)
        self.track_end.set(track.end_frame or 0)
        self.track_volume.set(track.volume_db)
        self.track_trajectory.set(track.trajectory_path)

    def apply_track_settings(self):
        selected=self.track_tree.selection()
        if not selected:
            messagebox.showwarning('음원','설정을 적용할 음원을 선택하세요.')
            return
        track_id=int(self.track_tree.item(selected[0],'values')[0])
        track=next((item for item in self.audio_tracks if item.id==track_id),None)
        if not track:
            return
        track.category=self.track_category.get()
        track.delivery=self.track_delivery.get()
        track.fx_subtype=self.track_fx_subtype.get()
        track.start_frame=int(self.track_start.get())
        track.end_frame=int(self.track_end.get())
        track.volume_db=float(self.track_volume.get())
        track.trajectory_path=self.track_trajectory.get()
        frame_count=self.video_engine.frame_count if self.video_engine.capture else 10**9
        try:
            track.validate(frame_count)
        except Exception as exc:
            messagebox.showerror('음원 설정 오류',str(exc));return
        self.refresh_track_tree()

    def set_track_frame(self,which):
        frame=int(self.video_frame_var.get())
        if which=='start':
            self.track_start.set(frame)
        else:
            self.track_end.set(frame)

    def select_track_trajectory(self):
        path=filedialog.askopenfilename(filetypes=[('CSV','*.csv')])
        if path:
            self.track_trajectory.set(path)

    def preview_audio_track(self):
        selected=self.track_tree.selection()
        if not selected:
            messagebox.showwarning('음원','미리 들을 음원을 선택하세요.')
            return
        track_id=int(self.track_tree.item(selected[0],'values')[0])
        track=next((item for item in self.audio_tracks if item.id==track_id),None)
        if not track:
            return
        try:
            self.engine.set_test_source(track.path)
            self.engine.mode='speaker'
            self.engine.update_settings(self.settings)
            self.engine.start_test(self.selected_device(),int(self.blocksize_var.get()),self.latency_var.get())
            self.render_status.set(f'미리 듣기: {track.name}')
        except Exception as exc:
            messagebox.showerror('미리 듣기 오류',str(exc))

    def refresh_timeline(self):
        if not hasattr(self,'timeline_canvas'):
            return
        canvas=self.timeline_canvas
        canvas.delete('all')
        frame_count=self.video_engine.frame_count if self.video_engine.capture else 1
        width=max(canvas.winfo_width(),500)
        canvas.create_line(40,20,width-20,20,fill='#333')
        for i,track in enumerate(self.audio_tracks):
            y=45+i*25
            start=max(0,min(track.start_frame,frame_count))
            end=frame_count if track.end_frame is None else max(start,min(track.end_frame,frame_count))
            x1=40+(width-60)*start/frame_count
            x2=40+(width-60)*end/frame_count
            canvas.create_rectangle(x1,y,x2,y+15,fill='#77aadd',outline='')
            canvas.create_text(38,y+7,text=f'{track.id}:{track.category}/{track.delivery}',anchor='e',font=('TkDefaultFont',9))
        current=int(self.video_frame_var.get()) if hasattr(self,'video_frame_var') else 0
        cx=40+(width-60)*current/max(frame_count,1)
        canvas.create_line(cx,12,cx,170,fill='red',width=2)

    def save_project_json(self):
        path=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('JSON','*.json')])
        if not path:
            return
        self.collect_settings()
        data=dict(self.settings)
        data['video_path']=self.video_path.get()
        data['tracks']=[track.to_dict() for track in self.audio_tracks]
        with open(path,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
        self.render_status.set(f'프로젝트 설정 저장: {path}')

    def load_project_json(self):
        path=filedialog.askopenfilename(filetypes=[('JSON','*.json')])
        if not path:
            return
        try:
            with open(path,'r',encoding='utf-8') as f:
                data=json.load(f)
            defaults=self.default_settings()
            defaults.update(data)
            self.settings=defaults
            if hasattr(self,'mode_var'):
                self.mode_var.set(self.settings.get('mode','speaker'))
                self.renderer_method_var.set(self.settings.get('renderer_method','binaural'))
                for name in ('FL','FR','RL','RR'):
                    self.mapping_vars[name].set(self.settings['mapping'][name])
                    self.speaker_vars[name].set(self.settings['speaker_db'][name])
                self.master_var.set(self.settings['master_db'])
                self.blocksize_var.set(str(self.settings['blocksize']))
                self.latency_var.set(self.settings['latency'])
                self.last_output_mode=self.mode_var.get()
            if data.get('video_path'):
                self.video_path.set(data['video_path'])
                self.open_video()
            self.audio_tracks=[AudioTrack.from_dict(item) for item in data.get('tracks',[])]
            self.next_track_id=max([track.id for track in self.audio_tracks] or [0])+1
            self.refresh_track_tree()
            self.render_status.set(f'프로젝트 설정 불러오기: {path}')
        except Exception as exc:
            messagebox.showerror('프로젝트 불러오기 오류',str(exc))

    def render_four_channel(self):
        if not self.video_engine.capture:
            messagebox.showwarning('렌더링','영상 파일을 먼저 선택하세요.')
            return
        if not self.audio_tracks:
            messagebox.showwarning('렌더링','렌더링할 음원을 등록하세요.')
            return
        path=filedialog.asksaveasfilename(defaultextension='.wav',filetypes=[('WAV','*.wav')])
        if not path:
            return
        try:
            frame_count=self.video_engine.frame_count
            fps=self.video_engine.fps
            self.engine.stop()
            for track in self.audio_tracks:
                track
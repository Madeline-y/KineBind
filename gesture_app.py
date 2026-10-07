"""KineBind desktop enrollment, replay and explicitly enabled mouse control."""
from __future__ import annotations

import argparse
from collections import deque
import ctypes
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from kinebind.actions import FakeMouse, Mouse, WindowsMouse
from kinebind.controller import GestureController
from kinebind.data import Motion, Profile
from kinebind.sources import Point, SerialFeed, read_csv
from kinebind.storage import ProfileStore
from kinebind.calibration import CATEGORIES, LABELS, INSTRUCTIONS, AXES
from imu_protocol import Sample

ROOT = Path(__file__).resolve().parent


class GestureApp:
    def __init__(self, root: tk.Tk, store: ProfileStore, mouse: Mouse, port: str = 'COM7',
                 dry_run: bool = False, smoke: bool = False):
        self.root, self.store = root, store
        self.service = GestureController(store, mouse, threaded=not smoke,
                                         test_folder=store.folder / 'recordings' if smoke else ROOT / 'recordings',
                                         test_source='SIMULATED smoke' if smoke else 'USB realtime test')
        self.feed: SerialFeed | None = None
        self.smoke, self.dry_run = smoke, dry_run
        self.busy = False
        self.closed = False
        self.pending_slot: int | None = None
        self.saved_profiles: dict[str, str] = {}
        self.history: deque[Point] = deque(maxlen=600)
        self.last_draw = 0.
        self.poll_id: str | None = None
        self.name = tk.StringVar(value='左挥')
        self.port = tk.StringVar(value=port)
        self.profile_choice = tk.StringVar()
        self.status = tk.StringVar(value='先连接设备，再新建或加载手势。')
        self.connection = tk.StringVar(value='设备未连接')
        self.progress = tk.StringVar(value='示范 0/5；非目标记录未采集')
        self.bounds_start = tk.StringVar(value='0.000')
        self.bounds_end = tk.StringVar(value='0.000')
        self.distance = tk.StringVar(value='100')
        self.learning_state = tk.StringVar(value='尚未学习')
        self.calibration_category = tk.StringVar(value=LABELS['static'])
        self.calibration_instruction = tk.StringVar(value=INSTRUCTIONS['static'])
        self.control_state = tk.StringVar(value='控制停用 · 本次事件 0')
        self.test_data_state = tk.StringVar(value='开始识别时自动保存测试 CSV 至 recordings。')
        self.event_count = 0
        self.root.title('KineBind — 学习手势与鼠标控制' + ('（模拟鼠标）' if dry_run else ''))
        self.root.geometry('1180x820')
        self.root.minsize(1000, 720)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TButton', padding=(8, 5))
        style.configure('TLabel', font=('Microsoft YaHei UI', 9))
        self.controls: list[ttk.Button] = []
        self._build()
        self.refresh_profiles()
        self._poll()

    def button(self, frame: Any, text: str, command: Callable[[], Any], protected: bool = True) -> ttk.Button:
        button = ttk.Button(frame, text=text, command=lambda: None if protected and self.busy else self.guard(command))
        if protected:
            self.controls.append(button)
        return button

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill='x')
        ttk.Label(top, text='串口').pack(side='left')
        ttk.Entry(top, textvariable=self.port, width=10).pack(side='left', padx=6)
        self.button(top, '连接设备', self.connect).pack(side='left', padx=3)
        self.button(top, '断开', self.disconnect, False).pack(side='left', padx=3)
        ttk.Label(top, textvariable=self.connection).pack(side='left', padx=12)
        self.button(top, '停止识别 / 学习', self.stop, False).pack(side='right', padx=4)
        ttk.Label(top, text='测试模式：不移动真实鼠标' if self.dry_run else '只有点击开始识别才控制鼠标').pack(side='right')
        body = ttk.Frame(self.root, padding=(10, 0, 10, 5))
        body.pack(fill='both', expand=True)
        sidebar = ttk.Frame(body)
        sidebar.pack(side='left', fill='y', padx=(0, 10))
        self.sidebar_canvas = tk.Canvas(sidebar, width=320, highlightthickness=0)
        scroll = ttk.Scrollbar(sidebar, orient='vertical', command=self.sidebar_canvas.yview)
        scroll.pack(side='right', fill='y')
        self.sidebar_canvas.pack(side='left', fill='both', expand=True)
        self.sidebar_canvas.configure(yscrollcommand=scroll.set)
        left = ttk.Frame(self.sidebar_canvas)
        pane = self.sidebar_canvas.create_window((0, 0), window=left, anchor='nw')
        left.bind('<Configure>', lambda _: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox('all')))
        self.sidebar_canvas.bind('<Configure>', lambda event: self.sidebar_canvas.itemconfigure(pane, width=event.width))
        def scroll_sidebar(event: Any) -> None:
            x = self.sidebar_canvas.winfo_rootx()
            if x <= event.x_root < x + self.sidebar_canvas.winfo_width():
                self.sidebar_canvas.yview_scroll(-int(event.delta / 120), 'units')
        self.root.bind('<MouseWheel>', scroll_sidebar, add='+')
        right = ttk.Frame(body)
        right.pack(side='left', fill='both', expand=True)

        profiles = ttk.LabelFrame(left, text='手势', padding=8)
        profiles.pack(fill='x', pady=(0, 7))
        row = ttk.Frame(profiles)
        row.pack(fill='x')
        ttk.Entry(row, textvariable=self.name, width=19).pack(side='left', fill='x', expand=True)
        self.button(row, '新建', self.new_profile).pack(side='right', padx=(5, 0))
        row = ttk.Frame(profiles)
        row.pack(fill='x', pady=(6, 0))
        self.profile_box = ttk.Combobox(row, textvariable=self.profile_choice, state='readonly', width=24)
        self.profile_box.pack(side='left', fill='x', expand=True)
        self.button(row, '加载', self.load_profile).pack(side='right', padx=(5, 0))
        self.button(profiles, '恢复未学习草稿', self.restore_draft).pack(fill='x', pady=(3, 0))

        enrollment = ttk.LabelFrame(left, text='1  逐次录入五个完整动作', padding=8)
        enrollment.pack(fill='x', pady=(0, 7))
        ttk.Label(enrollment, textvariable=self.progress, wraplength=280).pack(anchor='w')
        self.samples = tk.Listbox(enrollment, height=5, exportselection=False, font=('Microsoft YaHei UI', 9))
        self.samples.pack(fill='x', pady=5)
        self.samples.bind('<<ListboxSelect>>', lambda _: self.guard(self.inspect_sample))
        row = ttk.Frame(enrollment)
        row.pack(fill='x')
        self.button(row, '录制新示范', self.record_new).pack(side='left')
        self.button(row, '重录选中', self.record_replace).pack(side='left', padx=3)
        self.button(enrollment, '结束录制', self.finish).pack(fill='x', pady=4)
        self.button(enrollment, '导入已有 CSV 检查', self.import_csv).pack(fill='x')
        row = ttk.Frame(enrollment)
        row.pack(fill='x', pady=5)
        ttk.Label(row, text='起点').pack(side='left')
        ttk.Entry(row, textvariable=self.bounds_start, width=9).pack(side='left', padx=3)
        ttk.Label(row, text='终点').pack(side='left')
        ttk.Entry(row, textvariable=self.bounds_end, width=9).pack(side='left', padx=3)
        ttk.Label(row, text='秒').pack(side='left')
        ttk.Label(enrollment, text='自动建议范围需确认；可拖动调整，保留完整动作。', wraplength=280).pack(anchor='w')
        self.button(enrollment, '确认保存这次示范', self.save_sample).pack(fill='x', pady=(5, 0))
        self.button(enrollment, '重新建议当前范围（仍需确认）', self.service.suggest_review).pack(fill='x')

        learning = ttk.LabelFrame(left, text='2  非目标校准与学习', padding=8)
        learning.pack(fill='x', pady=(0, 7))
        category = ttk.Combobox(learning, textvariable=self.calibration_category, values=list(LABELS.values()), state='readonly')
        category.pack(fill='x')
        category.bind('<<ComboboxSelected>>', lambda _: self.calibration_instruction.set(INSTRUCTIONS[self.category_key()]))
        ttk.Label(learning, textvariable=self.calibration_instruction, wraplength=280).pack(anchor='w', pady=2)
        self.button(learning, '采集当前类别约 10 秒', self.record_background).pack(fill='x')
        self.button(learning, '导入非目标 CSV', self.import_background).pack(fill='x', pady=4)
        self.button(learning, '单加速度轴学习并保存（左挥）', lambda: self.learn(single_accel=True)).pack(fill='x')
        self.button(learning, '自动选轴学习并保存（其他动作）', self.learn).pack(fill='x', pady=(3, 0))
        self.button(learning, '查看学习报告', self.show_report).pack(fill='x', pady=(3, 0))
        ttk.Label(learning, textvariable=self.learning_state, wraplength=280).pack(anchor='w', pady=(5, 0))

        control = ttk.LabelFrame(left, text='3  回放或实时控制', padding=8)
        control.pack(fill='x')
        row = ttk.Frame(control)
        row.pack(fill='x')
        ttk.Label(row, text='每次向左').pack(side='left')
        ttk.Entry(row, textvariable=self.distance, width=6).pack(side='left', padx=4)
        ttk.Label(row, text='像素').pack(side='left')
        self.button(row, '保存绑定', self.save_binding).pack(side='right')
        self.button(control, '回放 CSV（不控制鼠标）', self.replay).pack(fill='x', pady=5)
        row = ttk.Frame(control)
        row.pack(fill='x')
        self.button(row, '开始识别', self.start).pack(side='left', fill='x', expand=True)
        self.button(row, '停止', self.stop, False).pack(side='left', fill='x', expand=True, padx=(5, 0))
        ttk.Label(control, textvariable=self.test_data_state, wraplength=280).pack(anchor='w', pady=(5, 0))

        self.figure = Figure(figsize=(8, 5), dpi=100, layout='constrained')
        self.axes = self.figure.subplots(2, 1, sharex=True)
        self.lines = []
        colors = ('#e45756', '#4c78a8', '#39855c')
        for group, axis in enumerate(self.axes):
            for name, color in zip((('ax_g', 'ay_g', 'az_g'), ('gx_dps', 'gy_dps', 'gz_dps'))[group], colors):
                self.lines.append(axis.plot([], [], label=name, color=color, linewidth=1)[0])
            axis.legend(loc='upper right')
            axis.grid(alpha=.25)
        self.axes[0].set_ylabel('Acceleration (g)')
        self.axes[1].set_ylabel('Angular velocity (deg/s)')
        self.axes[1].set_xlabel('Seconds since first sample')
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.span = SpanSelector(self.axes[0], self.select_bounds, 'horizontal', useblit=True,
                                 props=dict(alpha=.15, facecolor='#355c8c'), interactive=True)
        ttk.Label(right, textvariable=self.control_state).pack(anchor='w')
        self.log = tk.Text(right, height=7, wrap='word', font=('Consolas', 9), state='disabled')
        self.log.pack(fill='x', pady=(4, 0))
        ttk.Label(self.root, textvariable=self.status, padding=(10, 6), wraplength=1150).pack(fill='x')

    def guard(self, operation: Callable[[], Any]) -> None:
        try:
            operation()
        except Exception as error:
            self.service.stop()
            self.status.set(str(error))
            self.append_log('提示：' + str(error))
            if self.smoke:
                raise
            messagebox.showerror('KineBind', str(error), parent=self.root)

    def append_log(self, text: str) -> None:
        self.log.configure(state='normal')
        self.log.insert('end', time.strftime('%H:%M:%S') + '  ' + text + '\n')
        if int(self.log.index('end-1c').split('.')[0]) > 200:
            self.log.delete('1.0', '50.0')
        self.log.see('end')
        self.log.configure(state='disabled')

    def _ensure_idle_recording(self) -> None:
        if self.service.recorder.state in ('countdown', 'recording'):
            raise ValueError('请先结束当前录制。')

    def refresh_profiles(self) -> None:
        profiles, errors = self.store.list_profiles()
        self.saved_profiles = {f'{p.name} [{p.id[:6]}]' + (' · 可用' if p.model else ' · 未学习'): p.id for p in profiles}
        self.profile_box['values'] = list(self.saved_profiles)
        for error in errors:
            self.append_log('记录未加载：' + error)

    def refresh_current(self) -> None:
        p = self.service.profile
        if p is None: return
        self.name.set(p.name)
        self.distance.set(str(p.pixels))
        self.samples.delete(0, 'end')
        for i, clip in enumerate(p.samples, 1):
            self.samples.insert('end', f'{i}. 完整示范  {clip.motion.duration:.2f} 秒')
        categories = '；'.join(f'{LABELS[k]} {p.calibration[k].duration:.1f}s' if k in p.calibration else f'{LABELS[k]}未采' for k in CATEGORIES)
        self.progress.set(f'示范 {len(p.samples)}/5\n{categories}')
        if p.model:
            axes = getattr(p.model, 'axes', tuple(range(6)))
            version = '新规则' if getattr(p.model, 'version', 1) == 2 else '旧规则'
            self.learning_state.set(f'{version}；轴 {"、".join(AXES[i] for i in axes)}；接受距离 ≤ {p.model.threshold:.3f}')
        else:
            self.learning_state.set(str(p.report.get('reason', '尚未学习，或样本有修改')))
        self.refresh_profiles()
        self.profile_choice.set(next((name for name, ident in self.saved_profiles.items() if ident == p.id), f'{p.name} [{p.id[:6]}]'))

    def new_profile(self) -> None:
        self._ensure_idle_recording()
        self.service.new_profile(self.name.get())
        self.pending_slot = None
        self.refresh_current()
        self.status.set('已新建手势。录入五次完整动作，每次可单独检查。')

    def load_profile(self) -> None:
        self._ensure_idle_recording()
        profile_id = self.saved_profiles.get(self.profile_choice.get())
        if not profile_id: raise ValueError('请选择一个已保存手势。')
        self.service.select(self.store.load(profile_id))
        self.pending_slot = None
        self.refresh_current()
        self.status.set('手势已加载，鼠标控制停用。点击开始识别才启用。')

    def connect(self) -> None:
        if self.feed is not None: raise ValueError('设备已连接或正在连接。')
        port = self.port.get().strip()
        if not port: raise ValueError('请输入实际串口。')
        self.history.clear()
        self.feed = SerialFeed(port)
        self.feed.start()
        self.connection.set('连接中，等待六轴数据…')

    def disconnect(self) -> None:
        self.service.set_connected(False)
        if self.feed:
            self.feed.close()
            self.feed = None
        self.history.clear()
        self.connection.set('设备未连接')

    def record_new(self) -> None:
        self.pending_slot = None
        self.samples.selection_clear(0, 'end')
        self.service.begin_recording(time.monotonic(), 'sample')
        self.status.set('倒计时后做一个完整动作，完成全部阶段后点击结束录制。')

    def record_replace(self) -> None:
        selected = self.samples.curselection()
        if not selected: raise ValueError('请先选择要重录的示范。')
        self.pending_slot = int(selected[0])
        self.service.begin_recording(time.monotonic(), 'sample')
        self.status.set(f'准备重录第 {self.pending_slot + 1} 个示范。')

    def record_background(self) -> None:
        self.pending_slot = None
        kind = self.category_key()
        self.service.begin_recording(time.monotonic(), kind)
        self.status.set(INSTRUCTIONS[kind])

    def category_key(self) -> str:
        return next(k for k in CATEGORIES if LABELS[k] == self.calibration_category.get())

    def restore_draft(self) -> None:
        p = self.service.profile
        if p is None:
            raise ValueError('请先加载需要恢复草稿的手势。')
        try:
            draft = self.store.load_draft(p.id)
        except FileNotFoundError:
            raise ValueError('这个手势尚无保存的草稿。') from None
        self.service.select(draft)
        self.refresh_current()
        self.status.set('草稿已恢复；请检查示范与三类校准后重新学习。')

    def show_report(self, hidden: bool = False) -> None:
        p = self.service.profile
        if p is None:
            raise ValueError('请先新建或加载手势。')
        report = p.report
        states = {'learned': '已学习', 'failed': '学习或校准失败', 'needs_learning': '需要重新学习'}
        lines = [f'手势：{p.name}', f'状态：{states.get(report.get("status", ""), "尚无报告")}', str(report.get('reason', ''))]
        if 'axes' in report:
            lines.append('有效轴：' + '、'.join(AXES[i] for i in report['axes']))
        if 'threshold' in report:
            if report.get('learning_mode') == 'single_accel':
                lines.append('识别方式：单个加速度轴')
            lines.append(f'接受距离 ≤ {report["threshold"]:.3f}（差异阈值，不是概率或厘米）')
        if 'distance' in report:
            lines.append(f'相关非目标片段距离：{report["distance"]:.3f}')
        if 'category' in report:
            lines.append('相关类别：' + LABELS.get(report['category'], report['category']))
        if 'sample' in report:
            lines.append(f'相关示范：第 {report["sample"]} 个；选择该示范检查起止范围。')
        if 'interval' in report:
            lines.append('相关片段：' + '～'.join(f'{v:.2f}' for v in report['interval']) + ' 秒')
        for key, result in report.get('categories', {}).items():
            lines.append(f'{LABELS[key]}：最近距离 {result["distance"]:.3f}；校准事件 {result.get("events", 0)} 个')
        lines.append('\n补录：选择对应非目标类别后重新采集；示范问题可选择列表中的相应示范调整范围或重录。')
        window = tk.Toplevel(self.root)
        if hidden:
            window.withdraw()
        window.title('学习报告')
        view = tk.Text(window, width=74, height=22, wrap='word', font=('Microsoft YaHei UI', 10))
        view.pack(fill='both', expand=True, padx=10, pady=10)
        view.insert('1.0', '\n'.join(lines))
        view.configure(state='disabled')
        def go_to_problem() -> None:
            current = self.service.profile
            if current is None or current.id != p.id:
                raise ValueError('当前手势已切换，请重新查看该手势的报告。')
            sample = report.get('sample')
            category = report.get('category')
            if sample is not None:
                self.samples.selection_clear(0, 'end')
                self.samples.selection_set(sample - 1)
                self.inspect_sample()
                self.sidebar_canvas.yview_moveto(.1)
            elif category in CATEGORIES:
                self.calibration_category.set(LABELS[category])
                self.calibration_instruction.set(INSTRUCTIONS[category])
                self.sidebar_canvas.yview_moveto(.5)
            window.destroy()
        if 'sample' in report or report.get('category') in CATEGORIES:
            ttk.Button(window, text='检查 / 补录相关数据', command=lambda: None if self.busy else self.guard(go_to_problem)).pack(pady=(0, 8))

    def finish(self) -> None:
        self.service.finish_recording()
        self.status.set('录制已结束，请检查完整动作边界后保存。')

    def import_csv(self) -> None:
        self._ensure_idle_recording()
        self.service.stop()
        path = filedialog.askopenfilename(title='选择六轴记录', filetypes=[('CSV', '*.csv')], initialdir=ROOT / 'recordings')
        if not path: return
        self.pending_slot = None
        self.service.import_review(read_csv(Path(path)))
        self.status.set('检查建议范围是否包含完整动作，确认后保存，再选择下一次。')

    def import_background(self) -> None:
        self._ensure_idle_recording()
        self.service.stop()
        path = filedialog.askopenfilename(title='选择明确的非目标记录', filetypes=[('CSV', '*.csv')], initialdir=ROOT / 'recordings')
        if path:
            self.service.set_calibration(self.category_key(), read_csv(Path(path)))
            self.refresh_current()

    def inspect_sample(self) -> None:
        if self.busy or self.service.recorder.state in ('countdown', 'recording'): return
        p, selected = self.service.profile, self.samples.curselection()
        if p is None or not selected: return
        self.service.stop()
        index = int(selected[0])
        clip = p.samples[index]
        self.pending_slot = index
        self.service.review = clip.raw
        self.render(clip.raw, f'Review sample {index + 1}')
        self.bounds_start.set(f'{clip.start:.3f}')
        self.bounds_end.set(f'{clip.end:.3f}')
        self.span.extents = (clip.start, clip.end)
        self.canvas.draw_idle()

    def save_sample(self) -> None:
        self.service.save_review(float(self.bounds_start.get()), float(self.bounds_end.get()), self.pending_slot)
        self.pending_slot = None
        self.refresh_current()
        self.status.set('示范已确认。继续录入或选择下一次完整动作。')

    def select_bounds(self, start: float, end: float) -> None:
        if self.service.review is None or self.service.recorder.state in ('countdown', 'recording'): return
        self.bounds_start.set(f'{max(0., start):.3f}')
        self.bounds_end.set(f'{min(self.service.review.duration, end):.3f}')

    def render(self, motion: Motion, title: str) -> None:
        for i, line in enumerate(self.lines):
            line.set_data(motion.times, motion.values[:, i])
        for axis in self.axes:
            axis.relim()
            axis.autoscale_view(scalex=False)
            axis.set_xlim(0, max(.1, motion.duration))
        self.figure.suptitle(title)
        self.canvas.draw_idle()

    def learn(self, single_accel: bool = False) -> None:
        self._ensure_idle_recording()
        self.busy = True
        self.service.stop()
        self.status.set('正在学习一个加速度轴…' if single_accel else '正在比较示范与非目标窗口…')
        self._job(lambda: self.service.learn_current(use_new=True, single_accel=single_accel))

    def _job(self, operation: Callable[[], Any]) -> None:
        def run() -> None:
            try: operation()
            except Exception as error: self.service.notify('error', str(error))
            finally: self.service.notify('job_done', None)
        threading.Thread(target=run, name='kinebind-job', daemon=True).start()

    def save_binding(self) -> None:
        self._ensure_idle_recording()
        self.service.change_binding(int(self.distance.get()))
        self.refresh_current()
        self.status.set('固定移动距离已保存；控制停用。')

    def start(self) -> None:
        self.service.start_recognition()
        self.event_count = 0
        profile = self.service.profile
        self.status.set(f'识别已开启：{profile.name if profile else ""}。完整动作匹配后触发一次。' + (' 当前为模拟鼠标。' if self.dry_run else ''))
        self.append_log('开始识别；' + ('模拟鼠标' if self.dry_run else '实际鼠标'))

    def stop(self) -> None:
        self.service.stop()
        self.status.set('识别已停止，鼠标控制停用。')
        self.append_log('停止；旧会话的事件已失效')

    def replay(self) -> None:
        self._ensure_idle_recording()
        self.service.stop()
        path = filedialog.askopenfilename(title='选择回放 CSV', filetypes=[('CSV', '*.csv')], initialdir=ROOT / 'recordings')
        if not path: return
        motion = read_csv(Path(path))
        self.disconnect()
        self.service.start_recognition(replay=True)
        self.event_count = 0
        self.busy = True
        self.service.review = None
        self.render(motion, 'CSV replay (mouse disabled)')
        self.status.set('正在回放，不执行实际鼠标。结束后需重新连接设备以实时使用。')
        self._job(lambda: self.service.replay_motion(motion))

    def _handle_results(self) -> None:
        for _ in range(200):
            try: kind, value = self.service.results.get_nowait()
            except queue.Empty: break
            if kind == 'review':
                self.render(value, 'Review complete gesture (select bounds)')
                suggestion = self.service.suggestion
                start, end = suggestion.get('start', 0.), suggestion.get('end', value.duration)
                self.bounds_start.set(f'{start:.3f}')
                self.bounds_end.set(f'{end:.3f}')
                self.span.extents = (start, end)
                self.status.set(suggestion.get('reason', '检查完整动作范围后确认。'))
            elif kind == 'background':
                self.render(value, 'Non-target calibration recording')
                self.refresh_current()
                self.status.set(f'非目标记录已保存到草稿：{value.duration:.1f} 秒。')
            elif kind == 'calibration':
                category, motion, _ = value
                self.render(motion, f'Calibration: {LABELS[category]}')
                self.refresh_current()
                self.status.set(f'{LABELS[category]}已保存：{motion.duration:.1f} 秒；继续补齐三类。')
            elif kind == 'report':
                self.refresh_current()
                self.status.set(str(value.get('reason', '学习报告已更新。')))
            elif kind == 'event':
                event, outcome = value
                self.event_count += 1
                self.append_log(f'{event.start:.2f}～{event.end:.2f}s  距离 {event.score:.3f}  {outcome}')
                self.status.set(outcome)
            elif kind == 'rejected':
                session, at, match = value
                engine = self.service.recognizer
                if engine is not None and engine.session == session:
                    self.append_log(f'{at:.2f}s  拒绝 · 距离 {match.score:.3f} · {match.reason}')
            elif kind == 'error':
                self.service.stop()
                self.status.set(str(value))
                self.append_log('提示：' + str(value))
            elif kind == 'test_recording':
                self.test_data_state.set(f'正在保存测试数据：{value}')
                self.append_log(f'测试数据保存开始：{value}')
            elif kind == 'test_saved':
                path, count = value
                self.test_data_state.set(f'测试数据已保存：{count} 条\n{path}')
                self.append_log(f'测试数据已保存：{count} 条 · {path}')
            elif kind in ('progress', 'saved'):
                self.status.set(str(value))
            elif kind == 'learned':
                self.refresh_current()
                self.status.set(str(value))
                self.append_log(str(value))
            elif kind == 'replay_done':
                self.service.stop()
                self.status.set(str(value))
                self.append_log(str(value))
            elif kind == 'job_done':
                self.busy = False

    def _poll(self) -> None:
        if self.closed: return
        now = time.monotonic()
        if self.feed:
            feed = self.feed
            for _ in range(250):
                try: kind, value = feed.messages.get_nowait()
                except queue.Empty: break
                if kind == 'connected':
                    self.service.set_ranges(*value)
                    self.service.set_connected(True)
                    self.connection.set(f'{feed.port} · 已收到六轴数据')
                elif kind == 'point':
                    if now - value.received > .5:
                        if self.service.active or self.service.recorder.state in ('countdown', 'recording'):
                            self.service.set_connected(False)
                            self.service.notify('error', '数据延迟过大，请断开后重新连接。')
                            feed.close()
                            self.feed = None
                            self.history.clear()
                            self.connection.set('数据延迟，连接已关闭')
                            break
                        continue
                    self.history.append(value)
                    if self.service.connected:
                        try: self.service.on_sample(value.t, value.values, now, sample=value.sample)
                        except Exception as error:
                            self.service.set_connected(False)
                            self.service.notify('error', str(error))
                            feed.close()
                            self.feed = None
                            self.history.clear()
                            self.connection.set('数据异常，连接已关闭')
                            break
                elif kind == 'error':
                    self.service.set_connected(False)
                    self.service.notify('error', str(value))
                    feed.close()
                    self.feed = None
                    self.history.clear()
                    self.connection.set('连接失败，请检查后重连')
                    break
                elif kind == 'closed':
                    self.service.set_connected(False)
                    self.connection.set('连接已关闭，请重新连接')
                    self.feed = None
                    self.history.clear()
                    break
        self._handle_results()
        mode = ('回放中' if not self.service.enabled else ('模拟鼠标' if self.dry_run else '鼠标控制开启')) if self.service.active else '控制停用'
        self.control_state.set(f'{mode} · 本次事件 {self.event_count}')
        state = self.service.recorder.state
        if state == 'countdown':
            remaining = max(0, self.service.recorder.deadline - now)
            self.status.set(f'准备：{remaining:.1f} 秒；倒计时后开始动作。')
        elif state == 'recording':
            elapsed = (self.service.recorder.times[-1] - self.service.recorder.times[0]) if self.service.recorder.count > 1 else 0
            kind = self.service.recorder.kind
            suffix = f' {LABELS[kind]}约 10 秒自动结束。' if kind in CATEGORIES else (' 非目标约 30 秒自动结束。' if kind == 'background' else ' 完整动作结束后点击结束录制。')
            self.status.set(f'录制中：{elapsed:.1f} 秒。' + suffix)
        if now - self.last_draw > .2 and (state == 'recording' or (self.service.active and not self.busy)):
            if len(self.history) >= 2:
                points = list(self.history)
                self.render(Motion([p.t for p in points], [p.values for p in points]), 'Live IMU (recent samples)')
            self.last_draw = now
        for button in self.controls:
            button.configure(state='disabled' if self.busy else 'normal')
        if self.busy:
            self.profile_box.configure(state='disabled')
        else:
            self.profile_box.configure(state='readonly')
        self.poll_id = self.root.after(30, self._poll)

    def close(self) -> None:
        self.closed = True
        self.service.stop()
        self.disconnect()
        if self.poll_id:
            self.root.after_cancel(self.poll_id)
        self.root.destroy()


def smoke_test() -> None:
    """Exercise our own window withdrawn; only synthetic data and FakeMouse."""
    from kinebind.synthetic import background, gesture, join, calibration_records
    from kinebind.robust import RobustModel
    with tempfile.TemporaryDirectory() as folder:
        root = tk.Tk()
        root.withdraw()
        mouse = FakeMouse()
        app = GestureApp(root, ProfileStore(Path(folder)), mouse, dry_run=True, smoke=True)
        app.new_profile()
        app.service.set_connected(True)
        for duration, amplitude in [(1., 1.), (.9, .94), (1.1, 1.04), (.95, .98), (1.05, 1.02)]:
            app.pending_slot = None
            app.service.begin_recording(0)
            m = gesture(duration, amplitude)
            for t, row in zip(m.times, m.values): app.service.on_sample(float(t), row, 3 + float(t))
            app.finish()
            app._handle_results()
            app.save_sample()
        for category, motion in calibration_records().items():
            app.service.set_calibration(category, motion)
        app.service.learn_current(use_new=True)
        app._handle_results()
        app.show_report(hidden=True)
        app.start()
        series = join([background(.3), gesture(.85), background(.3), gesture(1.2), background(.3)])
        for i, (t, row) in enumerate(zip(series.times, series.values)):
            app.service.on_sample(float(t), row, float(t), sample=Sample(round(float(t) * 1000), i, tuple(row)))
        app._handle_results()
        assert mouse.requests == [100, 100], mouse.requests
        app.stop()
        app.refresh_profiles()
        app.profile_choice.set(next(iter(app.saved_profiles)))
        app.load_profile()
        assert not app.service.enabled
        app.learn(single_accel=True)
        deadline = time.monotonic() + 10.
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(.01)
        assert not app.busy, 'single acceleration learning did not finish'
        single_profile = app.service.profile
        assert single_profile is not None and isinstance(single_profile.model, RobustModel)
        assert single_profile.report.get('learning_mode') == 'single_accel'
        assert len(single_profile.model.axes) == 1 and single_profile.model.axes[0] < 3
        app.show_report(hidden=True)
        app.start()
        for i, (t, row) in enumerate(zip(series.times, series.values)):
            app.service.on_sample(float(t), row, float(t), sample=Sample(round(float(t) * 1000), i, tuple(row)))
        app._handle_results()
        assert mouse.requests == [100, 100, 100, 100], mouse.requests
        app.stop()
        app.load_profile()
        assert not app.service.enabled
        csv_paths = list((Path(folder) / 'recordings').glob('*.csv'))
        assert len(csv_paths) == 2, csv_paths
        for path in csv_paths:
            recorded = read_csv(path)
            assert len(recorded.times) == len(series.times)
            np.testing.assert_allclose(recorded.values, series.values)
        app._handle_results()
        assert '测试数据已保存' in app.test_data_state.get()
        app.render(gesture(), 'Synthetic reviewed gesture — GUI smoke test')
        root.update_idletasks()
        (ROOT / 'build').mkdir(exist_ok=True)
        app.figure.savefig(ROOT / 'build' / 'gesture_gui_smoke.png', dpi=120)
        app.start()
        for i, (t, row) in enumerate(zip(series.times[:11], series.values[:11])):
            app.service.on_sample(float(t), row, float(t), sample=Sample(round(float(t) * 1000), i, tuple(row)))
        app.close()
        csv_paths = list((Path(folder) / 'recordings').glob('*.csv'))
        assert len(csv_paths) == 3
        assert sorted(len(read_csv(path).times) for path in csv_paths) == [11, len(series.times), len(series.times)]
        print('GUI smoke passed: 4 fake movements, stop disabled, realtime CSV saved on stop and window close.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'gestures')
    parser.add_argument('--port', default='COM7')
    parser.add_argument('--dry-run', action='store_true', help='Use fake mouse with real sensor data.')
    parser.add_argument('--replay', type=Path, help='Headless replay; never moves actual mouse.')
    parser.add_argument('--profile', help='Saved profile ID to use in headless replay.')
    parser.add_argument('--smoke-test', action='store_true', help='Withdrawn GUI with synthetic data and fake mouse.')
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return 0
    store = ProfileStore(args.data_dir)
    if args.replay:
        profiles, errors = store.list_profiles()
        candidates = [p for p in profiles if p.model is not None and (args.profile is None or p.id == args.profile)]
        if len(candidates) != 1:
            parser.error('需要唯一可用手势；用 --profile 指定 ID。' + ('；'.join(errors) if errors else ''))
        controller = GestureController(store, FakeMouse(), threaded=False)
        controller.select(candidates[0])
        controller.start_recognition(replay=True)
        controller.replay_motion(read_csv(args.replay))
        events = []
        while not controller.results.empty():
            kind, value = controller.results.get_nowait()
            if kind == 'event':
                event, _ = value
                events.append(dict(start=event.start, end=event.end, score=event.score))
        controller.stop()
        print(json.dumps(dict(profile=candidates[0].name, count=len(events), mouse_executed=False, events=events), ensure_ascii=False, indent=2))
        return 0
    # Use physical screen coordinates consistently before constructing any windows.
    try: ctypes.WinDLL('user32').SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError): pass
    root = tk.Tk()
    GestureApp(root, store, FakeMouse() if args.dry_run else WindowsMouse(), args.port, args.dry_run)
    root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

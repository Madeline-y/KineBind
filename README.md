# KineBind：个人手势学习与鼠标控制

查看 [更新记录](UPDATE.md)。

目标：XIAO nRF52840 Sense → USB → Python → 实时六轴曲线与 CSV。

当前手势“左挥”（`left_swipe`）定义为完整的“向左挥动 → 向右回收”组合，
一次往返算一次手势，模板包含回收阶段。详细范围和确认记录见 [手势定义](docs/gesture_definitions.md)。

电脑端已支持五个示范、自动建议范围并人工确认、静止/普通晃动/拿放三类校准、自动有效轴学习、学习报告、回放与显式启停鼠标控制。旧手势仍按旧规则加载，采用新规则需要补齐三类校准后重新学习。

使用步骤见 [手势学习与验收说明](docs/gesture_learning_guide.md)，改动依据见 [改进规格](docs/superpowers/specs/2026-10-07-gesture-learning-robustness-design.md)。自动测试和合成数据验证不代表真实识别率已经达标。

```powershell
conda activate kinebind
python gesture_app.py --dry-run
```

`--dry-run` 接收真实设备数据并模拟鼠标执行，适合先检验识别。需要实际控制时运行 `python gesture_app.py`，点击“开始识别”后才执行鼠标移动。顶栏的停止按钮也会取消当前学习。

## 1. 上传采集固件

在 Arduino IDE 中打开 `firmware/imu_stream/imu_stream.ino`。

- 支持包：Seeed nRF52 mbed-enabled Boards，已验证的本机版本为 2.9.3。
- 开发板：XIAO nRF52840 Sense（界面可能显示 Seeed 前缀或 No Updates 后缀）。
- 传感器库：Seeed Arduino LSM6DS3，已验证的本机版本为 2.0.7。
- 端口：COM7；重新插拔或上传后如编号变化，请重新选择实际端口。
- 点击“验证”，成功后点击“上传”。新固件会替换之前的示例程序。

固件关闭三色用户 LED，传感器内部刷新频率设为 104 Hz，USB 输出目标频率为 50 Hz。
加速度量程 ±4 g，角速度量程 ±500 deg/s。加速度包含重力，没有做校准或滤波。
这不是硬实时采样：实际频率可从设备时间戳和曲线标题检查；错过的调度周期不会补发。

输出协议（115200 波特率设置）：

```text
timestamp_ms,sequence,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps
1020,0,0.5524,-0.7359,0.4265,1.5400,-3.3600,0.4900
1040,1,0.5529,-0.7349,0.4158,1.4700,-2.8700,0.4900
```

`timestamp_ms` 是设备启动后的毫秒数；`sequence` 是发送记录的序号。
旧 HighLevelExample 的多行文字输出不适用于此采集器；脚本会提示先上传新固件。

## 2. 运行电脑端

先退出 miniterm（Ctrl+]），关闭 Arduino 串口监视器和串口绘图器，保持 USB 连接。
在 PowerShell / Anaconda Prompt 中执行：

```powershell
conda activate kinebind
cd D:\15366\KineBind
python capture_imu.py --port COM7
```

上图显示加速度，下图显示角速度，默认展示最近 10 秒。
所有接受的六轴记录自动写入项目的 `recordings/imu_日期_时间.csv`。
关闭曲线窗口或在终端按 Ctrl+C 停止；串口关闭，CSV 刷新并保存。

CSV 额外包含 `received_at`（电脑接收时间，UTC+08:00）和 `elapsed_s`
（设备时间相对首条记录的秒数）。记录保留原始六轴数据；绘图慢时仅丢弃视觉更新。
序号跳跃会统计；设备重启或时间异常会终止本次采集，防止混合不同会话。
CSV 每秒刷新，正常退出时完整刷新；强制结束进程可能损失尚未刷新的尾部数据。

其他用法：

```powershell
# 只保存，不打开图，采集 30 秒。
python capture_imu.py --port COM7 --no-plot --duration 30

# 无需连接设备，使用明确标为 simulated 的合成数据检查界面。
python capture_imu.py --simulate --duration 10

# 指定新文件名，已有文件不会被覆盖。
python capture_imu.py --output recordings\first_motion.csv
```

现有 kinebind 环境可直接使用。重建环境时可运行 `conda env create -f environment.yml`；
已有同名环境无需再次创建。VS Code 的 Python 解释器选择 kinebind。

## 3. 常见问题

- **拒绝访问 / Access denied**：退出 miniterm，关闭 IDE 的串口监视器/绘图器。
- **找不到 COM7**：确认数据线和连接，运行 `python -m serial.tools.list_ports`。
- **仍是 Accelerometer / Gyroscope 分行输出**：上传项目里的 imu_stream 固件。
- **10 秒没有有效数据**：检查实际端口、固件上传结果与 USB 连接。
- **IMU initialization failed**：确认 Sense 型号、目标板型和传感器库。
- **Missing FQBN**：在新打开的固件窗口重新选择开发板，确认支持包已安装。
- **无图形界面**：用 `--no-plot` 采集；模拟模式也可以验证 CSV。

## 4. 软件检查

```powershell
python -m unittest discover -s tests -v
python capture_imu.py --simulate --no-plot --duration 2
```

自动检查覆盖协议、分片、时间戳回绕、设备重启、CSV、文件保护和旧固件提示。
模拟检查不代表真实硬件采样成功；实际 50 Hz 数据必须在上传固件后确认。

2026-10-06 验证结果：12 项自动测试通过；两秒模拟 CSV 保存 100 条记录，设备时间计算为
50 Hz；绘图渲染及 Windows TkAgg 事件循环检查通过。Sense 固件编译通过，程序占用
97,848 字节（12%），全局 RAM 占用 46,000 字节（19%）。尚未上传新固件或验证实板 50 Hz 输出。

参考：[Seeed 入门](https://wiki.seeedstudio.com/XIAO_BLE/)、
[Seeed IMU 使用](https://wiki.seeedstudio.com/XIAO-BLE-Sense-IMU-Usage/)、
[pySerial](https://pyserial.readthedocs.io/en/latest/shortintro.html)。

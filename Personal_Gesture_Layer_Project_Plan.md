# Personal Gesture Layer
## 可训练的个人手势控制器 —— V0 到产品化路线

> **核心理念：不是让用户学习设备定义好的手势，而是让设备学习用户自己的手势。**  
> 将人的动作变成一种可训练、可迁移、可跨设备复用的输入接口。

---

## 1. 项目一句话定义

做一个极小型可穿戴动作输入设备：

**用户录入自己的手势 → 系统学习该动作 → 以后再次做出该动作 → 设备识别 → 无线触发电脑/手机/机器人/智能家居等命令。**

核心抽象不是：

```text
Gesture → Key
```

而是逐渐演进为：

```text
Gesture → Intent → Context → Action
```

例如用户自己定义一个“向右轻甩”的动作，它代表的是语义 `NEXT`：

| 当前环境 | `NEXT` 实际执行 |
|---|---|
| PowerPoint | 下一页 |
| Chrome | 下一个标签页 |
| Spotify | 下一首 |
| PDF 阅读器 | 下一页 |
| VS Code | 下一个 Editor |
| 相机 | 下一种模式 |
| 机器人 | 下一个动作 |
| MIDI 软件 | 下一个 Pattern |

---

# 2. 第一版硬件选型

## Seeed Studio XIAO nRF52840 Sense

当前第一版确定使用：

**XIAO nRF52840 Sense**

不要求焊接排针，**未焊版即可**。

对于 V0 来说，只需要：

```text
XIAO nRF52840 Sense × 1
USB-C 数据线 × 1
电脑 × 1
```

甚至暂时**不需要电池、不需要外部 IMU、不需要屏幕、不需要任何机械结构**。

---

## 3. 为什么选择 XIAO nRF52840 Sense

这块板的优势不是“性能最强”，而是它把当前项目需要的关键能力集中在了约 **21 × 17.8 mm** 的板子里。

### 核心硬件

| 模块 | 能力 | 在项目中的作用 |
|---|---|---|
| Nordic nRF52840 | 64 MHz Cortex-M4F | 数据处理、手势识别、轻量 TinyML 推理 |
| 256 KB RAM | MCU 内存 | 保存传感器窗口、运行小模型 |
| 1 MB MCU Flash + 板载 Flash | 程序和数据 | 保存固件、模型等 |
| LSM6DS3TR-C | 6 轴 IMU | 获取手势运动数据 |
| Bluetooth Low Energy | 低功耗无线 | 无线发送命令 |
| PDM 麦克风 | 音频输入 | 后续做 Gesture + Voice |
| BQ25101 | 电池充电管理 | 后续直接接小型 LiPo |
| USB | 烧录/串口/供电 | 第一版开发与数据采集 |

### 六轴 IMU 能获取什么

板载 IMU 输出：

```text
ax  ay  az
gx  gy  gz
```

其中：

- `ax ay az`：三轴加速度
- `gx gy gz`：三轴角速度

例如：

```text
右甩
左甩
翻腕
画圈
向前戳
双击
快速旋转
```

这些动作都会产生不同的六维时间序列。

一个动作可以表示为：

```text
时间     ax    ay    az    gx    gy    gz
0ms      ...
20ms     ...
40ms     ...
60ms     ...
...
```

数学上可以写成：

\[
X \in R^{T \times 6}
\]

例如：

```text
50 Hz × 1 秒 × 6 轴
=
50 × 6
```

---

# 4. 第一版真正需要验证的核心假设

V0 不追求：

- 漂亮外壳
- 戒指形态
- 超长续航
- 大模型
- 手机 App
- 很多手势
- 云端服务

V0 只验证一件事：

> **“一个普通用户能不能快速教会设备一个属于自己的动作，并且之后稳定识别出来？”**

这是整个产品最重要的核心。

---

# 5. V0 系统架构

第一阶段建议使用 USB，而不是一开始就折腾蓝牙。

```text
你的手
  ↓
XIAO 板载 IMU
  ↓
ax ay az gx gy gz
  ↓
USB Serial
  ↓
Windows / Python
  ↓
采集个人动作样本
  ↓
建立手势模型
  ↓
再次做动作
  ↓
识别 Gesture ID
  ↓
电脑执行命令
```

第一阶段可以做到：

```text
右甩     → Ctrl + Tab
左甩     → Ctrl + Shift + Tab
画圈     → Win + Shift + S
翻腕     → Play / Pause
向前戳   → Enter
```

---

# 6. 第一版不要急着用深度学习

最重要的是先验证交互体验。

## V0.1：模板匹配 / DTW / kNN

用户创建一个新手势：

```text
手势名称：Screenshot

请重复动作：

Sample 1 ✓
Sample 2 ✓
Sample 3 ✓
Sample 4 ✓
Sample 5 ✓
```

然后保存这些动作的 IMU 序列。

新动作到来后，与用户录制的模板比较：

\[
D(X,A_1), D(X,A_2), ..., D(X,A_5)
\]

如果距离足够小：

```text
Gesture = Screenshot
Confidence = 0.93
```

立即触发命令。

### 这个阶段的优势

用户不需要：

```text
重新编译模型
重新量化
重新烧录固件
```

而是：

```text
录动作
↓
几秒后
↓
直接使用
```

这最符合：

> **Teach it a gesture. Bind it to anything.**

---

# 7. 第二阶段：TinyML 本地识别

当 V0 验证成功，再升级为：

```text
IMU
 ↓
Motion Detection
 ↓
固定长度窗口
 ↓
归一化
 ↓
1D CNN
 ↓
INT8 Quantization
 ↓
XIAO nRF52840 Sense
 ↓
Gesture ID
```

可能的模型结构：

```text
Input: 50 × 6

 ↓

Conv1D
32 channels

 ↓

Conv1D
64 channels

 ↓

Global Average Pooling

 ↓

Dense

 ↓

N Gestures
```

目标：

```text
整个识别过程
完全运行在手腕上的 MCU 中
```

电脑不负责识别，只负责收到结果后执行动作。

---

# 8. BLE 在产品中的作用

BLE：

**Bluetooth Low Energy，低功耗蓝牙。**

最终无线版：

```text
你的手势
   ↓
IMU
   ↓
本地识别
   ↓
Gesture ID
   ↓
BLE
   )))
Windows / 手机 / 平板
```

更进一步可以使用：

```text
BLE HID
```

让设备直接表现为：

```text
蓝牙键盘
蓝牙鼠标
蓝牙遥控器
```

例如设备自己发送：

```text
Ctrl + Tab
Volume Up
Volume Down
Play / Pause
Page Down
Win + Shift + S
```

因此最终用户不一定需要安装复杂驱动。

---

# 9. 产品真正的差异化

现有很多遥控戒指的思路是：

```text
厂商预定义 Gesture
         ↓
用户学习 Gesture
         ↓
绑定 Command
```

我们希望反过来：

```text
用户创造 Gesture
        ↓
设备学习 Gesture
        ↓
用户绑定 Intent
```

核心差异：

## Customize Commands

并不稀缺。

更重要的是：

# Customize Gestures

也就是：

> **不是 Customize Commands，而是 Customize Gestures。**

---

# 10. 从“手势快捷键”升级到“语义意图”

不要把：

```text
右甩
```

永远绑定：

```text
Ctrl + Tab
```

而应该定义：

```text
右甩 → NEXT
```

然后 Context Engine 根据当前环境决定：

```text
NEXT

├── Chrome       → Next Tab
├── PowerPoint   → Next Slide
├── Spotify      → Next Track
├── VS Code      → Next Editor
├── PDF          → Next Page
├── Robot        → Next Action
└── Camera       → Next Mode
```

最终产品核心：

\[
Gesture \rightarrow Intent \rightarrow Context \rightarrow Action
\]

---

# 11. 产品形态：不要把自己限制在“戒指”

真正值得做的是：

# Motion Core

一个约 20 mm 级的小型动作感知核心：

```text
┌──────────────────┐
│ Motion Core      │
│                  │
│ MCU              │
│ IMU              │
│ BLE              │
│ Battery          │
└──────────────────┘
```

设计磁吸/卡扣结构：

```text
                    Motion Core
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
        手腕夹          手指夹           笔夹
          │              │              │
        Wrist          Finger          Pen
```

同一颗核心可以迁移到完全不同的产品。

---

# 12. 可以迁移到哪些方向

## 12.1 手腕 —— Gesture Stream Deck

```text
右甩      → Next
左甩      → Previous
翻腕      → Cancel
向前戳    → Select
顺时针    → More
逆时针    → Less
```

最终替代一部分：

```text
快捷键
按钮
遥控器
Stream Deck
```

---

## 12.2 笔 —— AI Magic Pen

把 Motion Core 塞进笔中：

```text
画 ○  → Screenshot
画 Z  → Undo
画 ✓  → Confirm
```

继续发展可以做：

# Air Writing Recognition

即：

```text
空气写字
↓
IMU 时间序列
↓
识别字母 / 符号 / 命令
```

---

## 12.3 魔杖 —— 智能家居

```text
○      → 关闭灯
↑      → 打开灯
旋转   → 调亮度
向下   → 关闭设备
```

架构：

```text
Gesture
 ↓
Motion Core
 ↓
BLE
 ↓
Home Assistant / Gateway
 ↓
灯 / 空调 / 窗帘
```

---

## 12.4 音乐 —— Wearable MIDI Controller

动作不仅可以是离散命令，也可以变成连续参数：

```text
手抬高程度
      ↓
Pitch

手腕旋转速度
      ↓
Filter Cutoff

向下猛甩
      ↓
Drum Hit

画圈
      ↓
Loop
```

甚至：

```text
左手一个 Motion Core
右手一个 Motion Core
       ↓
人体成为 MIDI Controller
```

---

## 12.5 机器人控制

可以和另一个低成本项目：

**AI 自动追踪云台**

直接结合。

```text
手腕 Motion Core
       ↓
Gesture
       ↓
Intent
       ↓
机器人
```

例如：

```text
向左挥 → LOOK_LEFT
向右挥 → LOOK_RIGHT
向前推 → FOLLOW
翻腕   → STOP
```

机器人端：

```text
FOLLOW
 ↓
RT-DETR / RF-DETR
 ↓
锁定目标
 ↓
Tracking
 ↓
Pan/Tilt
```

形成：

# Human → Robot Interface

---

## 12.6 相机 / Vlog

```text
翻腕     → Start Recording
再次翻腕 → Stop Recording
右甩     → Switch Camera
画圈     → Enable Tracking
```

---

## 12.7 CAD / Blender / 3D 软件

动作可以映射成连续空间控制：

```text
手腕旋转
 ↓
Rotate Object

Pitch
 ↓
Zoom

Yaw
 ↓
Pan
```

注意：六轴 IMU 不适合长期精确位置追踪，但很适合短时间姿态与动作控制。

---

## 12.8 游戏

游戏不再规定：

```text
按 X → Fireball
```

而是：

```text
第一次：

“请设计你的 Fireball 动作”

重复 5 次
↓
建立个人动作模型
```

以后：

```text
你的专属动作
↓
Fireball
```

可以形成一个很有意思的概念：

> **每个人拥有自己的“魔法语言”。**

---

## 12.9 无障碍交互

传统输入设备要求：

```text
用户适应设备
```

这个项目可以反过来：

```text
设备学习用户还能稳定完成的动作
```

例如：

```text
轻微翻腕 → YES
轻敲     → NO
快速抬腕 → HELP
```

这可能成为比“酷炫手势”更有实际价值的方向。

---

# 13. 连续控制比单纯命令更重要

第一阶段：

\[
Gesture \rightarrow Command
\]

未来：

\[
Motion \rightarrow Continuous\ Control
\]

例如：

```text
慢速顺时针
→ 音量 +1

快速顺时针
→ 音量 +10
```

或者：

```text
手腕旋转 0°  ─────────── 180°
灯光亮度   0%             100%
```

IMU 本身还可以提供：

- 动作速度
- 动作方向
- 动作幅度
- 持续时间
- 角速度
- 姿态变化

这些都可以成为控制变量。

---

# 14. AI Agent 版本

最终不应该让一个手势对应一个键盘按键。

而是：

```text
Gesture
 ↓
Intent
 ↓
Agent
 ↓
理解当前上下文
 ↓
执行最合理的动作
```

例如：

```text
Intent = CAPTURE_THIS
```

Agent 根据环境决定：

```text
浏览器       → Screenshot Page
PDF          → Save Current Page
摄像头       → Take Photo
Blender      → Render Viewport
```

最终可以定义类似：

```python
gesture.on("circle", intent="CAPTURE_THIS")
gesture.on("swipe_right", intent="NEXT")
gesture.on("flip", intent="CANCEL")
```

而不是：

```python
gesture.on("circle", press_key("F5"))
```

---

# 15. 更长期的产品定义

项目名称可以暂时叫：

# Personal Gesture Layer

或：

# Motion Intent Interface

其真正的核心不是某一块开发板，而是：

```text
                    Personal Gesture Layer

                              │
             ┌────────────────┼────────────────┐
             ↓                ↓                ↓

       Gesture Model     Intent Engine     Device Adapter
             │                │                │
          你怎么动          你想干什么        设备怎么执行
```

最终真正可积累的资产可能包括：

- Personal Gesture Model
- Gesture Training UX
- Motion Segmentation
- Intent Protocol
- Context Engine
- Windows / macOS / Mobile Companion
- SDK
- Plugin Ecosystem
- Device Adapters

硬件只是入口。

---

# 16. 开发路线图

## V0 —— 数据跑通

**目标：看到自己的动作数据。**

```text
XIAO
 ↓ USB
电脑
 ↓
实时显示：
ax ay az gx gy gz
```

完成标准：

- 能稳定读取六轴 IMU
- 能保存 CSV
- 能画出动作曲线

---

## V0.1 —— 教它一个动作

**目标：验证“产品学习用户”。**

流程：

```text
点击：新建手势
 ↓
输入名称
 ↓
录制 5~10 次
 ↓
建立模板
 ↓
再做一次
 ↓
识别
```

第一阶段算法：

```text
DTW / kNN / Template Matching
```

完成标准：

> 用户可以在 1 分钟内创建一个新手势。

---

## V0.2 —— 4~6 个自定义手势

建议先：

```text
NEXT
BACK
SELECT
CANCEL
MORE
LESS
```

重点测试：

- 不同动作是否容易混淆
- 静止时是否误触发
- 同一个动作连续做是否稳定
- 不同速度是否还能识别

---

## V1 —— BLE 无线控制

加入：

```text
Gesture
 ↓
BLE
 ↓
Windows
```

可以先发送 Gesture ID。

之后升级 BLE HID。

完成标准：

> 拔掉 USB 后，戴着设备也可以控制电脑。

---

## V2 —— TinyML

训练：

```text
用户动作数据
 ↓
1D CNN
 ↓
Quantization
 ↓
INT8
 ↓
MCU
```

完成标准：

> 不依赖电脑进行识别。

---

## V3 —— 电池 + 可穿戴外壳

加入：

```text
100~300mAh LiPo
+
3D Printed Shell
+
腕带
```

第一版建议：

```text
手腕
```

不要直接做戒指。

原因：

- 手腕动作幅度更大
- IMU 信噪比更好
- 电池空间充足
- 调试方便

未来：

```text
Wrist
 ↓
Hand
 ↓
Finger
 ↓
Ring
```

---

## V4 —— Context Engine

PC Companion 获取：

```text
当前前台软件
```

然后：

```text
Gesture = NEXT
```

动态翻译为：

```text
Chrome      → Ctrl + Tab
PowerPoint  → PageDown
Spotify     → Next Track
VS Code     → Next Editor
```

---

## V5 —— Motion Core

设计自己的小型 PCB / 封装：

```text
MCU
+
IMU
+
BLE
+
Battery
+
Haptic
```

提供：

```text
Wrist Mount
Finger Mount
Pen Mount
Tool Mount
```

---

## V6 —— SDK / Gesture Ecosystem

让第三方可以定义：

```text
Intent
Action
Context
Device Adapter
```

目标：

```text
PC
Phone
Music
Home Assistant
Camera
Robot
Game
CAD
```

全部可以接入同一个 Gesture Layer。

---

# 17. 第一阶段明确不要做的东西

为了避免项目失控，V0 暂时不做：

- 自己画 PCB
- 戒指
- 手机 App
- 云服务器
- 大模型
- 语音识别
- 复杂外壳
- 20 个手势
- 多设备同时控制
- 用户身份认证

只做：

```text
一个 XIAO
+
USB
+
Python
+
个人动作识别
```

---

# 18. 下一步执行清单

## Step 1：购买

```text
Seeed Studio XIAO nRF52840 Sense × 1
```

推荐：

- Sense 版本
- 未焊排针版即可
- 第一阶段不用买电池

---

## Step 2：搭建开发环境

可选：

```text
Arduino IDE
或
PlatformIO + VS Code
```

Python：

```text
Python
numpy
pandas
pyserial
matplotlib
scipy
scikit-learn
```

---

## Step 3：读取 IMU

目标：

```text
Serial Monitor:

ax=...
ay=...
az=...
gx=...
gy=...
gz=...
```

---

## Step 4：实时画图

Python：

```text
XIAO
 ↓ Serial
Python
 ↓
Matplotlib
```

看到手势对应的真实曲线。

---

## Step 5：数据采集器

做一个最简单的界面：

```text
Gesture Name:
[ Screenshot ]

[ Start Recording ]

Samples:
✓ ✓ ✓ ✓ ✓
```

保存：

```text
dataset/
├── screenshot/
│   ├── 001.csv
│   ├── 002.csv
│   └── ...
├── next/
└── back/
```

---

## Step 6：第一个识别算法

先实现：

```text
DTW
+
Nearest Template
```

目标：

> 教它 5 次，再做第 6 次，可以正确判断。

---

## Step 7：执行真正的 Windows 命令

例如：

```text
Gesture = SCREENSHOT
 ↓
Win + Shift + S
```

做到这里，项目第一次形成真正完整闭环：

```text
你的动作
 ↓
真实传感器
 ↓
算法理解
 ↓
数字世界发生变化
```

---

# 19. 第一个 Milestone

不要把目标定成：

> “我要做 AI 手势戒指。”

第一个 Milestone 应该只有一句话：

# 教它一个以前从来没见过的动作，5 次之后，它能认出第 6 次。

如果这一件事情体验足够顺畅：

```text
Teach
 ↓
Learn
 ↓
Recognize
 ↓
Act
```

这个项目就值得继续。

---

# 20. 当前项目状态

```text
产品方向      ✅ 已确定
核心理念      ✅ 已确定
V0 硬件       ✅ 已确定
IMU           ✅ 板载
BLE           ✅ 板载
充电管理      ✅ 板载
开发路径      ✅ 已确定

现在需要做：

购买 XIAO nRF52840 Sense
       ↓
读取 IMU
       ↓
开始采第一批个人动作数据
```

---

# 参考资料

1. Seeed Studio — XIAO nRF52840 Series 官方 Wiki  
   https://wiki.seeedstudio.com/XIAO_BLE/

2. Seeed Studio — XIAO nRF52840 系列中文 Wiki  
   https://wiki.seeedstudio.com/cn/XIAO_BLE/

3. Nordic Semiconductor — nRF52840 官方产品页  
   https://www.nordicsemi.com/products/nrf52840

4. Bluetooth SIG — HID over GATT Profile  
   https://www.bluetooth.com/specifications/specs/hid-over-gatt-profile-hogp/

---

## 最后一句

这个项目真正有价值的地方不是：

> “挥一下手控制 PPT。”

而是：

> **让每个人都能够创造属于自己的动作语言，并让这套动作语言跨设备迁移。**

硬件只是第一层。

真正的长期目标是：

# Personal Gesture Layer

**让人的动作成为一种可编程接口。**

# Save Realtime Test Data Implementation Plan

> **For agentic workers:** 用户已授权当前会话只新增实时测试保存。引用的 superpowers 执行技能未安装，本会话直接完成，不派生代理；无 Git 仓库。

**Goal:** 从开始实时识别到停止，完整保存实际输入数据，方便定位漏识别。

**Architecture:** 新建流式 CSV 写入器，控制器可选配置测试目录；窗口启用自动保存。串口 Point 携带原始 Sample，保存真实设备时间戳和序号，全部有效输入在进入识别前写入；停止统一关闭文件，不依赖最近 600 条绘图缓存。

**Tech Stack:** Python、csv、Tkinter、unittest。

**Spec:** 用户要求“现在只新加一个功能，就是可以保存实时测试的数据”。

## Constraints

只改保存数据及其 CSV 回读支持，不改学习、阈值、有效轴或鼠标判定。每个实时会话一个新文件，不覆盖旧数据，不把回放/训练数据混入。原始六轴和时间戳、序号保留，用于分析；单轴识别规则不变。断开、停止、切换及关闭由现有 stop 边界完成写入。

## Task 1: 完整实时数据保存

Files: 新建 kinebind/live_recording.py、tests/test_live_recording.py；修改 kinebind/sources.py、controller.py、gesture_app.py；更新 docs/gesture_learning_guide.md。

- [x] 编写行为测试：`start_recognition()` 后传入带 `Sample` 的 `on_sample`，停止后 CSV 逐行保持时间戳、序号、六轴；超过 600 条仍完整；CSV 可由 `read_csv` 回读量程与时间。
- [x] 验证重复开始生成两个文件、断开关闭写入、停止后的样本不追加、回放不创建文件、保存失败停止控制。验证 SerialFeed 保留原始 Sample。
- [x] 实现 `LiveTestRecorder.start(profile)`、`write(t, sample)`、`close()`，exclusive 创建文件，周期 flush、停止 close；固定上海时间命名，文件包含相对测试时间、设备时间戳、序号、原始六轴、量程和手势标识。
- [x] 控制器新增可选 `test_folder: Path | None`，窗口配置项目 recordings；`on_sample(..., sample: Sample | None=None)` 在实时 active 时保存后再识别。stop 关闭且通知保存路径/条数。start/replay 仍沿现有权限与执行门。
- [x] 窗口显示自动保存说明及本次路径；串口 Point 增加可选原始 Sample，接收路径透传；隐藏检查使用临时目录与合成 Sample。
- [x] 运行单文件、完整测试、类型检查与 GUI smoke，复查关闭/异常分支，更新操作说明。无 Git 分支可提交。

Self-review: CSV 从识别输入边界写入，记录不限绘图缓存；新接口有默认值兼容既有调用；只有实时 start 会话保存，原有学习逻辑保持。

实施结果：65 项测试通过（12.674 秒），14 个源文件类型检查通过。隐藏窗口验证两轮完整CSV与关闭窗口时第三轮数据落盘。直接复查开始失败、写入失败、断开、后台错误、切换、重复停止与关闭路径；保存失效时先停用执行，文件使用 exclusive 模式并逐会话关闭，未改变学习和识别判定。检查均使用临时目录和模拟鼠标，没有改动个人手势或原有记录。

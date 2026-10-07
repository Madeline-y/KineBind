# Single Acceleration Learning Implementation Plan

> **For agentic workers:** 当前用户已授权本会话执行。所引用的 superpowers 执行技能未安装，使用当前会话逐项实施；不派生代理。目录没有 Git 仓库。

**Goal:** 左挥可以只学习并使用一个加速度轴，不搜索陀螺轴或多轴组合。

**Architecture:** `single_accel=True` 沿现有学习接口传递，新学习器仅尝试 ax、ay、az 三个单轴候选。静止等负例窗口若已被选中轴的时长、有效运动、结束或边界条件明确拒绝，则不参与 DTW 安全余量淘汰，但仍执行完整校准流式零事件检查。

**Tech Stack:** Python、NumPy、Tkinter、unittest。

**Spec:** 本轮用户要求“左挥我就要一个加速度”；保留原有完整动作、启停、旧记录兼容及独立实录验收要求。

## Global Constraints

不修改用户原始示范和校准；不按手势名字硬编码物理轴。新入口只尝试三个单加速度轴，保存后由模型 axes 保证单轴识别。其他入口保持原有行为，不降低实际非目标零事件要求。

## Task 1: 端到端单轴学习

Files: kinebind/matching.py、robust.py、controller.py、gesture_app.py；tests/test_robustness.py。

- [x] 增加控制器边界失败测试：`app.learn_current(use_new=True, single_accel=True)`，断言 `len(model.axes)==1` 且轴小于 3，重载后两次完整动作各一个模拟执行请求，三类校准零事件。
- [x] 对只有陀螺变化的合成示范调用同一入口，断言拒绝，禁止回退陀螺或多轴。
- [x] 在 `learn(..., single_accel=False)`、`learn_current(..., single_accel=False)` 和 `learn_robust(..., single_accel=False)` 传递模式。候选 `range(3)`，候选大小只取 1。
- [x] 单轴模式调用 `model.match(window)`，对被时长、运动、结束或边界门明确拒绝、返回不可参与匹配分数的窗口排除距离余量淘汰；保留最近距离、被证据条件拒绝的窗口数与参与距离校准的最小距离，流式校准仍全量回放并要求零事件。
- [x] 界面新增“单加速度轴学习并保存”，通过 `lambda: self.learn(single_accel=True)` 调用；成功报告显示单轴和加速度模式。
- [x] 跑新增文件、类型检查、完整测试与隐藏窗口检查。在内存对左挥6调用新入口复现，不自动保存模型，不执行真实鼠标。
- [x] 更新操作说明和实施结果；独立实录验收继续待完成。无 Git 分支可提交。

Self-review: 搜索只允许加速度单轴，旧入口保持旧行为；数据存储与执行门复用已有接口，不新增用户数据格式。已授权执行，无需再次询问执行方式。

实施结果：60 项自动测试通过，13 个源文件类型检查通过，GUI 隐藏窗口的新单轴后台学习、报告、保存重载、模拟执行及停止通过。直接复查单轴候选限制、旧接口默认值和共享匹配条件；未发现阻断问题。左挥6使用临时存储验证选中 az，阈值 0.1734050639，五次训练回放各一次请求，三类校准各零请求。个人原始文件 SHA256 未变。独立新记录验收仍未完成，不代表真实识别率。

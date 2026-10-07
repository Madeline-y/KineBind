# Gesture Learning Robustness Implementation Plan

> **For agentic workers:** 原技能推荐的 superpowers:subagent-driven-development / superpowers:executing-plans 未安装，本次按用户已授权的 implement 指令在当前会话逐项执行；不派生代理。

**Goal:** 分类校准、范围建议、有效轴学习和完整事件拒绝共同改进通用手势，并提供独立验收报告。

**Architecture:** 保留版本一学习和识别语义，通过模型版本分派到新匹配器。控制器复用现有录制、通知、回放、执行和会话边界；分类校准、范围建议及验收各用小模块隔离。

**Tech Stack:** Python 3.11、NumPy、SciPy、Tkinter、matplotlib、unittest。

**Spec:** [规格](../specs/2026-10-07-gesture-learning-robustness-design.md)

## Global Constraints

- 五个独立已确认示范；三类校准各约十秒。
- 有效轴依据重复性和非目标区分度，不能写死某个轴或手势名。
- 左挥为完整左挥和回收，其他动作不强制往返。
- 完整动作单次执行，主动启停，停止和断开使旧会话失效。
- 旧文件按旧语义加载；新学习失败保留旧可用模型及原始记录。
- 独立新动作至少 9/10；另采三类非目标各三十秒零事件及零执行请求。
- 不把合成测试或训练记录冒充独立实录验收；不修改固件和已有个人录制。

## Files and responsibilities

- 扩展 `kinebind/data.py`：分类记录和报告、版本化持久化、录制种类。
- 新增 `kinebind/calibration.py`：类别、数据质量、静止噪声与零偏、范围建议。
- 扩展 `kinebind/storage.py`：独立草稿，失败重新学习保护已保存模型。
- 扩展 `kinebind/controller.py`：分类录制、建议范围、学习报告、版本分派入口。
- 新增 `kinebind/robust.py`：有效轴学习、新版本模型、完整流式事件校准。
- 扩展 `kinebind/matching.py`：旧接口保留，新版本按版本分派。
- 扩展 `gesture_app.py`：分类引导、建议预览、报告及草稿恢复。
- 新增 `kinebind/acceptance.py`、`verify_gestures.py`：冻结模型、独立数据清单和可复核统计。
- 新增 `tests/test_robustness.py`，复用控制器、临时存储与 FakeMouse。

## 01 — 分类校准与旧文件兼容

**Interfaces:** `Profile.calibration: dict[str, Motion]`，类别 `static/shake/pick_place`；`Profile.report: dict`；控制器 `set_calibration(category, motion)`。

- [x] 先在控制器测试类别保存、单类替换、静止冒充活动、重载及旧文件加载。
- [x] 执行 `python -m unittest discover -s tests -p test_robustness.py -v`，确认缺少接口导致失败。
- [x] 扩展数据、控制器和窗口录入；类别独立，错误具体，旧模型可用状态与新草稿分离。
- [x] 完成类别记录到保存/恢复的整条行为测试；运行旧手势测试文件检查兼容。

```python
app.set_calibration('static', still)
app.set_calibration('shake', shaking)
app.set_calibration('pick_place', handling)
assert set(app.profile.calibration) == {'static', 'shake', 'pick_place'}
```

## 02 — 范围建议与人工确认

**Interfaces:** `suggest_bounds(motion)` 返回候选范围与解释；控制器 `suggestion` 供窗口预览，确认仍通过原有 `save_review`。

- [x] 先测试带前后静止和内部停顿的动作不被拆半，未确认建议不增加示范。
- [x] 实现运动变化检测、候选区间、歧义和截断提示，窗口使用候选预览。
- [x] 保存和重新查看保持已确认边界；运行本测试文件与语法检查。

```python
app.import_review(recording)
assert not app.profile.samples
app.save_review(app.suggestion['start'], app.suggestion['end'])
assert len(app.profile.samples) == 1
```

## 03 — 新模型与流式识别

**Interfaces:** `learn(profile, progress=None, use_new=False)` 分派；`RobustModel.to_dict/from_dict/match`；`Recognizer(model, gesture_id, session)` 根据版本选择引擎。

- [x] 写有效轴不同、静态姿态不能冒充动作、完整/半动作、纯平移、重载一致及三类拒绝测试。
- [x] 固定尺度、按有效维度归一化；从静止估计偏移与噪声；根据留一示范一致性及各类背景距离筛选轴组合。
- [x] 校准运动和结束条件，阈值不超过保守负例边界；流式背景校准零事件才保存。
- [x] 用原有事件和执行门对接，测试两个独立动作各一个 FakeMouse 请求，停止后零请求。
- [x] 学习使用数据快照，后台进度和错误可见；失败报告不覆盖旧可用模型。

```python
app.learn_current(use_new=True)
app.start_recognition()
for t, row in zip(series.times, series.values):
    app.on_sample(float(t), row, float(t))
assert mouse.requests == [100, 100]
```

## 04 — 诊断与补录闭环

**Interfaces:** 报告字段包括状态、版本、原因、示范编号、类别、片段时间、有效轴、阈值及类别摘要。

- [x] 测试失败报告定位类别，修正后报告变成功，报告与草稿可恢复。
- [x] UI 提供查看报告与恢复草稿，提示到已有检查或类别补录入口。
- [x] 将报告关联训练数据指纹，修订后过期结果不作为当前成功。
- [x] 自查失败、停止、断开、后台结束等状态及已有 GUI smoke 行为。

## 05 — 独立数据验收

**Interfaces:** 验收清单包含模型文件、人工确认目标区间与三类非目标记录；输出模型/数据 SHA256、逐次结果、事件和模拟执行统计。

- [x] 写按区间统计而非事件总数、提前/重复触发、训练数据重用、缺数据和模型版本检查测试。
- [x] 提供命令行入口和说明，缺记录返回未完成，失败返回失败，只有全部条件满足才通过。
- [x] 完成 GUI smoke、类型检查（工具可用时）、全部测试一次及独立代码复查。
- [x] 更新票据状态与用户操作说明，真实数据不足时保持 05 验收待完成。

```powershell
python verify_gestures.py --manifest acceptance_manifest.json --output acceptance_report.json
python -m compileall -q kinebind gesture_app.py verify_gestures.py
python -m unittest discover -s tests -v
python gesture_app.py --smoke-test
```

## Execution and review

用户已授权开始执行，不重复询问执行方式。先完成 01/02，再 03/04，最后验收工具与整体验证。没有 Git 仓库，不能提交到现有分支；保存文件并记录校验结果。未安装独立 tdd/code-review 技能，使用先失败测试和人工代码复查完成相应意图。

## 最终状态

01–04 与 05 的软件工具已实施。57 项测试、13 个源文件类型检查及隐藏窗口检查通过。

- [ ] 完成 05 的独立实录指标及真实鼠标验收；尚缺重新学习后的模型、十次新目标和三类各三十秒新记录。

详细结果见 docs/gesture_learning_validation.md；未取得真实数据前保持 05 待验收。

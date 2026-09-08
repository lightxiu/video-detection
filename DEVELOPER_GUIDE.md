# 开发教程：看懂并修改这个项目

> 这份文档教你**自己动手改代码**，不是替我干活。跟着「动手做」走一遍，你就知道每个功能在哪、怎么改、改完怎么验证。

---

## 0. 30 秒理解项目

一句话：读视频 → YOLO 检测目标 → 画框出视频，同时把检测结果统计起来 → 交给本地 LLM（Ollama）生成一份中文巡检报告。

```
视频 → VideoLoader → Detector(YOLO) → Visualizer → VideoWriter → 标注视频
                         ↘
                    DetectionStats(统计) → ReportGenerator(Ollama) → 巡检报告.md
```

---

## 1. 文件地图（每个模块在哪、干嘛）

| 文件 | 职责 | 里面有什么 |
|------|------|-----------|
| `main.py` | 主流程，把各模块串起来 | `main()`：argparse 参数、检测循环、报告生成 |
| `configs/model_config.py` | 所有配置集中在这 | `MODEL_CONFIG`、`LLM_CONFIG`、`CLASS_NAME_MAP` |
| `src/video_loader.py` | 读视频 | `VideoLoader`：`read()` / `get_fps()` / `release()` |
| `src/detector.py` | 检测（对外接口） | `Detector`：`detect()` / `names` |
| `src/backends.py` | 检测后端（可换引擎） | `PyTorchBackend`、`TensorRTBackend`(未实现)、`create_backend()` |
| `src/visualizer.py` | 画框、画统计、画 FPS | `Visualizer`：`draw()` |
| `src/video_writer.py` | 写视频文件 | `VideoWriter` |
| `src/fps_monitor.py` | 算 FPS | `FPSMonitor` |
| `src/report_generator.py` | **统计 + LLM 报告（本次核心）** | `DetectionStats`、`ReportGenerator` |

**记忆口诀**：改「怎么检测」→ `backends.py`；改「报告长什么样」→ `report_generator.py`；改「参数」→ `model_config.py`；改「串起来的流程」→ `main.py`。

---

## 2. 怎么跑起来

```bash
# 前置：Ollama 已启动 + 拉好模型、yolov8n.pt 在项目根目录、torch 是 CUDA 版

# 完整跑（检测 + LLM 报告）
python main.py --video vedio/vtest.avi

# 只跑检测、跳过 LLM（改检测相关代码时用，省 1-2 分钟等模型）
python main.py --video vedio/vtest.avi --no-report
```

输出：
- 标注视频 → `output/results/<视频名>_annotated.mp4`
- 报告 → `output/reports/<视频名>_report.md`

---

## 3. 数据怎么流（搞懂流程最重要）

打开 `main.py`，按顺序读这几行，就懂全流程了：

1. `loader = VideoLoader(args.video)` —— 打开视频
2. `detector = Detector(...)` —— 加载 YOLO 模型
3. 循环里：`results = detector.detect(frame)` → 检测一帧
4. `stats.update(results, detector.names)` → 把结果喂给统计器
5. `visualizer.draw(...)` → 画框
6. `writer.write(...)` → 写视频
7. 循环结束：`reporter.generate(stats.summary())` → 生成报告

**一句话流程**：每一帧「检测 → 统计 → 画框 → 写视频」，视频播完把累计统计交给 LLM 出报告。

---

## 4. 动手①：改 system prompt（最常用的改动）

**在哪写**：`src/report_generator.py` 里，类 `ReportGenerator` 的第一个常量 `REPORT_PROMPT`。

打开文件，你会看到：

```python
REPORT_PROMPT = (
    "你是边缘智能巡检系统的报告生成助手。请根据下面的目标检测统计结果，"
    "生成一份简洁、客观的中文巡检报告。\n\n"
    "## 检测统计\n{stats_text}\n\n"      # ← {stats_text} 会被真实统计数字替换
    "## 要求\n"
    "1. 使用 Markdown 格式，包含三部分：一、巡检概况；二、目标类别统计；三、结论与建议。\n"
    "2. 客观描述，不要编造统计之外的信息；...\n"
    "3. 结论面向运维人员，突出主要目标...\n"
    "4. 直接输出报告正文，不要输出任何额外解释。"
)
```

**可以改什么**（改完保存，重新 `python main.py --video ...` 就能看到效果）：

| 想改 | 改哪句 | 举例 |
|------|--------|------|
| 报告角色/领域 | 第 1 句「你是…助手」 | 改成「你是水产养殖巡检专家」 |
| 输出结构 | 要求 1 | 加第四部分「四、异常告警」 |
| 语气详略 | 要求 2 / 3 | 改成「每条结论不超过 20 字」 |
| 输出语言 | 要求里加一句 | 「全程用英文输出」 |
| 输出格式 | 要求 1 | 「用 JSON 输出」→ LLM 就会返回 JSON |

**验证方法**：改完跑一次，看 `output/reports/<视频名>_report.md` 的变化。

---

## 5. 动手②：直接和本地模型对话（理解 LLM 这一层）

代码里 `ReportGenerator` 做的事，本质就是向 Ollama 发一个 HTTP POST。你完全可以**不用代码、直接用命令和模型对话**，这样才知道代码在调什么。

### 5.1 看有哪些模型 / 拉模型

```bash
# Ollama 不在 PATH，用完整路径（或你自己配环境变量）
OLLAMA="/c/Users/user/AppData/Local/Programs/Ollama/ollama.exe"

"$OLLAMA" list            # 看本地有哪些模型
"$OLLAMA" pull qwen2.5:3b # 拉一个模型
"$OLLAMA" run qwen2.5:3b  # 进入对话（Ctrl+D 退出）
```

### 5.2 直接调 HTTP 接口（代码里的 `_call_ollama` 就是干这个）

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:3b",
  "prompt": "用一句话介绍你自己",
  "stream": false
}'
```

返回的 JSON 里，`response` 字段就是模型生成的文本。

### 5.3 对照：代码里对应的位置

打开 `src/report_generator.py`，看 `_call_ollama`：

```python
payload = {
    'model': self.model,
    'prompt': prompt,
    'stream': False,
    'options': {'temperature': 0.3},
}
resp = requests.post(f'{self.base_url}/api/generate', json=payload, timeout=self.timeout)
```

**这就是把上面那条 curl 用 Python 的 `requests` 发出去**，`resp.json()['response']` 取的就是模型回答。你改 `options` 里的 `temperature`（采样温度），就是改 curl 里没写但代码里写了的那个参数。

**建议练习**：先用 curl 手发几个请求（换 prompt、换 temperature），再看代码，一下就通了。

---

## 6. 动手③：改配置/阈值（一行见效）

都在 `configs/model_config.py`：

```python
MODEL_CONFIG = {
    'model_path': 'yolov8n.pt',   # 换模型文件
    'conf_threshold': 0.5,        # 抬到 0.6 可砍掉低置信度误检（飞盘/风筝就是误检）
    'iou_threshold': 0.45,
    'device': 'cuda',             # 没 GPU 就改 'cpu'
    'backend': 'pytorch',         # 预留 tensorrt/onnx
}

LLM_CONFIG = {
    'model': 'qwen2.5:3b',        # 换 qwen2.5:7b 更准更慢，1.5b 更快
    'report_interval': 0,         # 改成 30 = 每 30 秒出一份报告
    ...
}

CLASS_NAME_MAP = { ... }          # 英文类名 → 中文，80 类已补全
```

**改 `conf_threshold` 是最快能看出效果的实验**：改 0.5 → 0.7 重跑，对比报告里误检（飞盘/风筝）少了多少。

---

## 7. 动手④：改统计逻辑（进阶，体现工程能力）

统计在 `src/report_generator.py` 的 `DetectionStats` 类：

- `update(results, names)` —— 每一帧进来，累加计数/置信度/峰值/峰值帧号
- `summary()` —— 输出最终的统计字典

**想加一个新统计**（比如「每秒平均检测数」），三步：

1. 在 `__init__` 加一个计数器变量（如 `self._per_second = ...`）
2. 在 `update()` 里更新它
3. 在 `summary()` 里把它放进返回字典

然后 LLM 报告会自动带上（因为它读的就是 `summary()` 的结果拼进 prompt）。fallback 报告需要你顺带在 `_fallback_report()` 里加一行展示。

---

## 8. 改完怎么验证 + 调试

1. **快速验证检测改动**：`python main.py --video vedio/vtest.avi --no-report`（跳过 LLM，秒出结果）
2. **看完整效果**：去掉 `--no-report` 跑，看报告
3. **看标注视频**：打开 `output/results/` 里的 mp4，肉眼确认框画得对不对
4. **调试打印**：在 `DetectionStats.update` 或 `ReportGenerator.build_prompt` 里加 `print()`，看中间数据
5. **对比实验**：改参数前后各跑一次，报告和视频分别存好，对比

---

## 9. 速查表：想改 X → 去哪个文件

| 我想改… | 去这里 |
|---------|--------|
| 报告的语气/结构/语言 | `src/report_generator.py` → `REPORT_PROMPT` |
| 采样温度、超时 | `src/report_generator.py` → `_call_ollama` 的 `options` |
| 换 LLM 模型 | `configs/model_config.py` → `LLM_CONFIG['model']` |
| 检测置信度阈值 | `configs/model_config.py` → `MODEL_CONFIG['conf_threshold']` |
| 类别中文名 | `configs/model_config.py` → `CLASS_NAME_MAP` |
| 统计加字段 | `src/report_generator.py` → `DetectionStats` |
| 报告降级格式 | `src/report_generator.py` → `_fallback_report` |
| 换检测引擎(TensorRT/ONNX) | `src/backends.py` → `TensorRTBackend`(还没实现) |
| 检测循环/流程 | `main.py` → `main()` |
| 画框颜色/样式 | `src/visualizer.py` |

---

## 10. 建议的自学顺序

1. 先跑通 `--no-report`，看 `output/results/` 视频。
2. 读 `main.py`（就 70 行），搞懂流程。
3. 用 curl 直接和 Ollama 对话（第 5 节），搞懂 LLM 层。
4. 改一次 `REPORT_PROMPT`（第 4 节），看报告变化。
5. 改一次 `conf_threshold`（第 6 节），看误检变化。
6. 有余力再动 `DetectionStats` 加统计（第 7 节）。

每一步都能在 5 分钟内完成、都能看到结果，这样你才真正「懂了每一行」。

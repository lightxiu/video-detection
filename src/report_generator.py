import os

import requests


class DetectionStats:
    """聚合检测结果：类别数量、平均置信度、单帧峰值、时间范围。"""

    def __init__(self, source=None, fps=None):
        self.source = source
        self.fps = fps
        self.total_frames = 0
        self.total_detections = 0
        self._class_counts = {}       # class_name -> 累计出现次数
        self._class_conf_sum = {}     # class_name -> 置信度累加
        self._class_peak = {}         # class_name -> 单帧最大出现数

    def update(self, results, names):
        frame_counts = {}
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = names.get(cls, str(cls))
                frame_counts[name] = frame_counts.get(name, 0) + 1
                self._class_counts[name] = self._class_counts.get(name, 0) + 1
                self._class_conf_sum[name] = self._class_conf_sum.get(name, 0.0) + conf
                self.total_detections += 1

        for name, count in frame_counts.items():
            self._class_peak[name] = max(self._class_peak.get(name, 0), count)

        self.total_frames += 1

    def summary(self):
        detections = []
        for name in sorted(self._class_counts):
            count = self._class_counts[name]
            avg_conf = self._class_conf_sum[name] / count if count else 0.0
            detections.append({
                'class': name,
                'count': count,
                'avg_conf': round(avg_conf, 3),
                'peak_per_frame': self._class_peak[name],
            })

        duration = self.total_frames / self.fps if self.fps else None
        return {
            'source': self.source,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'duration_seconds': round(duration, 1) if duration else None,
            'total_detections': self.total_detections,
            'detections': detections,
        }


class ReportGenerator:
    """调用本地 Ollama 生成自然语言巡检报告（单次 API 调用）。"""

    REPORT_PROMPT = (
        "你是边缘智能巡检系统的报告生成助手。请根据下面的目标检测统计结果，"
        "生成一份简洁、客观的中文巡检报告。\n\n"
        "## 检测统计\n{stats_text}\n\n"
        "## 要求\n"
        "1. 使用 Markdown 格式，包含三部分：一、巡检概况；二、目标类别统计；三、结论与建议。\n"
        "2. 客观描述，不要编造统计之外的信息；平均置信度偏低的类别可提示需人工复核。\n"
        "3. 结论面向运维人员，突出主要目标、出现频次、是否需要关注。\n"
        "4. 直接输出报告正文，不要输出任何额外解释。"
    )

    def __init__(self, base_url='http://localhost:11434', model='qwen2.5:3b',
                 timeout=120, class_name_map=None):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.class_name_map = class_name_map or {}

    def _localize(self, name):
        return self.class_name_map.get(name, name)

    def build_prompt(self, summary):
        lines = [f"- 视频源：{summary['source']}"]
        lines.append(f"- 处理帧数：{summary['total_frames']}")
        if summary['duration_seconds'] is not None:
            lines.append(f"- 视频时长（秒）：{summary['duration_seconds']}")
        lines.append(f"- 检测目标总数：{summary['total_detections']}")

        if summary['detections']:
            lines.append("- 各类别统计：")
            for d in summary['detections']:
                name = self._localize(d['class'])
                lines.append(
                    f"  - {name}：{d['count']} 次，平均置信度 {d['avg_conf']}，单帧峰值 {d['peak_per_frame']}"
                )
        else:
            lines.append("- 未检测到目标")

        return self.REPORT_PROMPT.format(stats_text='\n'.join(lines))

    def generate(self, summary):
        prompt = self.build_prompt(summary)
        try:
            return self._call_ollama(prompt)
        except requests.RequestException as exc:
            return self._fallback_report(summary, str(exc))

    def _call_ollama(self, prompt):
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0.3},
        }
        resp = requests.post(f'{self.base_url}/api/generate', json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()['response'].strip()

    def _fallback_report(self, summary, error):
        lines = [
            '# 视频检测巡检报告',
            '',
            f'> 提示：本地 LLM（Ollama）暂不可用，以下为基于统计数据的自动摘要。错误：{error}',
            '',
            '## 一、巡检概况',
            f"- 视频源：{summary['source']}",
            f"- 处理帧数：{summary['total_frames']}",
        ]
        if summary['duration_seconds'] is not None:
            lines.append(f"- 视频时长：{summary['duration_seconds']} 秒")
        lines.append(f"- 检测目标总数：{summary['total_detections']}")
        lines.append('')
        lines.append('## 二、目标类别统计')

        if summary['detections']:
            lines.append('| 类别 | 出现次数 | 平均置信度 | 单帧峰值 |')
            lines.append('| --- | --- | --- | --- |')
            for d in summary['detections']:
                name = self._localize(d['class'])
                lines.append(f"| {name} | {d['count']} | {d['avg_conf']} | {d['peak_per_frame']} |")
        else:
            lines.append('未检测到目标。')

        lines.append('')
        lines.append('## 三、结论与建议')
        lines.append('（LLM 可用后将由模型生成自然语言结论。）')
        return '\n'.join(lines)

    def save(self, report, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        return path

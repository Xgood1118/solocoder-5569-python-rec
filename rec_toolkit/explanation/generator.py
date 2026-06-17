import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ReasonTemplate:
    channel: str
    template: str
    condition: Optional[str] = None


class ReasonGenerator:
    def __init__(self):
        self.templates: Dict[str, List[ReasonTemplate]] = {}
        self._init_default_templates()

    def _init_default_templates(self):
        default_templates = [
            ReasonTemplate(channel='user_cf', template='因为你喜欢{item}，猜你也喜欢这个'),
            ReasonTemplate(channel='user_cf', template='和你兴趣相似的用户还看了这些'),
            ReasonTemplate(channel='item_cf', template='和你看过的{item}相似'),
            ReasonTemplate(channel='item_cf', template='根据你喜欢的{item}推荐'),
            ReasonTemplate(channel='content', template='基于你的兴趣标签推荐'),
            ReasonTemplate(channel='content', template='因为你对{topic}感兴趣'),
            ReasonTemplate(channel='svd', template='根据你的偏好推荐'),
            ReasonTemplate(channel='svd', template='个性化推荐'),
            ReasonTemplate(channel='popular', template='全网热门'),
            ReasonTemplate(channel='popular', template='大家都在看'),
            ReasonTemplate(channel='cold_start', template='新人专享热门推荐'),
            ReasonTemplate(channel='cold_start', template='为你精心挑选'),
            ReasonTemplate(channel='association', template='买过的人还买了'),
            ReasonTemplate(channel='association', template='经常一起出现'),
            ReasonTemplate(channel='bandit', template='根据你的反馈推荐'),
            ReasonTemplate(channel='bandit', template='为你优化的推荐'),
        ]

        for t in default_templates:
            if t.channel not in self.templates:
                self.templates[t.channel] = []
            self.templates[t.channel].append(t)

    def add_template(self, channel: str, template: str, condition: Optional[str] = None):
        if channel not in self.templates:
            self.templates[channel] = []
        self.templates[channel].append(ReasonTemplate(
            channel=channel, template=template, condition=condition
        ))

    def generate_reason(self, channel: str, context: Optional[Dict] = None) -> str:
        context = context or {}

        if channel not in self.templates or not self.templates[channel]:
            return f'来自{channel}的推荐'

        templates = self.templates[channel]

        matching_templates = []
        for t in templates:
            if t.condition:
                if self._evaluate_condition(t.condition, context):
                    matching_templates.append(t)
            else:
                matching_templates.append(t)

        if not matching_templates:
            matching_templates = templates

        template = matching_templates[0].template

        try:
            return template.format(**context)
        except (KeyError, IndexError):
            return template

    def generate_reasons(self, channels: List[str], context: Optional[Dict] = None) -> List[str]:
        reasons = []
        for channel in channels:
            reason = self.generate_reason(channel, context)
            if reason not in reasons:
                reasons.append(reason)
        return reasons

    def _evaluate_condition(self, condition: str, context: Dict) -> bool:
        try:
            return bool(eval(condition, {}, context))
        except:
            return False

    def get_all_templates(self) -> Dict[str, List[str]]:
        result = {}
        for channel, templates in self.templates.items():
            result[channel] = [t.template for t in templates]
        return result

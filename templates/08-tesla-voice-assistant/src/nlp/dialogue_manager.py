"""
对话管理器
Dialogue manager for handling conversations
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class Intent(Enum):
    """意图类型"""
    # 车辆控制
    CLIMATE_CONTROL = "climate_control"
    SEAT_CONTROL = "seat_control"
    WINDOW_CONTROL = "window_control"
    MEDIA_CONTROL = "media_control"
    
    # 导航
    NAVIGATE = "navigate"
    FIND_CHARGER = "find_charger"
    FIND_PARKING = "find_parking"
    
    # 信息查询
    BATTERY_STATUS = "battery_status"
    TRIP_INFO = "trip_info"
    WEATHER = "weather"
    
    # 通用
    GREETING = "greeting"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """实体"""
    type: str
    value: str
    confidence: float


@dataclass
class IntentResult:
    """意图识别结果"""
    intent: Intent
    confidence: float
    entities: List[Entity] = field(default_factory=list)
    original_text: str = ""


@dataclass
class DialogueState:
    """对话状态"""
    intent: Optional[Intent] = None
    entities: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, any] = field(default_factory=dict)
    previous_intents: List[Intent] = field(default_factory=list)


class IntentClassifier:
    """意图分类器（基于规则）"""
    
    def __init__(self):
        # 意图模式
        self.intent_patterns = {
            Intent.CLIMATE_CONTROL: [
                r'(打开|开启|调到|设置).{0,5}(空调|暖气|冷气)',
                r'(温度|气温).{0,5}(\d+)',
                r'(太热|太冷|有点冷|有点热)',
            ],
            Intent.SEAT_CONTROL: [
                r'(座椅|座位).{0,5}(加热|通风|调节)',
                r'(座椅|座位).{0,5}(温度|高低)',
            ],
            Intent.MEDIA_CONTROL: [
                r'(播放|暂停|停止|下一首|上一首)',
                r'(音乐|歌曲|电台)',
                r'(音量).{0,5}(\d+|大点|小点)',
            ],
            Intent.NAVIGATE: [
                r'(导航|带我去|去).{0,10}(目的地|地址|位置)',
                r'(怎么去|路线)',
            ],
            Intent.FIND_CHARGER: [
                r'(充电站|充电桩|充电)',
                r'(找|最近).{0,5}(充电|充电站)',
            ],
            Intent.BATTERY_STATUS: [
                r'(电量|电池|还剩多少电)',
                r'(续航|还能跑多远)',
            ],
            Intent.WEATHER: [
                r'(天气|气温|下雨|下雪)',
            ],
            Intent.GREETING: [
                r'(你好|您好|hi|hello)',
            ],
            Intent.THANKS: [
                r'(谢谢|感谢|thanks)',
            ],
            Intent.GOODBYE: [
                r'(再见|拜拜|bye)',
            ],
        }
        
        logger.info("意图分类器初始化")
    
    def classify(self, text: str) -> IntentResult:
        """分类意图"""
        text_lower = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    confidence = 0.9  # 规则匹配置信度
                    entities = self._extract_entities(text, intent)
                    
                    return IntentResult(
                        intent=intent,
                        confidence=confidence,
                        entities=entities,
                        original_text=text
                    )
        
        # 未知意图
        return IntentResult(
            intent=Intent.UNKNOWN,
            confidence=0.0,
            original_text=text
        )
    
    def _extract_entities(self, text: str, intent: Intent) -> List[Entity]:
        """提取实体"""
        entities = []
        
        # 温度
        temp_match = re.search(r'(\d+).{0,3}(度|℃|°)', text)
        if temp_match:
            entities.append(Entity(
                type='temperature',
                value=temp_match.group(1),
                confidence=0.95
            ))
        
        # 地址/位置
        if intent in [Intent.NAVIGATE, Intent.FIND_CHARGER]:
            location_match = re.search(r'(去|到|找).{1,30}', text)
            if location_match:
                entities.append(Entity(
                    type='location',
                    value=location_match.group(0)[1:],  # 去掉"去"字
                    confidence=0.8
                ))
        
        # 媒体控制
        if intent == Intent.MEDIA_CONTROL:
            if '播放' in text:
                entities.append(Entity(type='action', value='play', confidence=0.95))
            elif '暂停' in text:
                entities.append(Entity(type='action', value='pause', confidence=0.95))
            elif '下一首' in text:
                entities.append(Entity(type='action', value='next', confidence=0.95))
        
        return entities


class DialogueManager:
    """对话管理器"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.state = DialogueState()
        
        # 响应模板
        self.response_templates = {
            Intent.CLIMATE_CONTROL: {
                'success': "好的，已为您{action}空调到{value}度",
                'clarify': "请问您想设置多少度？",
            },
            Intent.SEAT_CONTROL: {
                'success': "座椅{action}已{status}",
                'clarify': "请问您想调节哪个座椅？",
            },
            Intent.NAVIGATE: {
                'success': "好的，正在为您导航到{destination}",
                'clarify': "请问您要去哪里？",
            },
            Intent.FIND_CHARGER: {
                'success': "已找到{count}个附近的充电站，最近的距离{distance}公里",
                'clarify': "正在搜索附近的充电站...",
            },
            Intent.BATTERY_STATUS: {
                'success': "当前电量{soc}%，预计续航{range}公里",
            },
            Intent.WEATHER: {
                'success': "当前温度{temperature}度，{condition}",
            },
            Intent.GREETING: {
                'success': "您好，我是您的Tesla助手，有什么可以帮您的吗？",
            },
            Intent.THANKS: {
                'success': "不客气！",
            },
            Intent.GOODBYE: {
                'success': "再见，祝您一路顺风！",
            },
            Intent.UNKNOWN: {
                'success': "抱歉，我没听懂，您可以再说一遍吗？",
            },
        }
        
        logger.info("对话管理器初始化")
    
    def process(self, user_input: str) -> Tuple[str, IntentResult]:
        """
        处理用户输入
        
        Args:
            user_input: 用户输入文本
        
        Returns:
            (响应文本, 意图识别结果)
        """
        logger.info(f"用户输入: {user_input}")
        
        # 1. 意图识别
        intent_result = self.intent_classifier.classify(user_input)
        
        logger.info(
            f"识别意图: {intent_result.intent.value} "
            f"(置信度: {intent_result.confidence:.2f})"
        )
        
        # 2. 更新对话状态
        self._update_state(intent_result)
        
        # 3. 生成响应
        response = self._generate_response(intent_result)
        
        logger.info(f"响应: {response}")
        
        return response, intent_result
    
    def _update_state(self, intent_result: IntentResult):
        """更新对话状态"""
        # 保存之前的意图
        if self.state.intent:
            self.state.previous_intents.append(self.state.intent)
        
        # 更新当前意图
        self.state.intent = intent_result.intent
        
        # 更新实体
        for entity in intent_result.entities:
            self.state.entities[entity.type] = entity.value
        
        # 更新上下文
        self.state.context['last_intent'] = intent_result.intent.value
        self.state.context['timestamp'] = self._get_timestamp()
    
    def _generate_response(self, intent_result: IntentResult) -> str:
        """生成响应"""
        intent = intent_result.intent
        templates = self.response_templates.get(intent, {})
        
        # 检查是否需要澄清
        if self._needs_clarification(intent_result):
            return templates.get('clarify', templates.get('success', "请提供更多信息"))
        
        # 生成成功响应
        template = templates.get('success', "好的")
        
        # 填充模板变量
        response = self._fill_template(template, intent_result)
        
        return response
    
    def _needs_clarification(self, intent_result: IntentResult) -> bool:
        """检查是否需要澄清"""
        intent = intent_result.intent
        
        # 导航需要目的地
        if intent == Intent.NAVIGATE:
            return not any(e.type == 'location' for e in intent_result.entities)
        
        # 空调控制需要温度（如果指定了"调到"）
        if intent == Intent.CLIMATE_CONTROL:
            if '调到' in intent_result.original_text or '设置' in intent_result.original_text:
                return not any(e.type == 'temperature' for e in intent_result.entities)
        
        return False
    
    def _fill_template(self, template: str, intent_result: IntentResult) -> str:
        """填充模板"""
        response = template
        
        # 从实体中填充
        for entity in intent_result.entities:
            placeholder = f"{{{entity.type}}}"
            if placeholder in response:
                response = response.replace(placeholder, str(entity.value))
        
        # 根据意图填充特定变量
        intent = intent_result.intent
        
        if intent == Intent.BATTERY_STATUS:
            # 模拟电池状态
            response = response.format(
                soc=75,
                range=280
            )
        elif intent == Intent.FIND_CHARGER:
            response = response.format(
                count=3,
                distance=2.5
            )
        elif intent == Intent.WEATHER:
            response = response.format(
                temperature=25,
                condition="晴朗"
            )
        elif intent == Intent.CLIMATE_CONTROL:
            # 从实体获取温度，或使用默认
            temp = next(
                (e.value for e in intent_result.entities if e.type == 'temperature'),
                '24'
            )
            response = response.format(
                action='设置',
                value=temp
            )
        
        return response
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def reset(self):
        """重置对话状态"""
        self.state = DialogueState()
        logger.info("对话状态已重置")


def test_dialogue_manager():
    """测试对话管理器"""
    manager = DialogueManager()
    
    # 测试用例
    test_cases = [
        "你好",
        "打开空调到24度",
        "导航到最近的充电站",
        "还有多少电量",
        "播放音乐",
        "谢谢",
        "再见"
    ]
    
    logger.info("\n" + "="*60)
    logger.info("对话管理器测试")
    logger.info("="*60)
    
    for user_input in test_cases:
        response, intent_result = manager.process(user_input)
        
        logger.info(f"\n用户: {user_input}")
        logger.info(f"意图: {intent_result.intent.value}")
        logger.info(f"实体: {[f'{e.type}={e.value}' for e in intent_result.entities]}")
        logger.info(f"助手: {response}")
    
    logger.success("\n✅ 对话管理器测试完成")


if __name__ == "__main__":
    test_dialogue_manager()

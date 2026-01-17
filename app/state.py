from typing import List, Dict
from typing_extensions import TypedDict


class ChatState(TypedDict):
    question: str
    context: List[str]
    answer: str
    history: List[Dict[str, str]]

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prompts used by FOCUS: target extraction via in-context learning, existence check, teacher hint."""

import re

TARGET_EXTRACTION_EXAMPLES = [
    ("What is the color of the car?", ["car"]),
    ("Is the soccer ball on the left or right of the water dispenser?", ["soccer ball", "water dispenser"]),
    ("What is the material of the glove? (A) rubber (B) cotton (C) leather", ["glove"]),
    ("How many bikes are in the image?", ["bike"]),
    ("What is written on the sign above the red door?", ["sign above the red door"]),
]

_ANSWER_PREFIX = "I need the info about"


def build_target_extraction_prompt(question: str) -> str:
    """Few-shot prompt asking the model which object(s) a question is about (Fig. 2 (I))."""
    lines = [
        "For each question, list the object(s) that must be found in the image to answer it, separated by ';'.",
        "",
    ]
    for q, targets in TARGET_EXTRACTION_EXAMPLES:
        lines += [f"Question: {q}", f"Answer: {_ANSWER_PREFIX} {'; '.join(targets)}.", ""]
    lines += [f"Question: {question}", "Answer:"]
    return "\n".join(lines)


def parse_targets(text: str, max_targets: int = 3, max_words: int = 8) -> list[str]:
    """Parse ``"I need the info about a; b."`` into ``["a", "b"]``. Returns ``[]`` if nothing usable.

    Parts longer than ``max_words`` words are dropped: they are almost always a model that ignored the
    format, and would only produce a meaningless relevance map.
    """
    idx = text.find(_ANSWER_PREFIX)
    if idx >= 0:
        text = text[idx + len(_ANSWER_PREFIX) :]
    text = text.strip().split("\n")[0]
    targets = []
    for part in text.split(";"):
        part = re.sub(r"^(the|a|an)\s+", "", part.strip().rstrip(".").strip(), flags=re.IGNORECASE)
        if part and len(part.split()) <= max_words and part.lower() not in (t.lower() for t in targets):
            targets.append(part)
    return targets[:max_targets]


def existence_question(target: str) -> str:
    return f"Is there a {target} in the image? Answer Yes or No."


def relevance_question(target: str) -> str:
    """Generic template whose forward pass provides the value features for the relevance map."""
    return f"Is there a {target} in the image?"


DEFAULT_TEACHER_HINT = "Zoomed-in views of the image regions relevant to the question:"

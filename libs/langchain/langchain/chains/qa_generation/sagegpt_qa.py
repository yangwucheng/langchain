from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain import LLMChain, PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain.chains import QAGenerationChain
from langchain.schema import Document


class SageGPTQAGenerationChain(QAGenerationChain):
    async def _acall(self, inputs: Dict[str, Any], run_manager: Optional[AsyncCallbackManagerForChainRun] = None) -> \
            Dict[str, Any]:
        pass

    @property
    def _chain_type(self) -> str:
        raise NotImplementedError

    @staticmethod
    def get_json_list_str(a_str: str) -> Any | None:
        try:
            a_str = re.sub(r"\"\s*,\s*}", "\"}", a_str)
            a_str = re.sub(r"}\s*,\s*]", "}\n  ]", a_str)
            a_str = re.sub(r"\"[\s\]}]+$", "\"\n  }\n]", a_str)
            p = re.compile(r"(\[.*])", re.MULTILINE | re.S)
            matched_list = p.findall(a_str)
            if matched_list:
                logging.info("pattern (\\[.*]) matched")
                json.loads(matched_list[0])
                logging.info(f"pattern (\\[.*]) extract result: {matched_list[0]}")
                return matched_list[0]
            else:
                return None
        except:
            return None

    @staticmethod
    def split_text(text: str, max_pair_count: int = 5, question_pattern: str = r"^\d+\."):
        texts = []
        pair_count = 0
        current_text = ''
        for line in text.split("\n"):
            line = line.strip()
            if re.match(question_pattern, line):
                pair_count += 1
                if pair_count % max_pair_count == 1:
                    if current_text:
                        texts.append(current_text)
                        current_text = ''
            current_text += line + "\n"
        if current_text:
            texts.append(current_text)
        return texts

    @staticmethod
    def get_matched_index(matched):
        return matched.group('index')

    def fix_json_error(self, text: str) -> str:
        templ = """""你是一个JSON转换器，你必须将给定文本转换成如下格式：
[
  {{
    "question": "在这里插入问题1",
    "answer": "在这里插入问题1对应的答案"
  }},
  {{
    "question": "在这里插入问题N",
    "answer": "在这里插入问题N对应的答案"
  }}
]

例如：
给定文本：
```
1. 按照什么标准分类大坝？
大坝可以按照抵抗水头压力的机制不同，分为重力坝和拱坝。
2. 重力坝的特点是什么？
重力坝是利用坝体自身重量来抵抗上游水压力并保持自身稳定的建筑物，比如著名的三峡大坝就是混凝土重力坝。
```
转换结果为：
[
  {{
    "question": "按照什么标准分类大坝？",
    "answer": "大坝可以按照抵抗水头压力的机制不同，分为重力坝和拱坝。"
  }},
  {{
    "question": "重力坝的特点是什么？",
    "answer": "重力坝是利用坝体自身重量来抵抗上游水压力并保持自身稳定的建筑物，比如著名的三峡大坝就是混凝土重力坝。"
  }}
]

给定文本：
```
{text}
```
转换结果为：
"""
        logging.info(f"fix json error: {text}")
        text = re.sub(r"QA对(?P<index>\d+\.)", self.get_matched_index, text, flags=re.MULTILINE | re.S)
        chain = LLMChain(llm=self.llm_chain.llm, prompt=PromptTemplate.from_template(templ))
        result = ''
        for a_text in self.split_text(text.strip()):
            logging.info(f"fix json text piece: {a_text}")
            try:
                valid_json_list_str = self.get_json_list_str(chain.run(a_text))
                if valid_json_list_str:
                    logging.info(f"fixed valid json list str: {valid_json_list_str}")
                    result += valid_json_list_str + "\n"
                else:
                    logging.info(f"invalid json list str after llm fix")
            except Exception as e:
                logging.error(f"fix json text piece: {a_text} failed", e)

        return result

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, List]:
        if isinstance(inputs[self.input_key], list):
            docs = inputs[self.input_key]
        elif isinstance(inputs[self.input_key], Document):
            docs = [inputs[self.input_key]]
        else:
            docs = self.text_splitter.create_documents([inputs[self.input_key]])
        results = self.llm_chain.generate(
            [{"text": d.page_content} for d in docs], run_manager=run_manager
        )
        logging.info(f"results from llm {results}")
        qa = ''
        for res in results.generations:
            qa = self.get_json_list_str(res[0].text)
            if qa:
                break
            qa = self.fix_json_error(res[0].text)
            break

        return {self.output_key: qa}

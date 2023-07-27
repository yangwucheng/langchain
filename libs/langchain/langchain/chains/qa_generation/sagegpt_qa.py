from __future__ import annotations

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

    def fix_json_error(self, text: str) -> str:
        templ = """你必须将给定的文本转换成如下格式：


[
  {{
    \"question\": \"在这里插入问题1\",
    \"answer\": \"在这里插入问题1对应的答案\"
  }},
  {{
    \"question\": \"在这里插入问题N\",
    \"answer\": \"在这里插入问题N对应的答案\"
  }}
]


务必确保输出的结果是一个有效的JSON格式。
""" + \
                "例如：给定文本\"1. 按照什么标准分类大坝？\n大坝可以按照抵抗水头压力的机制不同，分为重力坝和拱坝。\n" \
                "2. 重力坝的特点是什么？\n重力坝是利用坝体自身重量来抵抗上游水压力并保持自身稳定的建筑物，比如著名的三峡大坝就是混凝土重力坝。\"，" \
                "转换结果为\"[\n  {{\n    \"question\": \"按照什么标准分类大坝？\",\n" \
                "    \"answer\": \"大坝可以按照抵抗水头压力的机制不同，分为重力坝和拱坝。\"\n  }},\n" \
                "  {{\n    \"question\": \"重力坝的特点是什么？\",\n" \
                "    \"answer\": \"重力坝是利用坝体自身重量来抵抗上游水压力并保持自身稳定的建筑物，比如著名的三峡大坝就是混凝土重力坝。\"\n  }}]" \
                "\"。\n\n按照上面的要求转换下列文本：\n\n\n{text}"

        chain = LLMChain(llm=self.llm_chain.llm, prompt=PromptTemplate.from_template(templ))
        return chain.run(text)

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
        p = re.compile(r"(\[.*])", re.MULTILINE | re.S)
        qa = ''
        for res in results.generations:
            matched_list = p.findall(res[0].text)
            if matched_list:
                logging.info("pattern (\\[.*]) matched")
                qa = matched_list[0]
                logging.info(f"pattern (\\[.*]) extract result: {qa}")
                break
            qa = self.fix_json_error(res[0].text)
            logging.info(f"without pattern result: {qa}")
            break

        return {self.output_key: qa}

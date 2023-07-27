from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain.chains import QAGenerationChain
import re

from langchain.schema import Document


class SageGPTQAGenerationChain(QAGenerationChain):
    async def _acall(self, inputs: Dict[str, Any], run_manager: Optional[AsyncCallbackManagerForChainRun] = None) -> \
            Dict[str, Any]:
        pass

    @property
    def _chain_type(self) -> str:
        raise NotImplementedError

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
        p = re.compile(r'```(.*)```', re.MULTILINE | re.S)
        qa = ''
        for res in results.generations:
            matched_list = p.findall(res[0].text)
            if matched_list:
                logging.info(f"pattern ``` matched")
                qa = json.loads(matched_list[0])
                logging.info(f"pattern ``` extract result: {qa}")
                break
            qa = res[0].text
            logging.info(f"without pattern result: {qa}")
            break

        return {self.output_key: qa}

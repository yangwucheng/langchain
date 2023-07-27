"""Test SS API wrapper."""

from pathlib import Path
from typing import Generator

import pytest

from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.chains.qa_generation.sagegpt_qa import SageGPTQAGenerationChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.chat_models.sagegpt_openai import SageGPTChatOpenAI
from langchain.document_loaders import PyMuPDFLoader, UnstructuredHTMLLoader
from langchain.llms.loading import load_llm
from langchain.llms.openai import OpenAI, OpenAIChat
from langchain.schema import LLMResult, HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_openai_call() -> None:
    """Test valid call to openai."""
    llm = SageGPTChatOpenAI(
        openai_api_base="http://127.0.0.1:28080/api/v1"
    )
    output = llm.predict_messages(
        messages=[HumanMessage(content="查询货号 ck库存")],
        model="sagegpt-1.0",
        plugins=[
            "23064",
            "23067",
            "23070",
            "23073",
            "71755",
            "72001"
        ],
        stream=False)
    print(output)


def test_qachain() -> None:
    llm = SageGPTChatOpenAI(
        openai_api_base="http://49.233.33.8:13689/api/v1"
    )
    templ = """请根据下面的文本片段生成5组QA对，5组QA对组成一个JSON列表，每组QA对是一个JSON字典:\n{text}"""
    chain = SageGPTQAGenerationChain.from_llm(llm=llm, prompt=PromptTemplate.from_template(templ))
    output = chain.run(
        "水库是人们为了防洪、发电、灌溉、供水、航运等，在山沟或河流的峡口处建造拦河坝，对河川径流在时间和空间上进行重新分配而形成的人工湖泊。水库一般由挡水建筑物、输水建筑物、泄水建筑物构成。（1）挡水建筑物，如大坝，其作用是拦截河流，抬高水位，形成水库；（2）输水建筑物，如输水隧洞、涵洞、输水管，其作用是输送水量，满足灌溉、发电、供水等兴利需求；（3）泄水建筑物，如溢洪道、泄洪洞、泄洪闸，其作用是宣泄洪水，保证大坝安全。根据《水利水电工程等级划分及洪水标准》，水库规模等级：大（1）型水库：总库容≥10亿立方米；大（2）型水库：1亿立方米≤总库容10亿立方米；中型水库：0.1亿立方米≤总库容1亿立方米；小（1）型水库：0.01亿立方米≤总库容0.1亿立方米；小（2）型水库：0.001亿立方米≤总库容0.01亿立方米。")
    print(output)


def test_pdf_loader_qachain() -> None:
    llm = SageGPTChatOpenAI(
        openai_api_base="http://49.233.33.8:13689/api/v1"
    )
    templ = """请根据下面的文本片段生成5组QA对，5组QA对组成一个JSON列表，每组QA对是一个JSON字典:\n{text}"""
    # loader = PyMuPDFLoader("../examples/layout-parser-paper.pdf")
    loader = PyMuPDFLoader("/Users/leoyang/Documents/7. 电子书/洞见数据之密.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    chain = SageGPTQAGenerationChain.from_llm(llm=llm, prompt=PromptTemplate.from_template(templ))
    output = chain.run({"text": texts[2]})
    print(output)


def test_html_loader_qachain() -> None:
    llm = SageGPTChatOpenAI(
        openai_api_base="http://49.233.33.8:13689/api/v1"
    )
    templ = """请根据下面的文本片段生成5组QA对，5组QA对组成一个JSON列表，每组QA对是一个JSON字典:\n{text}"""
    loader = UnstructuredHTMLLoader("../../../source-docs/sl.html")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    chain = SageGPTQAGenerationChain.from_llm(llm=llm, prompt=PromptTemplate.from_template(templ))
    output = chain.run({"text": texts[2]})
    print(output)


def test_openai_model_param() -> None:
    llm = OpenAI(model="foo")
    assert llm.model_name == "foo"
    llm = OpenAI(model_name="foo")
    assert llm.model_name == "foo"


def test_openai_extra_kwargs() -> None:
    """Test extra kwargs to openai."""
    # Check that foo is saved in extra_kwargs.
    llm = OpenAI(foo=3, max_tokens=10)
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = OpenAI(foo=3, model_kwargs={"bar": 2})
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        OpenAI(foo=3, model_kwargs={"foo": 2})

    # Test that if explicit param is specified in kwargs it errors
    with pytest.raises(ValueError):
        OpenAI(model_kwargs={"temperature": 0.2})

    # Test that "model" cannot be specified in kwargs
    with pytest.raises(ValueError):
        OpenAI(model_kwargs={"model": "text-davinci-003"})


def test_openai_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    llm = OpenAI(max_tokens=10)
    llm_result = llm.generate(["Hello, how are you?"])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == llm.model_name


def test_openai_stop_valid() -> None:
    """Test openai stop logic on valid configuration."""
    query = "write an ordered list of five items"
    first_llm = OpenAI(stop="3", temperature=0)
    first_output = first_llm(query)
    second_llm = OpenAI(temperature=0)
    second_output = second_llm(query, stop=["3"])
    # Because it stops on new lines, shouldn't return anything
    assert first_output == second_output


def test_openai_stop_error() -> None:
    """Test openai stop logic on bad configuration."""
    llm = OpenAI(stop="3", temperature=0)
    with pytest.raises(ValueError):
        llm("write an ordered list of five items", stop=["\n"])


def test_saving_loading_llm(tmp_path: Path) -> None:
    """Test saving/loading an OpenAI LLM."""
    llm = OpenAI(max_tokens=10)
    llm.save(file_path=tmp_path / "openai.yaml")
    loaded_llm = load_llm(tmp_path / "openai.yaml")
    assert loaded_llm == llm


def test_openai_streaming() -> None:
    """Test streaming tokens from OpenAI."""
    llm = OpenAI(max_tokens=10)
    generator = llm.stream("I'm Pickle Rick")

    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token["choices"][0]["text"], str)


def test_openai_multiple_prompts() -> None:
    """Test completion with multiple prompts."""
    llm = OpenAI(max_tokens=10)
    output = llm.generate(["I'm Pickle Rick", "I'm Pickle Rick"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)
    assert len(output.generations) == 2


def test_openai_streaming_error() -> None:
    """Test error handling in stream."""
    llm = OpenAI(best_of=2)
    with pytest.raises(ValueError):
        llm.stream("I'm Pickle Rick")


def test_openai_streaming_best_of_error() -> None:
    """Test validation for streaming fails if best_of is not 1."""
    with pytest.raises(ValueError):
        OpenAI(best_of=2, streaming=True)


def test_openai_streaming_n_error() -> None:
    """Test validation for streaming fails if n is not 1."""
    with pytest.raises(ValueError):
        OpenAI(n=2, streaming=True)


def test_openai_streaming_multiple_prompts_error() -> None:
    """Test validation for streaming fails if multiple prompts are given."""
    with pytest.raises(ValueError):
        OpenAI(streaming=True).generate(["I'm Pickle Rick", "I'm Pickle Rick"])


def test_openai_streaming_call() -> None:
    """Test valid call to openai."""
    llm = OpenAI(max_tokens=10, streaming=True)
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_openai_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = OpenAI(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    llm("Write me a sentence with 100 words.")
    assert callback_handler.llm_streams == 10


@pytest.mark.asyncio
async def test_openai_async_generate() -> None:
    """Test async generation."""
    llm = OpenAI(max_tokens=10)
    output = await llm.agenerate(["Hello, how are you?"])
    assert isinstance(output, LLMResult)


@pytest.mark.asyncio
async def test_openai_async_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = OpenAI(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    result = await llm.agenerate(["Write me a sentence with 100 words."])
    assert callback_handler.llm_streams == 10
    assert isinstance(result, LLMResult)


def test_openai_chat_wrong_class() -> None:
    """Test OpenAIChat with wrong class still works."""
    llm = OpenAI(model_name="gpt-3.5-turbo")
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_openai_chat() -> None:
    """Test OpenAIChat."""
    llm = OpenAIChat(max_tokens=10)
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_openai_chat_streaming() -> None:
    """Test OpenAIChat with streaming option."""
    llm = OpenAIChat(max_tokens=10, streaming=True)
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_openai_chat_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = OpenAIChat(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    llm("Write me a sentence with 100 words.")
    assert callback_handler.llm_streams != 0


@pytest.mark.asyncio
async def test_openai_chat_async_generate() -> None:
    """Test async chat."""
    llm = OpenAIChat(max_tokens=10)
    output = await llm.agenerate(["Hello, how are you?"])
    assert isinstance(output, LLMResult)


@pytest.mark.asyncio
async def test_openai_chat_async_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm = OpenAIChat(
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    result = await llm.agenerate(["Write me a sentence with 100 words."])
    assert callback_handler.llm_streams != 0
    assert isinstance(result, LLMResult)


def test_openai_modelname_to_contextsize_valid() -> None:
    """Test model name to context size on a valid model."""
    assert OpenAI().modelname_to_contextsize("davinci") == 2049


def test_openai_modelname_to_contextsize_invalid() -> None:
    """Test model name to context size on an invalid model."""
    with pytest.raises(ValueError):
        OpenAI().modelname_to_contextsize("foobar")


_EXPECTED_NUM_TOKENS = {
    "ada": 17,
    "babbage": 17,
    "curie": 17,
    "davinci": 17,
    "gpt-4": 12,
    "gpt-4-32k": 12,
    "gpt-3.5-turbo": 12,
}

_MODELS = models = [
    "ada",
    "babbage",
    "curie",
    "davinci",
]
_CHAT_MODELS = [
    "gpt-4",
    "gpt-4-32k",
    "gpt-3.5-turbo",
]


@pytest.mark.parametrize("model", _MODELS)
def test_openai_get_num_tokens(model: str) -> None:
    """Test get_tokens."""
    llm = OpenAI(model=model)
    assert llm.get_num_tokens("表情符号是\n🦜🔗") == _EXPECTED_NUM_TOKENS[model]


@pytest.mark.parametrize("model", _CHAT_MODELS)
def test_chat_openai_get_num_tokens(model: str) -> None:
    """Test get_tokens."""
    llm = ChatOpenAI(model=model)
    assert llm.get_num_tokens("表情符号是\n🦜🔗") == _EXPECTED_NUM_TOKENS[model]

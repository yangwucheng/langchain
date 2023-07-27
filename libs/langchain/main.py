import json
import logging
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
from flask import Blueprint, Flask, Response, request

from config import Config
from langchain import PromptTemplate
from langchain.chains.qa_generation.sagegpt_qa import SageGPTQAGenerationChain
from langchain.chat_models.sagegpt_openai import SageGPTChatOpenAI
from langchain.document_loaders import PyMuPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

executor = ThreadPoolExecutor(2)
app = Flask(__name__)
qa_bp = Blueprint('qa', __name__)


def init_log():
    # 输出日志到文件和控制台
    log_level = logging.DEBUG if app.config['DEBUG'] else logging.INFO

    logging.basicConfig(level=log_level,
                        format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            RotatingFileHandler('../../info.log', maxBytes=100 * 1024 * 1024, backupCount=5),
                            logging.StreamHandler()
                        ])


def do_extract_qa(source: str, source_type: str, destination: str, qa_count_per_trunk: int):
    try:
        if source_type.lower() == "pdf":
            loader = PyMuPDFLoader(app.config['SOURCE_BASE_PATH'] + source)
        elif source_type.lower() == "html":
            loader = UnstructuredHTMLLoader(app.config['SOURCE_BASE_PATH'] + source)
        else:
            loader = TextLoader(app.config['SOURCE_BASE_PATH'] + source)

        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        llm = SageGPTChatOpenAI(
            openai_api_base=app.config['SAGE_GPT_SERVICE_ADDRESS']
        )
        templ = f"请根据下面的文本片段生成{qa_count_per_trunk}组QA对，" \
                f"{qa_count_per_trunk}组QA对组成一个JSON列表，每组QA对是一个JSON字典:\n" \
                + "{text}"
        chain = SageGPTQAGenerationChain.from_llm(llm=llm, prompt=PromptTemplate.from_template(templ))

        i = 1
        for a_text in texts:
            try:
                logging.info(f"start extract qa from {source} chunk {i}")
                with open(app.config['DESTINATION_BASE_PATH'] + destination, 'a') as f:
                    f.write(chain.run({"text": a_text}))
                logging.info(f"finish extract qa from {source} chunk {i}")
            except Exception as e:
                logging.error(f"finish extract qa from {source} chunk {i} failed: ", e)
            i += 1
    except Exception as e:
        logging.error("do_extract_qa failed: ", e)


@qa_bp.route('/api/v1/qa', methods=['POST'])
def extract_qa():
    post_data = request.get_json()
    logging.info(
        f'user post data: {json.dumps(post_data, ensure_ascii=False)}')

    source = post_data['source']
    source_type: str = post_data['source_type']
    destination = post_data.get('destination', source + ".json")
    qa_count_per_trunk = post_data.get('qa_count_per_trunk', 5)
    executor.submit(do_extract_qa, source, source_type, destination, qa_count_per_trunk)

    return Response(json.dumps({
        "code": 0,
        "message": "QA抽取任务提交成功"
    }), content_type='application/json')


def init_config():
    load_dotenv()
    app.config.from_object(Config)


def create_app():
    app.register_blueprint(qa_bp)

    with app.app_context():
        init_config()
    init_log()
    return app


create_app()
if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=app.config['PORT'], use_reloader=True)

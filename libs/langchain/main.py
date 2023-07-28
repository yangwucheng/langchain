import json
import logging
import os
from peewee import SqliteDatabase
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
from task_model import TaskModel

executor = ThreadPoolExecutor(2)
app = Flask(__name__)
qa_bp = Blueprint('qa', __name__)
current_dir = os.path.dirname(__file__)


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


def do_extract_qa(a_task: TaskModel):
    try:
        logging.info(f"Task {a_task.id} running")
        a_task.status = "Running"
        a_task.save()

        if a_task.source_type == "pdf":
            loader = PyMuPDFLoader(app.config['SOURCE_BASE_PATH'] + a_task.source)
        elif a_task.source_type.lower() == "html":
            loader = UnstructuredHTMLLoader(app.config['SOURCE_BASE_PATH'] + a_task.source)
        else:
            loader = TextLoader(app.config['SOURCE_BASE_PATH'] + a_task.source)

        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=a_task.chunk_size, chunk_overlap=a_task.chunk_overlap)
        texts = text_splitter.split_documents(documents)
        a_task.total_chunk = len(texts)
        a_task.save()
        logging.info(f"Task {a_task.id} total chunks: {a_task.total_chunk}")

        llm = SageGPTChatOpenAI(
            openai_api_base=app.config['SAGE_GPT_SERVICE_ADDRESS']
        )
        templ = f"你必须从给定文本中抽取出{a_task.qa_count_per_chunk}" + """组QA对，输出的结果格式如下：
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
输出的结果必须是一个有效的JSON结构，不要输出任何与QA对无关的信息。

给定文本：
{text}
QA对为：
"""
        chain = SageGPTQAGenerationChain.from_llm(llm=llm, prompt=PromptTemplate.from_template(templ))
        i = 1
        has_error = False
        for a_text in texts:
            try:
                logging.info(f"start extract qa from {a_task.id}:{a_task.source} chunk {i}")
                qa, error_count = chain.run({"text": a_text})
                if qa:
                    with open(app.config['DESTINATION_BASE_PATH'] + a_task.destination, 'a') as f:
                        f.write(qa + "\n")
                logging.info(f"finish extract qa from {a_task.id}:{a_task.source} chunk {i}, qa result: {qa}")
                if error_count == 0:
                    a_task: TaskModel = TaskModel.get_by_id(a_task.id)
                    a_task.success_chunk += 1
                    a_task.save()
                else:
                    a_task: TaskModel = TaskModel.get_by_id(a_task.id)
                    a_task.failed_chunk += 1
                    a_task.save()
                    has_error = True
            except Exception as e:
                logging.error(f"finish extract qa from {a_task.id}:{a_task.source} chunk {i} failed: ", e)
                a_task: TaskModel = TaskModel.get_by_id(a_task.id)
                a_task.failed_chunk += 1
                a_task.save()
                has_error = True
            i += 1
        a_task: TaskModel = TaskModel.get_by_id(a_task.id)
        if has_error:
            a_task.status = "Finished(partial failed)"
        else:
            a_task.status = "Finished(completely success)"
        a_task.save()
    except Exception as e:
        logging.info(f"Task {a_task.id} failed")
        logging.error("do_extract_qa failed: ", e)
        try:
            a_task: TaskModel = TaskModel.get_by_id(a_task.id)
            a_task.status = "Failed"
            a_task.save()
        except Exception as e1:
            logging.error("save db failed: ", e1)


@qa_bp.route('/api/v1/qa', methods=['POST'])
def extract_qa():
    post_data = request.get_json()
    logging.info(
        f'user post data: {json.dumps(post_data, ensure_ascii=False)}')

    a_task = TaskModel.create(
        source=post_data['source'],
        source_type=post_data['source_type'].lower(),
        destination=post_data.get('destination', post_data['source'] + ".json"),
        qa_count_per_chunk=post_data.get('qa_count_per_chunk', 5),
        chunk_size=500,
        chunk_overlap=50
    )
    logging.info(f"Task {a_task.id} submitted")
    executor.submit(do_extract_qa, a_task)

    return Response(json.dumps({
        "code": 0,
        "message": f"QA task submitted，task id is {a_task.id}"
    }), content_type='application/json')


@qa_bp.route('/api/v1/status/<task_id>', methods=['GET'])
def get_status(task_id: int):
    logging.info(f'get task {task_id} status')

    try:
        a_task = TaskModel.get_by_id(task_id)
        return Response(json.dumps({
            "code": 0,
            "message": f"Task {task_id} {a_task.status}, "
                       f"total chunks: {a_task.total_chunk}, "
                       f"success chunks: {a_task.success_chunk}, "
                       f"failed chunks: {a_task.failed_chunk}"
        }), content_type='application/json')
    except Exception as e:
        logging.error(f"get task {task_id} status failed: ", e)
        return Response(json.dumps({
            "code": 1,
            "message": f"get status failed, please make sure task id is correct."
        }), content_type='application/json')


def init_database():
    db_file_path = os.path.join(current_dir, '../db/aigq.db')
    app.db = SqliteDatabase(db_file_path, pragmas={'journal_mode': 'wal'})

    TaskModel._meta.database = app.db
    if not TaskModel.table_exists():
        TaskModel.create_table()


def init_config():
    load_dotenv()
    app.config.from_object(Config)


def create_app():
    app.register_blueprint(qa_bp)

    with app.app_context():
        init_config()
    init_log()
    init_database()
    return app


create_app()
if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=app.config['PORT'], use_reloader=True)

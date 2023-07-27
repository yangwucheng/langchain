import os


class Config:
    """基础配置"""
    DEBUG = os.getenv('DEBUG', "False").lower() == "true"
    PORT = int(os.getenv('PORT', "8282"))

    SAGE_GPT_SERVICE_ADDRESS = os.getenv('SAGE_GPT_SERVICE_ADDRESS',
                                         "https://sagegpt-platform.4paradigm.com/chat-engine/api/v1")
    SOURCE_BASE_PATH = os.getenv('SOURCE_BASE_PATH', os.path.dirname(__file__) + "/source-docs/")
    DESTINATION_BASE_PATH = os.getenv('DESTINATION_BASE_PATH', os.path.dirname(__file__) + "/destination-docs/")

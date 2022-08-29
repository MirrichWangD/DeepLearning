import os

environ = os.environ
PROJECT_PATH = os.path.dirname(__file__)
TEMPLATE_FOLDER = os.path.join(PROJECT_PATH, "templates")
STATIC_FOLDER = os.path.join(PROJECT_PATH, "static")
DEBUG = True  # open debug /or hot restart

# ****** 上传配置
UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'sql'}

# 中间文件存储位置
FILE_STORE = '/files'

############################### flask project auto config ######################################
DOWNLOAD_FOLDER = '/tmp'
HOME_PATH = '/'

############################### flask start config ######################################
SECRET_KEY = 'kingsware123456'
HOST = '0.0.0.0'
PORT = 8089

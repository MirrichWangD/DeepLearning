from kingsware.aiam import app
import config
from flask import session


@app.route('/')
def hello_world():
    session.clear()
    return 'Hello World!'


if __name__ == '__main__':
    app.run(config.HOST, config.PORT)

# -*- encoding: utf-8 -*-
"""
    @File        : wmc_demo.py
    @Time        : 2022/8/18 11:44
    @Author      : Mirrich Wang
    @Version     : Python 3.X.X (Anaconda)
    @Description : None
"""

import json
import pymongo
import pandas as pd

from datetime import datetime
from sqlalchemy import create_engine
from pm4py.objects.log.util import dataframe_utils
# ------------------------------------ #
# 数据库信息
# ------------------------------------ #

# service = "MySQL"
# service = "MSSQL"
service = "MongoDB"
sql_dict = {
    "username": "root",
    "password": "root",
    "ip": "localhost",
    "db_name": "kingsware",
}

if service == "MySQL":
    sql_dict.update({
        "port": "3306",
        "sql": "SELECT * FROM receipt;"})

elif service == "MSSQL":
    sql_dict.update({"ip": "175.178.125.212",
                     "port": "1234",
                     "username": "sa",
                     "password": "Mirrich160016",
                     "sql": "SELECT * FROM guest.receipt;"})
else:
    sql_dict.update({"port": "27017",
                     "set_name": "receipt",
                     "filter_rule": "{}, {\"_id\": 0}"})

# --------------------- #
# 读取本地文件
# --------------------- #

username = sql_dict.get('username')
password = sql_dict.get('password')
ip = sql_dict.get("ip")
port = sql_dict.get("port")
db_name = sql_dict.get('db_name')

if service == "MySQL":
    sql = sql_dict.get("sql")
    conn = create_engine(f"mysql+pymysql://{username}:{password}@{ip}:{port}/{db_name}?charset=utf8")
    log_df = pd.read_sql(sql, con=conn)

elif service == "MSSQL":
    sql = sql_dict.get("sql")
    conn = create_engine(f"mssql+pymssql://{username}:{password}@{ip}:{port}/{db_name}?charset=utf8")
    # conn = pymssql.connect(server=ip, port=int(port), user=username, password=password, database=db_name)
    log_df = pd.read_sql(sql, con=conn)

else:  # MongoDB
    set_name = sql_dict.get("set_name")
    filter_rule = sql_dict.get("filter_rule")
    client = pymongo.MongoClient(f"mongodb://%s{ip}:{port}" % (f"{username}:{password}@" if username else ""))
    log_df = eval(f"pd.DataFrame(list(client['{db_name}']['{set_name}'].find({filter_rule})))")

log_df = log_df.fillna("")  # 处理空值以达到标准 json 格式

# ------------------------------------ #

log_df = dataframe_utils.convert_timestamp_columns_in_df(log_df)

print(log_df)
res = [{col: log_df[col].drop_duplicates().to_list()} for col in log_df.columns
       if type(log_df[col][0]) in (int, str, float)]
json.dump(res, open('res.json', 'w+'), indent=4)

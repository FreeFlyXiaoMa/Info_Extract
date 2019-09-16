#!-encoding
# utf-8
# --!

import os
import shutil
import re
import json

"""
封装的方法类

"""

#文件复制函数
def mycopy(srcfile,destfile):
    #print('拷贝pdf文件')
    if not os.path.isfile(srcfile):
        print('%s not exists'%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)  #分割文件路径和名称
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        else:
            print('拷贝pdf%s'%(srcfile))
            shutil.copy(srcfile,destfile)

#数据保存为JSON格式
def store_json(data,file_name):
    json_data=json.dumps(data)
    with open(file_name,'w',encoding='utf-8') as file:
        file.write(json_data)
        file.close()
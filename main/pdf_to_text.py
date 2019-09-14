from pdfminer.pdfparser import PDFParser
from pdfminer.pdfparser import PDFDocument
#from pdfminer.pdfdocument import PDFDocument
#from pdfminer.pdfpage import PDFPage

#from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.layout import *
from pdfminer.converter import PDFPageAggregator

import os
import re


def extract_layout_by_page(pdf_path):

# 提取页面布局

    laparams = LAParams()

    fp = open(pdf_path, 'rb')
    parser = PDFParser(fp)
    document = PDFDocument(parser)

    parser.set_document(document)
    document.set_parser(parser)

    #initialize the documentation
    document.initialize("")

    if not document.is_extractable:
        raise PDFTextExtractionNotAllowed

    rsrcmgr = PDFResourceManager()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    layouts = []
    for page in document.get_pages():
        interpreter.process_page(page)
        layouts.append(device.get_result())

    return layouts

count=0
# translate from pdf to txt with a standard format and save
def pdf_to_txt(filename):

    pdf_path='D:/2019-03_pdf/'+filename

    newname1='D:/Documents/'+str(os.path.splitext(filename)[0]) +str('.txt')
    #pdf有密码的时候，解密后的存放地址
    dest_path='/home/mayajun/Documents/'+filename

    """
    如果在解析过程中报PDFEncryptionError错误，说明pdf有密码，密码为空字符串，需要解密一下
    """
    #call('qpdf --password==%s --decrypt %s %s'%('',pdf_path,dest_path),shell=True)

    page_layouts = extract_layout_by_page(pdf_path) #页面布局列表
    for current_page in page_layouts:
        # 获取文本
        for x in current_page:
            if hasattr(x, "get_text"):
                # result.append(x.get_text())
                results = x.get_text()
                # print(type(results))
                if results == ' \n':
                    continue
                # results=list(results)
                with open(newname1, 'a', encoding='utf-8') as f:
                    f.write(results)
                f.close()
        # 恢复文本原来换行

    with open(newname1, 'r', encoding='UTF-8-sig') as f:
        content = f.readlines()
    pattern = re.compile(r'\s{1,2}\n')  #regular pattern
    newcontent = []
    for i in content:
        if bool(re.search(pattern, i)) is False:
            j = i.replace('\n', '').replace(' \n', '')
            newcontent.append(j)
        else:
            newcontent.append(i)
    f.close()
    with open(newname1, 'w', encoding='UTF-8') as f:
        for i in newcontent:
            f.write(i)
    f.close()
    #config_utils.mycopy(pdf_path,'D:/Documents')
    global count
    count+=1
    #print('抽取了{%d}篇PDF'%(count))
    return newname1



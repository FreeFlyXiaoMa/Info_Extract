import csv
import os


from pdfminer.pdfparser import PDFSyntaxError
from extraction_info.main import pdf_to_text
from extraction_info.allotment_share_issue import return_dic

# "/F/2019-03_pdf/" pdf:14570

with open('D:/NNDemo/extraction_info/03pdf.csv','r',encoding='utf-8-sig') as f:
    reader=csv.reader(f)
    rows=[row for row in reader]
    print(rows[0:2])

pgfx=[] #配股发行

for i in rows:
    if i[1]=='配股预案':
        pgfx.append(i[0])

#print('权益变动列表大小：',len(right_changes))
#-----------------------------------------------------------------------------------------------------------------------

src_path='D:/data_resource/'  #源文件地址
dest_path='D:/Documents/'   #pdf内容抽取后，txt文件存放地址

datas=[]    #存储结果list
i=0
for dir in os.listdir("D:/data_resource/"):   #所有的文件列表
    title=os.path.splitext(dir)[0]  #dir---文件全称
    typ=os.path.splitext(dir)[1]    #type==pdf

    if title in pgfx:  # 配股发行
        try:
            filename = pdf_to_text.pdf_to_txt(dir)
            return_dic(filename,title)
            # print(filename)
        except PDFSyntaxError:
            pass
        except FileNotFoundError:
            pass



#print('文件拷贝了%s份'%(i))

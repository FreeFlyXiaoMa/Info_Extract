# -*- coding: utf-8 -*-
#@Time    :2019/6/24 12:17
#@Author  :XiaoMa
import re
"""
配股预案
"""

def return_dic(filepath,title):
    dic={
        'title':'',
        '公司':'',
        '证券简称':'',
        #'法定代表人':'',
        #'董事会秘书':'',
        #'征集方式':'',
        #'薪酬绩效标准':'',
        #'生效时间':''
    }

    #薪酬方案--
    dic['title']=title
    with open(filepath,encoding='utf-8') as f:
        text=f.read()
        sentences=re.split(' ，|\n|；|。|《|》',text)
        first_line=sentences[0]
        #pattern=re.compile(r'\s.*有限公司',first_line)
        if first_line !='':
            dic['公司']=first_line
        #print('公司==',first_line)

        for i in sentences:
            pattern=re.compile(r'(证券简称：)(\w*)')
            if bool(re.search(pattern,i)) is True:
                value=re.search(pattern,i).group(2)
                #print('证券简称：',value)
                dic['证券简称']=value
            ##################################################################################
            if "薪酬" in title:
                pattern=re.compile(r'(领取薪酬的|所称的董事指|所称的董事是指|激励公司)(.*)')
                if bool(re.search(pattern,i)) is True:
                    value=re.search(pattern,i).group(2)
                    #print('适用对象:',value)
                    dic['方案适用对象']=value
                pattern=re.compile(r'(公司)(\w*)(年度薪酬)')
                if bool(re.search(pattern,i)) is True:
                    value=re.search(pattern,i).group(2)
                    #print('使用对象---',value)
                    dic['方案适用对象']=value

                pattern=re.compile(r'(适用期限为)(.*)')
                if bool(re.search(pattern,i)) is True:
                    value=re.search(pattern,i).group(2)
                    #print('期限',value)
                    dic['方案适用期限']=value
                pattern=re.compile(r'方案适用期限：(.*)(至新的)')
                if bool(re.search(pattern,i)) is True:
                    value=re.search(pattern,i).group(1)
                    value+='起'
                    #print('期限：',value)
                    dic['方案适用期限']=value
                pattern=re.compile(r'(\d.*)年(.*)月(.*)日至(.*)年(.*)月(.*)日')
                if bool(re.search(pattern,i)) is True:
                    value=re.search(pattern,i).group(0)
                    #print('适用期限：',value)
                    dic['方案适用期限']=value
                else:
                    pattern=re.compile(re.compile(r'(，)(每年)(.+?)(月底前)'))
                    if bool(re.search(pattern,i)) is True:
                        value = re.search(pattern, i).group(3)
                        value = '每年' + value + '月底前'
                        dic['方案适用期限']=value

                pattern=re.compile(r'(每年可领取|董事津贴为|标准为人民币)(\d.*万元)')
                if bool(re.search(pattern,i)) is True:
                    value=re.search(pattern,i).group(2)
                    value+=str(r'/每年')
                    #print('薪酬：',value)
                    dic['薪酬标准']=value
            #########################################################################
            else:
                pattern=re.compile(r'法定代表人：(\w*)')
                if bool(re.search(pattern,i)) is True:
                    value=re.search(pattern,i).group(0)
                    #print('法定代表人：',value)
                    dic['法定代表人']=value

                pattern=re.compile('董事会秘书：(\w*)')
                if bool(re.search(pattern,i)) is True:
                    value=re.search(pattern,i).group(1)
                    #print('董事会秘书:',value)
                    dic['董事会秘书']=value

                if '公开征集' in i or '全体股东' in i:
                    dic['征集方式']='公开征集'
                pattern=re.compile('^(\d.*)年(.+?)月(.+?)日(\W)')
                if bool(re.search(pattern,i)) is True:
                    value=re.search(pattern,i).group()
                    if len(value) <20:
                        print('生效时间：', value)
                        dic['生效时间']=value

    return dic


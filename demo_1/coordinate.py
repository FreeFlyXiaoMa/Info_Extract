from pdfminer.pdfparser import PDFParser,PDFDocument,PDFPage
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
#from pdfminer.pdfdocument import PDFDocument
#from pdfminer.pdfpage import PDFPage
#from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.layout import *
from pdfminer.converter import PDFPageAggregator
import matplotlib.pyplot as plt
from matplotlib import patches
from collections import defaultdict
import math
import random
import pandas as pd

"""
PDF表格文字提取
"""
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

TEXT_ELEMENTS = [
    LTTextBox,
    LTTextBoxHorizontal,
    LTTextLine,
    LTTextLineHorizontal
]

def flatten(lst):
    """Flattens a list of lists"""
    return [subelem for elem in lst for subelem in elem]

def extract_characters(element):
    """
    Recursively extracts individual characters from
    text elements.
    """
    if isinstance(element, LTChar):
        return [element]

    if any(isinstance(element, i) for i in TEXT_ELEMENTS):
        return flatten([extract_characters(e) for e in element])

    if isinstance(element, list):
        return flatten([extract_characters(l) for l in element])

    return []

example_file = "D:/2019-03_pdf/2018年度业绩快报.pdf"
page_layouts = extract_layout_by_page(example_file)
print(len(page_layouts))

objects_on_page = set(type(o) for o in page_layouts)
print(objects_on_page)

current_page = page_layouts[0]

texts,rects = [],[]
# seperate text and rectangle elements
for e in current_page:
    if isinstance(e, LTTextBoxHorizontal):
        texts.append(e)
    elif isinstance(e, LTRect):
        rects.append(e)
print(len(texts),texts[0:10])
print(len(rects),rects[0:10])
# # sort them into
characters = extract_characters(texts)
print(len(characters),characters[0:5])


#
# with open('test4.txt', 'a') as f:
#     for i in texts:
#         f.write(str(i)+'\n')

def draw_rect_bbox(x0, y0, x1, y1, ax, color):
    # # Draws an unfilled rectable onto ax.
    ax.add_patch(
        patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            color=color
        )
    )

def draw_rect(rect, ax, color="black"):
    tup=rect.bbox
    x0=tup[0]
    y0=tup[1]
    x1=tup[2]
    y1=tup[3]
    draw_rect_bbox(x0,y0,x1,y1, ax, color)


def width(rect):
    x0, y0, x1, y1 = rect.bbox
    return min(x1 - x0, y1 - y0)


def area(rect):
    x0, y0, x1, y1 = rect.bbox
    return (x1 - x0) * (y1 - y0)


def cast_as_line(rect):

    # 用最长维度的线代替矩形

    x0, y0, x1, y1 = rect.bbox

    if x1 - x0 > y1 - y0:
        return (x0, y0, x1, y0, "H")
    else:
        return (x0, y0, x0, y1, "V")


lines = [cast_as_line(r) for r in rects
         if width(r) < 2 and
         area(r) > 1]
xmin, ymin, xmax, ymax = current_page.bbox
size = 6
fig, ax = plt.subplots(figsize=(size, size * (ymax / xmax)))

for l in lines:
    x0, y0, x1, y1, _ = l
    plt.plot([x0, x1], [y0, y1], 'k-')

# for c in characters:
#     draw_rect(c, ax, "red")

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.show()

def does_it_intersect(x, xmin, xmax):
    return (x <= xmax and x >= xmin)


def find_bounding_rectangle(x, y, lines):
    
    # Given a collection of lines, and a point, try to find the rectangle
    # made from the lines that bounds the point. If the point is not
    # bounded, return None.
    #寻找字符边界

    v_intersects = [l for l in lines
                    if l[4] == "V"
                    and does_it_intersect(y, l[1], l[3])]

    h_intersects = [l for l in lines
                    if l[4] == "H"
                    and does_it_intersect(x, l[0], l[2])]

    if len(v_intersects) < 2 or len(h_intersects) < 2:
        return None

    v_left = [v[0] for v in v_intersects
              if v[0] < x]

    v_right = [v[0] for v in v_intersects
               if v[0] > x]

    if len(v_left) == 0 or len(v_right) == 0:
        return None

    x0, x1 = max(v_left), min(v_right)

    h_down = [h[1] for h in h_intersects
              if h[1] < y]

    h_up = [h[1] for h in h_intersects
            if h[1] > y]

    if len(h_down) == 0 or len(h_up) == 0:
        return None

    y0, y1 = max(h_down), min(h_up)

    return (x0, y0, x1, y1)



box_char_dict = {}
for c in characters:
    # choose the bounding box that occurs the majority of times for each of these:
    bboxes = defaultdict(int)
    l_x, l_y = c.bbox[0], c.bbox[1]
    bbox_l = find_bounding_rectangle(l_x, l_y, lines)
    bboxes[bbox_l] += 1

    c_x, c_y = math.floor((c.bbox[0] + c.bbox[2]) / 2), math.floor((c.bbox[1] + c.bbox[3]) / 2)
    bbox_c = find_bounding_rectangle(c_x, c_y, lines)
    bboxes[bbox_c] += 1

    u_x, u_y = c.bbox[2], c.bbox[3]
    bbox_u = find_bounding_rectangle(u_x, u_y, lines)
    bboxes[bbox_u] += 1

    # 以三个点为基准寻找边框，若相同则边框定
    # 若不相同以中心点边框为准
    # if all values are in different boxes, default to character center.
    # otherwise choose the majority.
    if max(bboxes.values()) == 1:
        bbox = bbox_c
    else:
        bbox = max(bboxes.items(), key=lambda x: x[1])[0]

    if bbox is None:
        continue

    if bbox in box_char_dict.keys():
        box_char_dict[bbox].append(c)
        continue

    box_char_dict[bbox] = [c]

xmin, ymin, xmax, ymax = current_page.bbox
size = 6

fig, ax = plt.subplots(figsize=(size, size * (ymax / xmax)))

for l in lines:
    x0, y0, x1, y1, _ = l
    plt.plot([x0, x1], [y0, y1], 'k-')

# for c in characters:
#     draw_rect(c, ax, "red")

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.show()



for x in range(int(xmin), int(xmax), 10):
    for y in range(int(ymin), int(ymax), 10):
        bbox = find_bounding_rectangle(x, y, lines)

        if bbox is None:
            continue
        if bbox in box_char_dict.keys():
            continue

        box_char_dict[bbox] = []
#
# with open('test5.txt', 'a') as f:
#     for i in box_char_dict.items():
#         f.write(str(i)+'\n')

def chars_to_string(chars):
    
    # 将字符集转化为字符串
    if not chars:
        return ""
    rows = sorted(list(set(c.bbox[1] for c in chars)), reverse=True)
    text = ""
    for row in rows:
        sorted_row = sorted([c for c in chars if c.bbox[1] == row], key=lambda c: c.bbox[0])
        text += "".join(c.get_text() for c in sorted_row)
    return text

def boxes_to_table(box_record_dict):

    # 将单元格-字符 字典转换为行列table
    # of lists of strings. Tries to split cells into rows, then for each row
    # breaks it down into columns.
    #
    boxes = box_record_dict.keys()
    rows = sorted(list(set(b[1] for b in boxes)), reverse=True)
    table = []
    for row in rows:
        sorted_row = sorted([b for b in boxes if b[1] == row], key=lambda b: b[0])
        table.append([chars_to_string(box_record_dict[b]) for b in sorted_row])
    return table

tables=boxes_to_table(box_char_dict)
for table in tables:
    print(table)

# with open('D:/work/pdf/table7.xlsx', 'a') as output:
#     for i in range(len(tables)):
#         for j in range(len(tables[i])):
#             if tables[i][j] is '':
#                 continue
#             x = str(tables[i][j]).replace(' ','')
#             output.write(x)  # write函数不能写int类型的参数，所以使用str()转化
#             output.write('\t')  # 相当于Tab一下，换一个单元格
#         output.write('\n')  # 写完一行立马换行
# output.close()


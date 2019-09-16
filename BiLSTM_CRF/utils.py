import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    PER = get_A_entity(tag_seq, char_seq)
    LOC = get_LOC_entity(tag_seq, char_seq)
    ORG = get_ORG_entity(tag_seq, char_seq)
    return PER, LOC, ORG

def reverse(lines,tags):
    """
    transform demo_sent and predicted_tag into out

    :param lines: [list['','','','',]]
    :param tags: [list['O','B-a'....]]
    :return:str ['3234_23424_54345/o  .....' ,'']
    """
    # list(demo_sent.strip().split('_'))
    results=[]
    for line,tag in zip(lines,tags):
        new_tag=[]
        for i in tag:
            if i=="B-a" or i=="I-a":
                new_tag.append('/a')
            elif i=="B-b" or i=="I-b":
                new_tag.append('/b')
            elif i == "B-c" or i == "I-c":
                new_tag.append('/c')
            else:
                new_tag.append('/o')
        for i in range(len(new_tag)):
            if i!=len(new_tag)-1:
                if new_tag[i]==new_tag[i+1]:
                    new_tag[i]='_'
                else:
                    new_tag[i]+='  '
        out=[]
        for (num,tag) in zip(line,new_tag):
            ou=num+tag
            out.append(ou)
        result=''.join(out)
        results.append(result)
    return results


#
def get_PER_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-PER':
            if 'per' in locals().keys():
                PER.append(per)
                del per
            per = char
            if i+1 == length:
                PER.append(per)
        if tag == 'I-PER':
            per += char
            if i+1 == length:
                PER.append(per)
        if tag not in ['I-PER', 'B-PER']:
            if 'per' in locals().keys():
                PER.append(per)
                del per
            continue
    return PER

def get_A_entity(tag_seq,char_seq):

    result=[]
    Y=list(zip(char_seq,tag_seq[0]))
    A=[]
    for i in range(len(char_seq)-1):
        (char,tag)=Y[i]
        (next_char,next_tag)=Y[i+1]
        if tag=='A':
            a=char
            A.append(a)
            if next_tag!='A':
                A_entity=''.join(A)
                result.append(A_entity)
                A=[]

    return result
def get_B_entity(tag_seq,char_seq):

    result=[]
    Y=list(zip(char_seq,tag_seq[0]))
    A=[]
    for i in range(len(char_seq)-1):
        (char,tag)=Y[i]
        (next_char,next_tag)=Y[i+1]
        if tag=='B':
            a=char
            A.append(a)
            if next_tag!='B':
                A_entity=''.join(A)
                result.append(A_entity)
                A=[]

    return result
def get_B_entity(tag_seq,char_seq):

    result=[]
    Y=list(zip(char_seq,tag_seq[0]))
    A=[]
    for i in range(len(char_seq)-1):
        (char,tag)=Y[i]
        (next_char,next_tag)=Y[i+1]
        if tag=='B':
            a=char
            A.append(a)
            if next_tag!='B':
                A_entity=''.join(A)
                result.append(A_entity)
                A=[]

    return result



def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LOC':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i+1 == length:
                LOC.append(loc)
        if tag == 'I-LOC':
            loc += char
            if i+1 == length:
                LOC.append(loc)
        if tag not in ['I-LOC', 'B-LOC']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_ORG_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ORG':
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org = char
            if i+1 == length:
                ORG.append(org)
        if tag == 'I-ORG':
            org += char
            if i+1 == length:
                ORG.append(org)
        if tag not in ['I-ORG', 'B-ORG']:
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            continue
    return ORG


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

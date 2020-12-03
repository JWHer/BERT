import pandas as pd
import numpy as np
import re
import csv

def preprocess(sent):
    sent_list = []
    if(('.' in sent) | ('?' in sent) | ('!' in sent)):
        if('    ' in sent):
            sent_list.append(sent.split('  ')[2:][0])
            sent_sub = ''.join(sent.split('  ')[2:][1:])
            if('(' in sent_sub):
                sent_sub = re.sub('\)', '.', sent_sub)
                sent_sub = re.sub('\(', '', sent_sub)
                sent_sub = re.sub('\.\.', '.', sent_sub)
                sent_sub = re.sub('\.\.\.', '.', sent_sub)
                sent_sub = re.sub('\.\.\.\.', '.', sent_sub)
                sent_list.extend(sent_sub.split('.'))
            else:
                sent_list.extend(sent_sub.split('.'))
        elif('(' in sent):
            sent = re.sub('\)', '.', sent)
            sent = re.sub('\(', '', sent)
            sent = re.sub('\.\.', '.', sent)
            sent = re.sub('\.\.\.', '.', sent)
            sent = re.sub('\.\.\.\.', '.', sent)
            sent_list.extend(sent.split('.'))
        else:
            sent_list.extend(sent.split('.'))
    
    else:
        sent_list.append(sent)
    
    return sent_list

def listSplit(list, char):
    ret=[]
    for l in list:  
        # 대문자면 그대로
        if l.isupper():
            #if ("INT." in l) or ("EXT." in l):
            #    loc=l.find('.')
            #    ret.append(l[:loc])
            #    ret.append(l[loc+1:])
            #else:
            ret.append(l)
            continue
        # 아니면 구분자로
        ret.extend(l.split(char))
    return ret

def listSDot(list):
    ret=[]
    for l in list:
        if l.isupper():
            if ("INT." in l) or ("EXT." in l):
                loc=l.find('.')
                ret.append(l[:loc])
                ret.append(l[loc+1:])
            else:
                ret.append(l)
            continue
        # 아니면 구분자로
        ret.extend(splitDot(l))
    return ret

def splitDot(mstr):
    ret=[]

    left=''
    right=mstr

    while True:
        c=right.find('.')
        if c==-1:
            ret.append(left+right)
            return ret
        elif (len(right)>c+1 and (right[c+1]=='.' or right[c+1]=='"')):
            left=right[:c+2]
            right=right[c+2:]
            continue
        elif (len(right)>c+2 and right[c+2]=='.'):
            left=right[:c+3]
            right=right[c+3:]
            continue
        else: 
            ret.append(left+right[:c])
            left=''
            right=right[c+1:]  
    return ret

def trimList(list):
    ret=[]
    for l in list:
        if l.isalpha and len(l)>0:
            ret.append(l.strip())
    return ret

def appendLower(list): 
    i=0
    while i<len(list):
        if len(list[i])>0 and list[i][0].islower():
            list[i-1]+=' '+list[i]
            del list[i]
        else:
            i+=1
    return list


data=pd.read_csv("movie_Action.csv");
data.info()

for i in range(231, 232): 
    ex = data.loc[i, 'script']
#print(ex)
#if input("Continue")==n:
#    exit()

    ex_list = ex.split('\n')
    ex_list = listSplit(ex_list, ';')
    ex_list = listSplit(ex_list, ':')
    ex_list = listSplit(ex_list, '. ')#listSDot(ex_list)
    ex_list = trimList(ex_list)
    ex_list = appendLower(ex_list)

    for e in ex_list:
        print(e)
    print(len(ex_list))

    #if input(data.loc[i, 'movie_name']+" Save? ")=='y':

        #script_prepro = '\n'.join(ex_list)
        ##print(script_prepro)
        #data.loc[i, 'script_prepro'] = script_prepro
        #sv=data[i:i+1]
        #sv.to_csv('output'+str(i)+'.csv', index=False)

    with open(str(i)+' '+data.loc[i, 'movie_name']+'.csv', 'w',newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for s in ex_list:
            writer.writerow([s])
    #else: break

print("Done")
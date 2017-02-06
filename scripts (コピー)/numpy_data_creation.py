#!/usr/bin/env python
import numpy as np
import sys
import re
import configparser
import os
#settings
args = sys.argv
inconf=configparser.SafeConfigParser()
inconf.read(args[1])

train_src_file  = inconf.get('settings',"train_src_file")
train_trg_file  = inconf.get('settings',"train_trg_file")
numpy_dir       = inconf.get('settings',"numpy_dir")
trg_dict        = inconf.get('settings',"dict")

src_max_len     = 0
trg_max_len     = 0
#tasks
data_prepare= inconf.get('tasks',"data_prepare")

#SRC numpy data creation
if data_prepare is "1":
    print("#START data preparing")
    data=["train","dev","test"]
    buf=data[0]
    df=open(trg_dict)
    dict={"null":-1}
    trg_dim = sum(1 for line in open(trg_dict))
    I=np.matrix(np.identity(trg_dim))

    for i,w in enumerate(df.readlines()):
        w=re.sub(r'\n',"",w)
        dict.update({w:i})
    print("Load dict done.")

    for d in data:
        buf=d
        train_src_file=re.sub(buf,d,train_src_file)
        train_trg_file=re.sub(buf,d,train_trg_file)
        sf=open(train_src_file)
        tf=open(train_trg_file)
        trg_lines=tf.readlines()
        src_lines=sf.readlines()
        tf.close
        sf.close
        data_num=0
        src_sample=[]
        trg_sample=[]
        filename=""

        for line in src_lines:
            start=r"\["
            end=r"\]"
            if(re.search(start,line)):
                src_sample=[]
                line=re.sub(r'\[',"",line)
                line=re.sub(r' +$',"",line)
                line=re.sub(r'\n',"",line)
                file_name=line
            else:
                line=re.sub(r'^ +',"",line)
                line=re.sub(r' +$',"",line)
                line=re.sub(r'\n',"",line)

                if(re.search(end,line)):
                    #input formulation
                    line=re.sub(r'\ ]',"",line)
                    src_sample.append(line.split(" "))
                    if src_max_len<len(src_sample):
                        src_max_len=len(src_sample)
                    src_sample=np.array(src_sample).astype(np.float32)
                    src_npz=numpy_dir+d+"/src/"+file_name

                    if not os.path.exists(numpy_dir+d+"/src/"):
                        os.mkdir(numpy_dir+d+"/src/")

                    np.save(src_npz,src_sample)
                    data_num+=1
                    src_sample=[]
                    print(file_name)
                else:
                    src_sample.append(line.split(" "))
        print("Src data prepare done.",data_num)


        for line in trg_lines:
            line=re.sub(r'\n',"",line)
            tmp=line.split(" ")
            for i,w in enumerate(tmp):
                if(i>0):
                #print(dict[w])
                    trg_sample.append(I[dict[w]])
                else:
                    file_name=w
            if trg_max_len< len(trg_sample):
                trg_max_len=len(trg_sample)

            trg_sample=np.array(trg_sample).astype(np.float32)
            trg_npz=numpy_dir+d+"/trg/"+file_name
            if not os.path.exists(numpy_dir+d+"/trg/"):
                os.mkdir(numpy_dir+d+"/trg/")
            np.save(trg_npz,trg_sample)
            data_num+=1
            trg_sample=[]
                #lsprint(trg_sample.shape)
        print("Trg data prepare done.")

print("src_max_len", src_max_len)
print("trg_max_len", trg_max_len)

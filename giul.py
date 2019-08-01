import re
import sys
args = sys.argv # 読み込みファイル名の取得
FILE_IN = args[1] # ファイル名の格納

def urlWrite(url):
    with open(FILE_IN+".txt","a") as w:
        w.write(url+"\n")

if __name__ == "__main__": 
    with open(FILE_IN+".html","r") as f:
        pattern = '.+display_url'
        while(True):
            line = f.readline().replace("\n","")
            if re.match(pattern,line):
                break
        line = line.split(",")
        for i in line:
            if re.match(pattern,i):
                url = i.split('"')
                urlWrite(str(url[3]))
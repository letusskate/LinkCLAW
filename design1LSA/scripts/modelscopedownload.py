from modelscope import snapshot_download
import os
os.environ["MODELSCOPE_CACHE"] = "/home/dell/wyb/modelscope/models"  # 例如 "D:/my_models"

cnt = 0
while cnt<5000:
    try:
        model_id = snapshot_download("facebook/contriever-msmarco")
        break
    except:
        cnt = cnt+1
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('download stop, retry cnt=',cnt)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
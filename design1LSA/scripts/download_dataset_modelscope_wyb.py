from modelscope.hub.api import HubApi



from modelscope import snapshot_download
import os
os.environ["MODELSCOPE_CACHE"] = "/home/dell/wyb/modelscope/models"  # 例如 "D:/my_models"

YOUR_ACCESS_TOKEN = 'ms-193e5f47-ee0d-4156-8e77-af7c934b8d4d'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

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
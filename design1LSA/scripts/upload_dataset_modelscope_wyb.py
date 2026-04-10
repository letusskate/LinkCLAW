from pathlib import Path

from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'ms-193e5f47-ee0d-4156-8e77-af7c934b8d4d'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

owner_name = 'WangYubo'
dataset_name = 'wyb_dataset_cache'

api.upload_folder(
    repo_id=f"{owner_name}/{dataset_name}",
    folder_path=str(Path(__file__).resolve().parents[1] / 'dataset_cache'),
    commit_message='upload dataset folder to repo',
    repo_type = 'dataset'
)
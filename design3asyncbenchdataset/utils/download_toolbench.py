import os
import sys
import zipfile
import argparse
import requests
from urllib.parse import urlparse
import time


def download_from_url(url: str, output_path: str, chunk_size: int = 8192) -> bool:
    """
    从URL下载文件，支持断点续传和进度显示
    """
    try:
        headers = {}
        file_size = 0
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            headers['Range'] = f'bytes={file_size}-'
            print(f"Resuming download from {file_size} bytes...")
        
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0)) + file_size
        
        mode = 'ab' if os.path.exists(output_path) else 'wb'
        
        with open(output_path, mode) as f:
            downloaded = file_size
            start_time = time.time()
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        remaining = (total_size - downloaded) / speed if speed > 0 else 0
                        
                        sys.stdout.write(f"\rProgress: {progress:.1f}% | {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB | Speed: {speed / (1024*1024):.1f}MB/s | ETA: {remaining:.1f}s")
                        sys.stdout.flush()
        
        print(f"\nDownload completed: {output_path}")
        return True
        
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False


def download_from_google_drive(file_id: str, output_path: str) -> bool:
    """
    从Google Drive下载文件
    """
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    return save_response_content(response, output_path)


def get_confirm_token(response):
    """获取Google Drive的确认令牌"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, output_path: str, chunk_size: int = 32768) -> bool:
    """保存响应内容到文件"""
    try:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        start_time = time.time()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        remaining = (total_size - downloaded) / speed if speed > 0 else 0
                        
                        sys.stdout.write(f"\rProgress: {progress:.1f}% | {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB | Speed: {speed / (1024*1024):.1f}MB/s | ETA: {remaining:.1f}s")
                        sys.stdout.flush()
        
        print(f"\nDownload completed: {output_path}")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """解压ZIP文件"""
    try:
        print(f"Extracting {zip_path} to {extract_to}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            for i, file in enumerate(file_list, 1):
                zip_ref.extract(file, extract_to)
                if i % 100 == 0 or i == total_files:
                    sys.stdout.write(f"\rExtracted: {i}/{total_files} files")
                    sys.stdout.flush()
        
        print(f"\nExtraction completed!")
        return True
    except Exception as e:
        print(f"\nExtraction failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='ToolBench Dataset Downloader')
    parser.add_argument('--output_dir', type=str, default='./data', 
                        help='Directory to save downloaded data')
    parser.add_argument('--source', type=str, default='manual', 
                        choices=['google', 'tsinghua', 'manual'],
                        help='Download source: google (Google Drive), tsinghua (Tsinghua Cloud), or manual (manual download instructions)')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip download and only extract existing zip')
    parser.add_argument('--skip_extract', action='store_true',
                        help='Skip extraction after download')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    zip_path = os.path.join(args.output_dir, 'data.zip')
    
    # Google Drive file ID for ToolBench data (from official repo)
    google_drive_file_id = '1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk'
    
    # Tsinghua Cloud URL (placeholder - needs confirmation from official repo)
    tsinghua_url = None  # Official URL not confirmed
    
    print("=" * 60)
    print("ToolBench Dataset Downloader")
    print("=" * 60)
    print("\nOfficial Download Sources (from ToolBench GitHub):")
    print("  1. Google Drive: https://drive.google.com/file/d/1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk/view")
    print("  2. Tsinghua Cloud: Check official GitHub for latest link")
    print()
    
    if args.source == 'manual':
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("-" * 60)
        print("Since Google Drive may require login or have regional restrictions,")
        print("please follow these steps:")
        print()
        print("Option 1: Direct Browser Download")
        print("  1. Open: https://drive.google.com/file/d/1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk/view")
        print("  2. Click 'Download' button")
        print("  3. Save as 'data.zip' in this directory")
        print()
        print("Option 2: Using gdown (Python tool)")
        print("  pip install gdown")
        print("  gdown https://drive.google.com/uc?id=1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk")
        print()
        print("Option 3: Using wget")
        print("  wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk&confirm=yes' -O data.zip")
        print()
        print("After downloading, place data.zip in:")
        print(f"  {os.path.abspath(args.output_dir)}")
        print()
        print("Then run this script again with --skip_download to extract")
        print("=" * 60)
        
        if not os.path.exists(zip_path):
            print(f"\nNo data.zip found at {zip_path}")
            print("Please download first, then run with --skip_download")
            return
        else:
            print(f"\nFound existing data.zip at {zip_path}")
            args.skip_download = True
    
    if not args.skip_download:
        if args.source == 'google':
            print(f"\nAttempting Google Drive download...")
            print(f"File ID: {google_drive_file_id}")
            print(f"Output: {zip_path}")
            print()
            
            success = download_from_google_drive(google_drive_file_id, zip_path)
            
            if not success:
                print("\n" + "=" * 60)
                print("Google Drive download failed!")
                print("This may be due to:")
                print("  - Regional restrictions")
                print("  - Login requirements")
                print("  - Network issues")
                print()
                print("Please use manual download instead:")
                print("  1. Visit: https://drive.google.com/file/d/1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk/view")
                print("  2. Download manually")
                print("  3. Run: python download_toolbench.py --skip_download")
                print("=" * 60)
                return
        elif args.source == 'tsinghua':
            if not tsinghua_url:
                print("\nTsinghua Cloud URL not available!")
                print("Please check the official ToolBench GitHub for the latest link")
                print("or use manual download from Google Drive")
                return
            
            print(f"\nDownloading from Tsinghua Cloud...")
            print(f"URL: {tsinghua_url}")
            print(f"Output: {zip_path}")
            print()
            
            success = download_from_url(tsinghua_url, zip_path)
            
            if not success:
                print("\nTsinghua Cloud download failed!")
                print("Please try manual download or check GitHub for updated link")
                return
    else:
        if not os.path.exists(zip_path):
            print(f"Error: {zip_path} not found!")
            print(f"Please download data.zip first and place it in {args.output_dir}/")
            return
        print(f"Skipping download, using existing file: {zip_path}")
    
    if not args.skip_extract:
        print("\n" + "=" * 60)
        print("Extracting dataset...")
        print("=" * 60)
        
        if extract_zip(zip_path, args.output_dir):
            print("\n" + "=" * 60)
            print("Dataset ready!")
            print("=" * 60)
            print(f"\nData directory: {os.path.abspath(args.output_dir)}")
            print("\nYou can now use generate_async_toolbench.py to process this data!")
    else:
        print("\nSkipping extraction.")


if __name__ == "__main__":
    main()

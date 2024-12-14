from PIL import Image
import io
import base64
import requests

url = 'http://localhost:5000/post'
files = {'file': open('process/darkFace.png', 'rb')}
# files = {'file': open('origin.jpg', 'rb')}
response = requests.post(url, files=files)

try:
    # 解析返回的 JSON 数据
    data = response.json()
    
    if data['code'] == 200:
        # 获取 base64 编码的图像数据
        base64_img1 = data['data']['img1']
        base64_img2 = data['data']['img2']
        base64_img3 = data['data']['img3']
        base64_img4 = data['data']['img4']
        
        # 解码 base64 数据并转换为图像
        data_img1 = base64.b64decode(base64_img1)
        data_img2 = base64.b64decode(base64_img2)
        data_img3 = base64.b64decode(base64_img3)
        data_img4 = base64.b64decode(base64_img4)

        img1 = Image.open(io.BytesIO(data_img1))
        img2 = Image.open(io.BytesIO(data_img2))
        img3 = Image.open(io.BytesIO(data_img3))
        img4 = Image.open(io.BytesIO(data_img4))
        
        # 显示图像
        img1.show()
        img2.show()
        img3.show()
        img4.show()
    else:
        print(f"Error: {data['msg']}")
except requests.exceptions.JSONDecodeError:
    print("Error: Response is not in JSON format")
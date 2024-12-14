from flask import Flask, request, jsonify
import cv2
import face_recognition
import numpy as np
import base64
from process.bgrHistogramEqualization import invoke as bgrInvoke
from process.faceLightTransform import invoke as faceInvoke
from process.illuminationNormalization import invoke as illuminationInvoke
from process.lightCompensation import invoke as lightInvoke

app = Flask(__name__)

@app.route('/post', methods=['POST'])
def process_pic():
    if 'file' not in request.files:
        return jsonify({'code': 400, 'msg': 'No file part', 'data': {}})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'code': 400, 'msg': 'No selected file', 'data': {}})
    
    # 读取文件并转换为 OpenCV 图像
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


    ### 人脸捕获

    # 检测人脸位置
    face_locations = face_recognition.face_locations(img)

    if len(face_locations) == 0:
        return jsonify({'code': 400, 'msg': 'No face detected', 'data': {}})

    top, right, bottom, left = face_locations[0]
    # 裁剪人脸
    img = img[top:bottom, left:right]


    ### 算法处理

    # 对图像进行直方图均衡化
    img1 = bgrInvoke(img)

    # 对图像进行人脸光照变换
    img2 = faceInvoke(img)

    # 对图像进行光照归一化
    img3 = illuminationInvoke(img)

    # 对图像进行不均匀光照补偿
    img4 = lightInvoke(img)


    ### 返回结果
    
    # 将处理后的图像编码为 PNG 格式
    _, buffer = cv2.imencode('.png', img1)
    res_img1 = buffer.tobytes()
    _, buffer = cv2.imencode('.png', img2)
    res_img2 = buffer.tobytes()
    _, buffer = cv2.imencode('.png', img3)
    res_img3 = buffer.tobytes()
    _, buffer = cv2.imencode('.png', img4)
    res_img4 = buffer.tobytes()

    # 将图像转换为 base64 编码
    res_base64_img1 = base64.b64encode(res_img1).decode('utf-8')
    res_base64_img2 = base64.b64encode(res_img2).decode('utf-8')
    res_base64_img3 = base64.b64encode(res_img3).decode('utf-8')
    res_base64_img4 = base64.b64encode(res_img4).decode('utf-8')
    
    return jsonify({
        'code': 200, 
        'msg': '', 
        'data': {
            'img1': res_base64_img1,
            'img2': res_base64_img2,
            'img3': res_base64_img3,
            'img4': res_base64_img4
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
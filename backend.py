"""
植物病害本地初筛服务
使用ResNet50模型对上传的图片进行病害分类
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import json
import os

# ==================== 配置 ====================
MODEL_PATH = 'best_model.pth'  # 模型路径
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 类别名称映射
CLASS_NAMES = [
    'Anthracnose',       # 炭疽病
    'Brown_Stem_Spot',   # 褐斑病
    'Gray_Blight',       # 灰霉病
    'Soft_Rot',          # 软腐病
    'Stem_Canker',       # 茎溃疡病
    'Healthy'            # 健康
]

# 中文名称映射
CLASS_NAMES_CN = {
    'Anthracnose': '炭疽病',
    'Brown_Stem_Spot': '褐斑病',
    'Gray_Blight': '灰霉病',
    'Soft_Rot': '软腐病',
    'Stem_Canker': '茎溃疡病',
    'Healthy': '健康'
}

# ==================== 模型定义 ====================
def create_model(num_classes=6):
    """创建ResNet50模型"""
    model = models.resnet50(weights=None)  # 不加载预训练权重
    
    # 替换分类头
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model

# ==================== 加载模型 ====================
print("正在加载模型...")
model = create_model(len(CLASS_NAMES))

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型加载成功！验证准确率: {checkpoint.get('val_acc', 'N/A')}")
else:
    print(f"警告: 模型文件 {MODEL_PATH} 不存在，请先运行 main.py 训练模型")
    print("将使用随机初始化的模型进行演示")

model = model.to(DEVICE)
model.eval()

# 图片预处理
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# ==================== 推理函数 ====================
def predict(image_bytes):
    """对图片进行推理"""
    try:
        # 打开图片
        image = Image.open(io.BytesIO(image_bytes))
        
        # 预处理
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # 推理
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # 获取结果
        probs = probabilities.cpu().numpy()
        predicted_class = CLASS_NAMES[probs.argmax()]
        
        # 返回top-k结果
        top_k = 3
        top_indices = probs.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'class': CLASS_NAMES[idx],
                'class_cn': CLASS_NAMES_CN[CLASS_NAMES[idx]],
                'confidence': float(probs[idx])
            })
        
        return {
            'success': True,
            'predicted_class': predicted_class,
            'predicted_class_cn': CLASS_NAMES_CN[predicted_class],
            'confidence': float(probs.max()),
            'top_predictions': results
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# ==================== Flask API ====================
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许跨域

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """接收base64编码的图片，返回预测结果"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'success': False, 'error': '缺少image字段'}), 400
        
        # 解码base64图片
        image_data = base64.b64decode(data['image'])
        
        # 推理
        result = predict(image_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'ok', 'model_loaded': os.path.exists(MODEL_PATH)})

if __name__ == '__main__':
    print("=" * 50)
    print("植物病害本地初筛服务")
    print(f"设备: {DEVICE}")
    print(f"模型: {MODEL_PATH}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)

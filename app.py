from flask import Flask, request, jsonify
from ultralytics import YOLO
from ocr_with_preprocessing import *

app = Flask(__name__)
# 预加载模型
# model_weights = "/usr/src/ultralytics/detect/side_detect_v3.pt"
model_weights = "./detect/side_detect_v4.onnx"
model = YOLO(model_weights, task='detect')
model_loaded = True


# 确保模型只在需要时加载的辅助函数
def get_model():
    """获取预加载的模型实例。"""
    global model, model_loaded
    if not model_loaded:
        model = YOLO(model_weights)  # 如果模型未加载，这里可以添加异常处理或日志记录
        model_loaded = True
    return model


@app.route('/ocr', methods=['POST'])
def ocr():
    model = get_model()

    # 从请求中获取图片路径
    data = request.json
    img_path = data.get('img_path')
    if not img_path:
        return jsonify({'error': 'No img_path provided'}), 400

    # OCR 和 YOLO 模型配置
    trained_data_path = "./detect/chi_sim.traineddata"
    whitelist_characters = "双喜软硬经典1906百年春天中细支工坊红五叶神莲香盛世蓝玫王金逸品紫椰树国花悦勿忘我魅影世纪尊()（）"
    # pattern = r"(?:双\s*喜|椰\s*树|红\s*玫)\s*\([^)]+?\)?"
    brand_list = ["双喜(软经典)", "双喜(软经典1906)", "双喜(百年经典)", "双喜(春天细支)", "双喜(春天中支)",
                  "双喜(经典工坊)", "双喜(软红五叶神)", "双喜(软01)", "双喜(莲香)", "双喜(盛世)", "双喜(软蓝红玫王)",
                  "双喜(硬)", "双喜(硬01)", "双喜(硬蓝红玫王)", "双喜(硬经典1906)", "双喜(硬经典)", "双喜(硬金五叶神)",
                  "双喜(硬红五叶神)", "双喜(硬逸品)", "双喜(硬紫红玫王)", "双喜(花悦)", "双喜(国喜细支)",
                  "双喜(勿忘我)",
                  "双喜(春天魅影)", "双喜(金01)", "双喜(硬世纪经典)", "双喜(五叶神金尊)", "椰树(硬)", "红玫(硬金)"]

    # 进行检测
    results = model(img_path, conf=0.5, classes=2, imgsz=1024)
    boxes_obj = results[0].boxes

    # 过滤标签坐标
    label_class_id = [k for k, v in results[0].names.items() if v == 'label'][0]
    label_boxes_xyxy = [box for i, box in enumerate(boxes_obj.xyxy) if boxes_obj.cls[i] == label_class_id]

    # 抠出标签并进行 OCR
    extracted_images = extract_subimages(img_path, label_boxes_xyxy)
    ocr_results = ocr_on_extracted_images(extracted_images, trained_data_path, whitelist_characters, brand_list)

    # 返回第一个 OCR 结果
    return jsonify({'data': {'ocr_result': ocr_results[0]}, 'message': "请求成功", 'code': 200})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

import time
from PIL import Image
import cv2
import numpy as np
import pytesseract
import os
import re
from fuzzywuzzy import process, fuzz
from scipy.spatial import ConvexHull

os.environ['PATH'] += os.pathsep + '/opt/local/bin'


# 预处理
def preprocess_image(img_cv):
    img_gray_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # image = cv2.equalizeHist(img_gray_cv)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # SECTION 调参
    # # 应用 CLAHE 算法
    # image = clahe.apply(img_gray_cv)
    image = img_gray_cv
    img_blur = cv2.GaussianBlur(image, (3, 3), 0)

    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_thresh


def ocr_with_preprocessing(img_cv, trained_data_path, whitelist_characters):
    os.environ['TESSDATA_PREFIX'] = os.path.dirname(trained_data_path)

    img_thresh = preprocess_image(img_cv)
    # CV - PIL
    img_preprocessed = Image.fromarray(cv2.cvtColor(img_thresh, cv2.COLOR_BGR2RGB))

    # OCR psm = 1, 7, 13 可
    custom_oem_psm_config = r'--oem 2 --psm 1 -c tessedit_char_whitelist=' + whitelist_characters  # SECTION 调参
    text = pytesseract.image_to_string(img_preprocessed, lang='chi_sim', config=custom_oem_psm_config)

    return text


def weighted_score(match, base_score, weight=0.1):
    # 字长权重
    length_component = weight * len(match)
    return base_score + length_component


def ocr_on_extracted_images(extracted_images, trained_data_path, whitelist_characters, brand_list):
    highest_score_match = ""
    highest_score = 0

    for img_cv in extracted_images:
        # 清晰度增强
        img_cv = cv2.resize(img_cv, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        try:
            # 尝试做透视变换
            img_transformed = perspective_transform(img_cv)
            if img_transformed is None:
                print("Perspective transform returned None. Using the original image for OCR.")
                img_transformed = img_cv
        except Exception as e:
            # 透视变换失败
            print(f"Perspective transform failed: {e}")
            img_transformed = img_cv  # 使用原图

        result = ocr_with_preprocessing(img_transformed, trained_data_path, whitelist_characters)

        for line in result.split('\n'):
            # 清理每一行以删除不需要的字符
            clean_line = ''.join(filter(lambda char: char in whitelist_characters, line.strip()))

            # 忽略清理后的空行
            if clean_line:
                # 使用模糊匹配找到每一行的最接近匹配
                matches = process.extract(clean_line, brand_list, scorer=fuzz.token_sort_ratio)
                # 对匹配结果进行后处理，以选择最佳匹配
                for match, score in matches:
                    # 使用加权分数计算最终分数
                    final_score = weighted_score(match, score)
                    # 更新最高分和匹配
                    if final_score > highest_score:
                        highest_score = final_score
                        highest_score_match = match

    # 只返回评分最高的匹配项和得分
    return highest_score_match, highest_score if highest_score_match else (None, 0)


def find_largest_four_corners(corners):
    # If there are exactly four corners, no need to process further
    if len(corners) == 4:
        return corners
    # Convert corners to a numpy array for processing
    points = np.array([point[0] for point in corners])

    # Using ConvexHull to find the convex hull of the points
    hull = ConvexHull(points)

    # Extracting the points forming the convex hull
    hull_points = points[hull.vertices]

    # Cluster points based on distance using DBSCAN
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=10, min_samples=1).fit(hull_points)
    labels = clustering.labels_

    # Gather points by labels
    clustered_points = {}
    for label, point in zip(labels, hull_points):
        if label in clustered_points:
            clustered_points[label].append(point)
        else:
            clustered_points[label] = [point]

    # Find centroid of the clusters to represent the "corner" and calculate areas
    cluster_centroids = []
    cluster_areas = []
    for label, group in clustered_points.items():
        group = np.array(group)
        centroid = np.mean(group, axis=0)
        if len(group) > 1:  # If cluster has more than one point, find convex hull area
            cluster_hull = ConvexHull(group)
            area = cluster_hull.volume
        else:  # Single point has no area but include it with zero area
            area = 0
        cluster_centroids.append(centroid)
        cluster_areas.append(area)

    # Sort centroids based on area in descending order
    sorted_centroids = [x for _, x in
                        sorted(zip(cluster_areas, cluster_centroids), key=lambda pair: pair[0], reverse=True)]

    # Select the top four centroids with the largest areas
    selected_corners = sorted_centroids[:4]

    # Convert selected corners to the required format (list of lists of lists)
    final_corners = [[[int(x[0]), int(x[1])]] for x in selected_corners]
    final_corners = np.array(final_corners)
    return final_corners


def perspective_transform(img_cv, epsilon_lam=0.02):
    #  从路径加载
    # image = Image.open(image_path)
    # image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    img_thresh = preprocess_image(img_cv)
    # 轮廓检测
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 取最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 近似轮廓为多边形，寻找角点
    epsilon = epsilon_lam * cv2.arcLength(largest_contour, True)  # SECTION 调参
    corners = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 没有4个角点则返回None或者最大凸包
    if len(corners) > 4:
        corners = find_largest_four_corners(corners)
    elif len(corners) < 4:
        return None

    # cv2.drawContours(img_d, [corners], -1, (0, 255, 255), 5)
    # 角点顺序: [左上, 右上, 右下, 左下]
    corners = corners.reshape((4, 2))
    ordered_corners = np.zeros((4, 2), dtype=np.float32)
    add = corners.sum(1)
    ordered_corners[0] = corners[np.argmin(add)]
    ordered_corners[2] = corners[np.argmax(add)]
    diff = np.diff(corners, axis=1)
    ordered_corners[1] = corners[np.argmin(diff)]
    ordered_corners[3] = corners[np.argmax(diff)]

    # 计算透视转换后的大小
    width = max(np.linalg.norm(ordered_corners[0] - ordered_corners[1]),
                np.linalg.norm(ordered_corners[2] - ordered_corners[3]))
    height = max(np.linalg.norm(ordered_corners[0] - ordered_corners[3]),
                 np.linalg.norm(ordered_corners[1] - ordered_corners[2]))

    # 构建目标点
    dst = np.array([[0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]], dtype="float32")

    # 计算透视转换矩阵
    matrix = cv2.getPerspectiveTransform(ordered_corners, dst)
    warped = cv2.warpPerspective(img_cv, matrix, (int(width), int(height)))

    # 裁剪上半部分
    top_half_height = warped.shape[0] // 4
    top_half = warped[top_half_height - 20:top_half_height * 2 + 10, :, :]
    # top_half = cv2.cvtColor(top_half, cv2.COLOR_BGR2RGB)
    # CV -> PIL
    # warped_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    return top_half


def extract_subimages(img_path, label_boxes_xyxy):
    # Load the original image
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sub_images = []  # List to hold the extracted sub-images

    # Extract sub-images
    for box in label_boxes_xyxy:
        x1, y1, x2, y2 = map(int, box)
        sub_image = image_rgb[y1:y2, x1:x2]
        sub_images.append(sub_image)

    return sub_images


if __name__ == "__main__":
    image_path = "/Users/apple/work/智能盘点/Screenshot 2023-11-01 at 17.16.22.png"
    # trained_data_path = "/Users/apple/work/智能盘点/chi_sim.traineddata"
    trained_data_path = '/Users/apple/work/智能盘点/chi_sim.traineddata'
    whitelist_characters = "双喜软硬经典1906百年春天中细支工坊红五叶神莲香盛世蓝玫王金逸品紫椰树国花悦勿忘我魅影世纪尊()"
    img_transformed = perspective_transform(image_path)
    t1 = time.time()
    result = ocr_with_preprocessing(img_transformed, trained_data_path, whitelist_characters)
    t2 = time.time()
    print("OCR Result:", result)
    print("time cost:", t2 - t1)
    # img_preprocessed.show()

    # 构建牌号正则表达式模式
    pattern = r"(?:双\s*喜|椰\s*树|红\s*玫)\s*\([^)]+\)"

    # 从 OCR 结果中提取牌号
    matches = re.findall(pattern, result)
    cleaned_matches = [match.replace(" ", "") for match in matches]

    print(cleaned_matches)

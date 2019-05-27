import sys
import os
import cv2
import numpy as np

original_image = None
valid_rects = None

def car_number_detection():
    image_file_name = 'original.png'

    global original_image
    original_image = cv2.imread(image_file_name)

    gray_image = gray_scale(original_image)
    save_image(gray_image, image_file_name, 'gray')

    threshold_image = adaptive_threshold(gray_image)
    save_image(threshold_image, image_file_name, 'threshold')

    (contours, contour_image) = get_contours(threshold_image)
    save_image(contour_image, image_file_name, 'contour')

    (rects, rect_contour_image) = rect_contours(contours)
    save_image(rect_contour_image, image_file_name, 'rect')

    global valid_rects
    (valid_rects, valid_rect_image) = validate_rect(rects)
    save_image(valid_rect_image, image_file_name, 'valid-rect')

    result_idxs = validate_rect_group(valid_rects)

    detection_image = detection(result_idxs)
    save_image(detection_image, image_file_name, 'detection')

## 이미지 저장
# image : source image
# image_file_name : image name
# middle_name : image's middle name
def save_image(image, image_file_name, middle_name):
    image_name, image_extension = os.path.splitext(image_file_name)
    cv2.imwrite(image_name + '-' + middle_name + image_extension, image)

## temp image를 생성해 주는 함수
# @return: original_image와 똑같은 사이즈의 temp image
def create_temp_image():
    # original_image 의 크기를 가져옴
    global original_image
    height, width, channel = original_image.shape

    # 이미지 생성을 위해서 이미지 크기의 빈 array 선언
    temp = np.zeros((height, width, channel), dtype=np.uint8)

    return temp

## 이미지 흑백으로 변경
# image : source image
# @return gray scale image
def gray_scale(image):
    # 색 변경. gray scale로 변경
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

## 이미지를 임계치 값으로 변경
# image : source image
# @return thresholded image
def adaptive_threshold(image):
    # 노이즈 제거
    blur = cv2.GaussianBlur(image, ksize=(5,5), sigmaX=0)

    # 이미지의 threshold 설정
    return cv2.adaptiveThreshold(
        blur,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )

## 이미지의 윤곽선을 찾아줌
# image : source image
# @return (contour list, contour image)
def get_contours(image):
    # 윤곽선 찾기
    contours, _ = cv2.findContours(
        image,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    # 빈 이미지 생성
    contour_image = create_temp_image()

    # 윤곽선을 그려줌
    cv2.drawContours(contour_image, contours=contours, contourIdx=-1, color=(255, 255, 255))

    return contours, contour_image

## 윤곽선을 사각형 모양으로 그리기 위한 함수
# contours: 윤곽선 목록
# @return: 사각형 목록, 사각형 이미지
def rect_contours(contours):
    # 사각형의 위치 정보를 저장하기 위해 선언
    rects = []
    # 이미지 저장을 위한 이미지 생성
    rect_contour_image = create_temp_image()

    for contour in contours:
        # 윤곽선의 x, y 좌표, 폭, 높이를 가져옴
        x, y, w, h = cv2.boundingRect(contour)
        
        # 이미지에 사각형을 그려줌
        cv2.rectangle(rect_contour_image, pt1=(x,y), pt2=(x+w,y+h), color=(255,255,255), thickness=2)

        # 사각형 정보를 넣어줌
        # cx: x좌표의 중심, cy: y 좌표의 중심
        rects.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    return rects, rect_contour_image

## 사각형 중 유효한 사각형를 추출
# rects : 사각형 목록
# @return 유효한 사각형 목록, 유효한 사각형 이미지
def validate_rect(rects):
    # 사각형의 최소 넓이
    MIN_AREA = 80
    # 사각형의 최소 폭, 높이
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    # 사각형의 최소, 최대 가로 세로 비율
    MIN_RATIO, MAX_RATIO = 0.25, 1.0
    # 유효한 사각형 목록
    valid_rects = []
    # 유효한 사각형에 부여되는 index
    idx = 0
    # 이미지 저장을 위한 이미지 생성
    valid_rect_image = create_temp_image()

    for rect in rects:
        # 넓이
        area = rect['w'] * rect['h']
        # 비율
        ratio = rect['w'] / rect['h']

        if area > MIN_AREA \
        and rect['w'] > MIN_WIDTH \
        and rect['h'] > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            # 인덱스를 부여하고 valid_rects에 추가
            rect['idx'] = idx
            idx += 1
            valid_rects.append(rect)
            # 사각형 추가
            cv2.rectangle(valid_rect_image, pt1=(rect['x'], rect['y']), pt2=(rect['x']+rect['w'], rect['y']+rect['h']), color=(255,255,255), thickness=2)

    return valid_rects, valid_rect_image

## 유효한 사각형 그룹을 가져오는 함수, recursive function
# rects : 사각형 목록
# @return 유효한 사각형 그룹의 목록
def validate_rect_group(rects):
    # 사각형의 대각선 길이의 5배가 최대 간격
    MAX_DIAG_MULTIPLYER = 5
    # 사각형의 중심 최대 각도
    MAX_ANGLE_DIFF = 12.0
    # 사각형의 면적 차이
    MAX_AREA_DIFF = 0.5
    # 사각형의 넓이 차이
    MAX_WIDTH_DIFF = 0.8
    # 사각형의 높이 차이
    MAX_HEIGHT_DIFF = 0.2
    # 사각형의 그룹의 최소 갯수
    MIN_N_MATCHED = 3

    matched_result_idxs = []

    for rect1 in rects:
        matched_rect_idxs = []

        for rect2 in rects:
            if rect1['idx'] == rect2['idx']:
                continue

            # 각을 구하기 위한 중심 거리 계산
            dx = abs(rect1['cx'] - rect2['cx'])
            dy = abs(rect1['cy'] - rect2['cy'])

            # 각 계산
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy/dx))

            # rect1의 대각선 길이
            diagonal1 = np.sqrt(rect1['w'] ** 2 + rect1['h'] ** 2)

            # 중심 간격
            distance = np.linalg.norm(np.array([rect1['cx'], rect1['cy']]) - np.array([rect2['cx'], rect2['cy']]))

            # 면적 비율
            rect1_area = rect1['w'] * rect1['h']
            rect2_area = rect2['w'] * rect2['h']
            area_diff = abs(rect1_area - rect2_area) / rect1_area

            # 폭의 비율
            width_diff = abs(rect1['w'] - rect2['w']) / rect1['w']

            # 높이의 비율
            height_diff = abs(rect1['h'] - rect2['h']) / rect1['h']

            # 조건 확인
            if distance < diagonal1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF \
            and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF \
            and height_diff < MAX_HEIGHT_DIFF:
                matched_rect_idxs.append(rect2['idx'])

        # rect1도 넣어준다.
        matched_rect_idxs.append(rect1['idx'])

        # rect group이 기준 이하면 결과에 포함하지 않음
        if len(matched_rect_idxs) < MIN_N_MATCHED:
            continue
        else:
            # 결과에 포함
            matched_result_idxs.append(matched_rect_idxs)
            
            # 매칭이 안된 것끼리 다시 진행
            unmatched_rect_idxs = []

            for rect in rects:
                if rect['idx'] not in matched_rect_idxs:
                    unmatched_rect_idxs.append(rect['idx'])

            global valid_rects
            unmatched_rect = np.take(valid_rects, unmatched_rect_idxs)

            # recursive call
            recursive_rect_list = validate_rect_group(unmatched_rect)

            # recursive 결과 취합
            for idx in recursive_rect_list:
                matched_result_idxs.append(idx)

            break
    
    return matched_result_idxs

## 최종적으로 detection하여 비식별화하기 위한 함수
# result_idxs : 최종적으로 선택된 group list
# @return 비식별 처리 된 image
def detection(result_idxs):
    global valid_rects
    global original_image
    # 최종 사각형 저장하기 위한 배열
    result_group = []

    for idx in result_idxs:
        result_group.append(np.take(valid_rects, idx))

    for group in result_group:
        min_x, min_y = sys.maxsize, sys.maxsize
        max_x, max_y = sys.maxsize * -1, sys.maxsize * -1

        for rect in group:
            min_x = min_x if min_x < rect['x'] else rect['x']
            min_y = min_y if min_y < rect['y'] else rect['y']
            max_x = max_x if max_x > rect['x']+rect['w'] else rect['x']+rect['w']
            max_y = max_y if max_y > rect['y']+rect['h'] else rect['y']+rect['h']

        cv2.rectangle(original_image, pt1=(min_x, min_y), pt2=(max_x, max_y), color=(0, 0, 0), thickness=cv2.FILLED)

    return original_image
            

if __name__ == '__main__':
    car_number_detection()


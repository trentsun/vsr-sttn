import cv2

def get_subtitle_area(video_path):
    # 打开视频并读取一帧
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        return None
    
    # 显示帧并等待用户选择区域
    roi = cv2.selectROI("Select Subtitle Area", frame)
    cv2.destroyAllWindows()
    
    # 转换ROI格式为 (ymin, ymax, xmin, xmax)
    xmin, ymin, width, height = roi
    xmax = xmin + width
    ymax = ymin + height
    
    return (ymin, ymax, xmin, xmax)

# 使用示例
if __name__ == '__main__':
    video_path = input("Please input video file path: ").strip()
    sub_area = get_subtitle_area(video_path)
    print(sub_area)
    segments_file = 'segments.json'
    # SubtitleRemover(video_path, sub_area=sub_area, segments_file=segments_file).run()
import cv2
import os

# 设置摄像头的编号，0通常是默认的内置摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# 创建保存图片的目录
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

# 初始化图片编号
image_counter = 1

try:
    while True:
        # 捕获摄像头的一帧
        ret, frame = cap.read()

        # 如果正确地读取了帧，ret为True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # 调整帧的大小
        frame_resized = cv2.resize(frame, (420, 420), interpolation=cv2.INTER_AREA)

        # 构建图片的保存路径，使用递增的序号命名图片
        file_path = os.path.join(save_dir, f"{image_counter}.jpg")
        image_counter += 1  # 更新图片编号

        # 保存图片
        cv2.imwrite(file_path, frame_resized)

        # 显示图片
        cv2.imshow('frame', frame_resized)

        # 按'q'退出循环
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
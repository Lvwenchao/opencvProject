# write by Mrlv
# coding:utf-8
import cv2

vedio = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv2.VideoWriter(r'E:\pythonProject\opencvProject\resources\output.avi', fourcc, 20.0, (640, 400))
while True:
    ret, frame = vedio.read()
    if ret:
        frame = cv2.flip(frame, 0)
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    else:
        break

vedio.release()
out.release()
cv2.destroyAllWindows()

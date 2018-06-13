import numpy as np
import cv2
from sklearn import linear_model, datasets
from numpy.linalg import lstsq
import serial

video="/home/usuario/ManuelaReto3/Servo/Video/output.avi"

kernel = np.ones((15,15),np.uint8)

cap = cv2.VideoCapture(0)

def ransac(points):
    pixels = np.argwhere(points == 255)
    x = pixels[:, 0]
    y = pixels[:, 1]
    x = np.transpose([x])
    y = np.transpose([y])

    ransac = linear_model.RANSACRegressor()
    ransac.fit(x, y)
    inliers = ransac.inlier_mask_
    line_x = np.arange(x.min(), x.max())[:, np.newaxis]
    line_y = ransac.predict(line_x)

    points = [(line_x[1][0], line_y[1][0]), (line_x[2][0], line_y[2][0])]
    p_x, p_y = zip(*points)
    A0 = np.vstack([p_x, np.ones(len(p_x))]).T
    m, b = lstsq(A0, p_y, rcond=None)[0]

    return(m,b,inliers)

def drawline(m,b):
    if b > 0:
        cv2.line(frame, (int((rows * m) + b), rows), (int(b), 0), (255, 255, 0), 2)
    elif b < 0:
        cv2.line(frame, (0, int(-1 * b / m)), (cols, int((cols - b) / m)), (255, 255, 0), 2)


ser = serial.Serial('/dev/ttyACM0', 115200)


while(cap.isOpened()):
    ret, frame = cap.read()

    if ret==True:
        detecta = False
        #frame = cv2.flip(frame,0)
        rows, cols, _ = frame.shape

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([45, 50, 100])
        upper_green = np.array([80, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        closing = cv2.dilate(mask, kernel, iterations=2)
        points = cv2.erode(closing, kernel, iterations=2)
        res = cv2.bitwise_and(frame, frame, mask=points)

        pixels = np.argwhere(points == 255)
        x = np. transpose(pixels[:, 0])
        y = np. transpose(pixels[:, 1])

        if len(pixels)>1000:
            detecta=True
            #RANSAC
            #PRIMERA LINEA
            m1,b1,inliers1=ransac(points)

            points[x[inliers1], y[inliers1]] = 0

            # SEGUNDA LINEA
            pixels = np.argwhere(points == 255)
            x = np.transpose(pixels[:, 0])
            y = np.transpose(pixels[:, 1])

            if len(pixels) > 500:

                m2, b2, inliers2 = ransac(points)

                interseccionx = int((b2 - b1) / (m1 - m2))
                intersecciony = int((m2 * b1 - m1 * b2) / (m2 - m1))

                centro = int(cols / 2)
                error = intersecciony - centro
                print(error)

                drawline(m1,b1)
                drawline(m2,b2)
                cv2.line(frame, (intersecciony, 0), (intersecciony, 4608), (255, 0, 255), 2)
                cv2.line(frame, (int(cols/2), 0), (int(cols/2), rows), (0, 0, 255), 2)

                if -10<error<10:
                    dir = 'q'
                    ser.write(dir.encode("utf-8"))

                elif -10>error:
                    if error<(400):
                        dir = 'q'
                        ser.write(dir.encode("utf-8"))
                    else:
                        dir = 'd'
                        ser.write(dir.encode("utf-8"))
                        print('derecha')

                elif error>10:
                    if error>(400):
                        dir = 'q'
                        ser.write(dir.encode("utf-8"))
                    else:
                        dir = 'i'
                        ser.write(dir.encode("utf-8"))
                        print('izquierda')
        if detecta==False:
            dir = 'q'
            ser.write(dir.encode("utf-8"))

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            dir = 'q'
            ser.write(dir.encode("utf-8"))
            ser.close()
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

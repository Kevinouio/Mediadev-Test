import cv2
import numpy as np

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def getMainContourBoundingBox(img, minArea):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_bbox = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea and area > max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            best_bbox = (x, y, w, h)
            max_area = area
    return best_bbox

def getContours(img, imgContour, trackerArea, bbox, imgOriginal):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > trackerArea:
            x, y, w, h = cv2.boundingRect(cnt)
            # Check if the contour is within the bounding box
            if x >= bbox[0] and y >= bbox[1] and x + w <= bbox[0] + bbox[2] and y + h <= bbox[1] + bbox[3]:
                # Draw on imgContour
                cv2.drawContours(imgContour, [cnt], -1, (0, 0, 255), 3)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
                cv2.drawContours(imgContour, [approx], -1, (0, 255, 0), 3)
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(imgContour, "Points: " + str(len(approx)), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(imgContour, "Area: " + str(int(area)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw on originalImage
                cv2.drawContours(imgOriginal, [cnt], -1, (0, 0, 255), 3)
                cv2.drawContours(imgOriginal, [approx], -1, (0, 255, 0), 3)
                cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(imgOriginal, "Points: " + str(len(approx)), (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(imgOriginal, "Area: " + str(int(area)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

img = cv2.imread("Brain tumor Images/brain1.jpg")

frameHeight, frameWidth = 640, 480

cv2.namedWindow("Tracking HSV")
cv2.resizeWindow("Tracking HSV", 640, 480)
cv2.createTrackbar("Hue Min", "Tracking HSV", 0, 255, lambda x: x)
cv2.createTrackbar("Hue Max", "Tracking HSV", 255, 255, lambda x: x)
cv2.createTrackbar("Sat Min", "Tracking HSV", 0, 255, lambda x: x)
cv2.createTrackbar("Sat Max", "Tracking HSV", 255, 255, lambda x: x)
cv2.createTrackbar("Vel Min", "Tracking HSV", 0, 255, lambda x: x)
cv2.createTrackbar("Vel Max", "Tracking HSV", 255, 255, lambda x: x)

cv2.namedWindow('Tracking', cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('Tracking', 640, 480)
cv2.createTrackbar('threshold1', 'Tracking', 150, 255, lambda x: x)
cv2.createTrackbar('threshold2', 'Tracking', 150, 255, lambda x: x)
cv2.createTrackbar('Area', 'Tracking', 10000, 20000, lambda x: x)

while True:
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue Min", "Tracking HSV")
    h_max = cv2.getTrackbarPos("Hue Max", "Tracking HSV")
    s_min = cv2.getTrackbarPos("Sat Min", "Tracking HSV")
    s_max = cv2.getTrackbarPos("Sat Max", "Tracking HSV")
    v_min = cv2.getTrackbarPos("Vel Min", "Tracking HSV")
    v_max = cv2.getTrackbarPos("Vel Max", "Tracking HSV")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    res = cv2.bitwise_and(img, img, mask=mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (frameWidth, frameHeight))

    horStack = stackImages(0.5, [img, res, mask])

    threshold1 = cv2.getTrackbarPos('threshold1', 'Tracking')
    threshold2 = cv2.getTrackbarPos('threshold2', 'Tracking')
    trackerArea = cv2.getTrackbarPos('Area', 'Tracking')
    kernel = np.ones((5, 5))

    imgContour = mask.copy()

    imgCanny = cv2.Canny(mask, threshold1, threshold2)
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    # Get the bounding box of the main contour (tumor)
    bbox = getMainContourBoundingBox(imgDil, trackerArea)

    imgCopy = img.copy()
    if bbox:
        getContours(imgDil, imgContour, trackerArea, bbox, imgCopy)

    imgStack = stackImages(0.5, ([img, mask, res], [res, imgContour, imgCopy]))

    cv2.imshow('Tracking', imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

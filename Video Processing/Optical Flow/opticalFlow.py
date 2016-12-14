import numpy as np
import cv2
import video

# Definition of function to render the flow lines on top of the retrieved video stream
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    visualization = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Polylines are utilized to indicate flow and magnitude of motion
    cv2.polylines(visualization, lines, 0, (255, 0, 0))
    # Add a dot at each given point for a static reference point
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(visualization, (x1, y1), 1, (255, 0, 255), -1)
    return visualization

# Import sys for access to the hardware of the host machine
if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    # Begin streaming from the built in capture device.
    cam = video.create_capture(fn)
    ret, prev = cam.read()
    # Convert the video stream to grayscale.
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Main loop that constantly captures the frames, converts them to grayscale and displays the optical flow lines
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # This method is utilized for dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray

        cv2.imshow('flow', draw_flow(gray, flow))

        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows()

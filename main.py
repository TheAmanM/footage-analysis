import numpy as np
import cv2
import imutils
import os
import time
import math

def printWithGap(text):
    print(f" {text}")

def analyzeVideo(list):
    # Find sum of every element in list
    sum = 0.0
    for i in list:
        sum += i
    
    # Find length of list
    listLength = len(list)


    clearConsole()
    printWithGap(f"Operation complete! File saved as \"{opname}\"")
    emptyLine()
    printWithGap("Time analysis: ")
    emptyLine()
    printWithGap(f"Total time taken to compile all frames: {str(round(sum, 2))} seconds")
    printWithGap(f"Average time taken per frame: {str(round(sum / listLength, 2))} seconds")
    emptyLine()
    printWithGap(f"Maximum time taken for a frame: {str(round(max(list), 2))} seconds")
    printWithGap(f"Minimum time taken for a frame: {str(round(min(list), 2))} seconds")
    emptyLine()


def Check(a,  b):
    global totalViolations
    dist = ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
    calibration = (a[1] + b[1]) / 2
    if 0 < dist < 0.25 * calibration:
        return True
    else:
        return False

def clear():
    clearConsole()
    emptyLine()
    printWithGap("Social Distancing Detector (in-progress): ")
    emptyLine()
    emptyLine()
    printWithGap("Progress: ")
    emptyLine()

def clearConsole():
    _ = os.system('cls')

def emptyLine():
    printWithGap("")

def getInputName():
    path = ""
    while (not os.path.isfile(path)):
        path = input(" Enter a valid input video name (such as input.mp4): ")

    shouldTakeScreenshotsKey = ""
    folderName = ""
  
    while(not(shouldTakeScreenshotsKey.lower() == "n" or shouldTakeScreenshotsKey.lower() == "y")):
        shouldTakeScreenshotsKey = input(" Would you like to save a screenshot of every frame? Please press \"y\" or \"n\": ")
 
    if (shouldTakeScreenshotsKey.lower() == "y"):
        while (folderName == ""):
            folderName = input(f" Enter a valid screenshots folder (such as \"screenshots\" or \"{path} screenshots\"). If the folder does not exist, the program will create it for you: ")
        if (not os.path.isdir(folderName)):
            print(f"Making directory called {folderName}")
            os.mkdir(folderName)

    return path, shouldTakeScreenshotsKey, folderName

def Setup(yolo):
    global net, ln, LABELS
    weights = "cfg/yolo.weights"
    config = "cfg/yolo.cfg"
    labelsPath = "cfg/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")  
    net = cv2.dnn.readNetFromDarknet(config, weights)
    ln = net.getLayerNames()
    # ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]

def ImageProcess(image):
    global processedImg
    global count, timeTakenList
    global numBlocks, blockCount, blocksString
    global totalViolations
    global shouldTakeScreenshots
    (H, W) = (None, None)
    frame = image.copy()
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    starttime = time.time()
    layerOutputs = net.forward(ln)
    stoptime = time.time()
    count = count + 1
    numBlocks = math.floor((count * blockCount) / length)
    blocksString = "Visual progress: ["
    for i in range(numBlocks):        
        blocksString += "█"
    for i in range(blockCount - numBlocks):
        blocksString += "_"
    blocksString += "]"
    clear()
    printWithGap(f"Frame {count} out of {length} ({round((count * 100) / length, 2)}%): ")
    timeTakenToProcessFrame = round(stoptime - starttime, 2)
    printWithGap(f"The frame was processed in {timeTakenToProcessFrame} seconds")
    timeTakenList.append(float(timeTakenToProcessFrame))
    emptyLine()
    printWithGap(blocksString)
    confidences = []
    outline = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            maxi_class = np.argmax(scores)
            confidence = scores[maxi_class]
            if LABELS[maxi_class] == "person":
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    outline.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))

    # box_line = cv2.dnn.NMSBoxes(outline, confidences, 0.5, 0.9)
    box_line = cv2.dnn.NMSBoxes(outline, confidences, 0.5, 0.3)

    if len(box_line) > 0:
        flat_box = box_line.flatten()
        pairs = []
        center = []
        status = [] 
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])
            status.append(False)

        for i in range(len(center)):
            for j in range(len(center)):
                close = Check(center[i], center[j])

                if close:
                    pairs.append([center[i], center[j]])
                    status[i] = True
                    status[j] = True
        index = 0

        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            if status[index] == True:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 4)
                totalViolations = totalViolations + 1
            elif status[index] == False:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
            index += 1
        for h in pairs:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
        totalViolationsText = f"People at risk: {str(totalViolations)}"
        if(totalViolations == 0):
            frame = cv2.putText(frame, totalViolationsText, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            frame = cv2.putText(frame, totalViolationsText, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    processedImg = frame.copy()

    # Save frame to Screenshots folder
    if (shouldTakeScreenshots.lower() == "y"):
        savePath = f"{folderName}/{str(count)}.jpg"
        cv2.imwrite(savePath, processedImg)
    totalViolations = 0

clearConsole()

inputName, shouldTakeScreenshots, folderName = getInputName()
clear()

create = None
frameno = 0
filename = inputName
yolo = "yolo"
opname = f"{inputName[:-4]} output.mp4"
printWithGap(opname)
cap = cv2.VideoCapture(filename)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
videoFps = int(cap.get(cv2.CAP_PROP_FPS))
count = 0
blockCount = 30
totalViolations = 0
timeTakenList = []

time1 = time.time()
clear()

while(True):

    ret, frame = cap.read()
    if not ret:
        break
    current_img = frame.copy()
    if(videoWidth < 1920):
        current_img = imutils.resize(current_img, width=1920)
        # videoWidth = 1920
    video = current_img.shape
    frameno += 1

    Setup(yolo)
    ImageProcess(current_img)
    Frame = processedImg
    if create is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        create = cv2.VideoWriter(opname, fourcc, videoFps, (Frame.shape[1], Frame.shape[0]), True)
    create.write(Frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
time2 = time.time()
analyzeVideo(timeTakenList)

cap.release()
cv2.destroyAllWindows()
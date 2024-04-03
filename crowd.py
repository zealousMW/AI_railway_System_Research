from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2

# Initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# Define the minimum safe distance (in pixels) that two people can be
# from each other
MIN_DISTANCE = 40

# Function to detect people
def detect_people(frame, net, ln, personIdx=0):
    # Grab the dimensions of the frame and initialize the list of results
    (H, W) = frame.shape[:2]
    results = []

    # Construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # Initialize our lists of detected bounding boxes, centroids, and confidences
    boxes = []
    centroids = []
    confidences = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        # Loop over each of the detections
        for detection in output:
            # Extract the class ID and confidence (i.e., probability)
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter detections by (1) ensuring that the object
            # detected was a person and (2) that the minimum
            # confidence is met
            if classID == personIdx and confidence > MIN_CONF:
                # Scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x, y)-coordinates to derive the top
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update our list of bounding box coordinates,
                # centroids, and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    # Apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    # Ensure at least one detection exists
    if len(idxs) > 0:
        # Loop over the indexes we are keeping
        for i in idxs.flatten():
            # Extract the bounding box coordinates (x, y)
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # Return the list of results
    return results

# Load the COCO class labels our YOLO model was trained on
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# Derive the paths to the YOLO weights and model configuration
weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"

# Load our YOLO object detector trained on the COCO dataset
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Determine only the output layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture("crowd_monitoring_sample.mp4")

# Loop over the frames from the video stream
while True:
    # Read the next frame from the file (grabbed, frame)
    (grabbed, frame) = vs.read()

    # If the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # Resize the frame and then detect people (and only people) in it
    frame = imutils.resize(frame, width=900)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

    # Initialize the set of indexes that violate the minimum social distance
    violate = set()

    # Ensure there are at least two people detections (required in order to compute our pairwise distance maps)
    if len(results) >= 2:
        # Extract all centroids from the results and compute the Euclidean distances between all pairs of centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # Loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # Check to see if the distance between any two centroid pairs is less than the configured number of pixels
                if D[i, j] < MIN_DISTANCE:
                    # Update our violation set with the indexes of the centroid pairs
                    violate.add(i)
                    violate.add(j)

    # Loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # Extract the bounding box and centroid coordinates, then
        # initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        # If the index pair exists within the violation set, then
        # update the color
        if i in violate:
            color = (0, 0, 255)

        # Draw (1) a bounding box around the person and (2) the
        # centroid coordinates of the person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    # Draw the total number of social distancing violations on the
    # output frame
    text = "Crowded areas identified till now: {}".format(int(len(violate) / 2))
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    # Display the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break

# Release the video stream and close any open windows
vs.release()
cv2.destroyAllWindows()

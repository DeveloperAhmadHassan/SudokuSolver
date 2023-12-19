from imutils.perspective import four_point_transform
from keras.src.utils import img_to_array
from skimage.data import cell
from skimage.segmentation import clear_border
import numpy as np
import cv2
from keras.models import load_model
from sudoku import Sudoku
import imutils

debug = True


def find_puzzle(image):
    # CODE_EXP: Convert the image to greyscale
    #   cv2.cvtColor(actual_image, color_conversion_code)
    #   color_conversion_code(from color to gray_scale): cv2.COLOR_BGR2GRAY
    #                                                    cv2.COLOR_RGB2GRAY
    #   This method returns a classic gray_scale image
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # INFO: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html#:~:text=Image%20Blurring%20(Image%20Smoothing),%2C%20edges)%20from%20the%20image.
    #   Blurring reduces noice

    # CODE_EXP: cv2.GaussianBlur(image, kernel, )
    grayImage = cv2.GaussianBlur(grayImage, (7, 7), 3)

    # INFO: https://pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/
    #   A Threshold needs to applied to an image so that foreground and background can be seperated
    #   A value(T) is given. If Pixel < T Then Pixel = 0; Else Pixel = T; T is usually 255

    # INFO: https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-2-adaptive-thresholding/
    #   The Threshold value is calculated for each pixel based on its local neighborhood
    #   It allows the capturing of smaller details

    # CODE_EXP: Convert the gray_scale image to threshold, to identify borders and values
    #   cv2.adaptiveThreshold(image, max_threshold, adaptive_method, threshold_type, neighbor_block_size, constant)
    #       image: actual image from path
    #       max_threshold: usually 255, max value that can be assigned to a pixel
    #       adaptive_method: method using which the threshold is calculated
    #           cv2.ADAPTIVE_THRESH_MEAN_C: (Mean of the neighbourhood area values – constant value).
    #           cv2.ADAPTIVE_THRESH_GAUSSIAN_C: (Gaussian-weighted sum of the neighbourhood values – constant value)
    #       threshold_type: The value that the Pixel will be assigned is decided here
    #       neighbor_block_size: The amount of values that need to be seen to calculate threshold for each pixel
    #       constant: the number that needs to be subtracted each time ***(NO IDEA WHY)***
    thresh = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # CODE_EXP: Inverting the color of the image helps to identify the things that we are interested in
    #   Bright(Usually White): Interested
    #   Dark(Usually Black): Not Interested(Background)
    #   This can be done in ***2 ways***
    #       cv2.bitwise_not(image)
    #       cv2.adaptiveThreshold(threshold_type = cv2.THRESH_BINARY_INV)
    inv_thresh = cv2.bitwise_not(thresh)

    # CODE_EXP: Getting the outline of all the objects in the image
    #   cv2.findContours(image_copy, mode, method)
    #   returns contours(the outlines) and hierarchy(***NO IDEA ABOUT THIS***)
    contours, hierarchy = cv2.findContours(inv_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # CODE_EXP: Sort the contours in descending order(reverse=True) to get the largest values on top
    #   The largest values represent the (big)box lines
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    borderContours = None

    # INFO: The code below is a result of EXTREME HIT and TRIAL
    #   It has no mathematical proof
    for c in contours:
        # INFO: The smaller the size of the image, the worse it gets
        curve = cv2.approxPolyDP(c, 0.09 * cv2.arcLength(c, True), True)

        # INFO: If the curve has four sides, it is the (big)border
        if 4 == len(curve):
            borderContours = curve
            print(curve)
            break

    # if debug:
    #     output = image.copy()
    #     # CODE_EXP: cv2.drawContours(image_to_draw_on, contours, starting_index, color(B,G,R), thickness_of_lines)
    #     cv2.drawContours(output, [borderContours], -1, (0, 0, 255), 2)
    #     cv2.waitKey(0)

    # INFO:

    # CODE_EXP: four_point_transform(image, xy_coordinates_of_four_points)
    #   borderContours.reshape(4, 2): This transforms the array into a 4 rows, 2 columns array; Representing x and y
    transformedPuzzle = four_point_transform(image, borderContours.reshape(4, 2))
    threshImage = four_point_transform(grayImage, borderContours.reshape(4, 2))

    return transformedPuzzle, threshImage


def extract_digit(cell):
    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    thresh = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh = clear_border(thresh)
    # check to see if we are visualizing the cell thresholding step
    # if debug:
    #     cv2.imshow("Cell Thresh", thresh)
    #     cv2.waitKey(0)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # print(f"cnts: {len(cnts)}")
    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return None
    # return thresh
    # otherwise, find the largest contour in the cell and create a
    # mask for the contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    # compute the percentage of masked pixels relative to the total
    # area of the image
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    # if less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    if percentFilled < 0.03:
        return None
    # return thresh
    # apply the mask to the thresholded cell

    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    # check to see if we should visualize the masking step
    # if debug:
    #     cv2.imshow("Digit", digit)
    #     cv2.waitKey(0)
    # return the digit to the calling function
    return digit


def solvePuzzle(imagePathFromAPI):
    model = load_model('output/digit_classifier.keras')
    board = np.zeros((9, 9), dtype="int")

    imagePath = 'uploads/' + imagePathFromAPI
    # CODE_EXP: Getting the 'actual image' from the image path
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=1600)

    (puzzleImage, warped) = find_puzzle(image)

    # INFO: Assuming that the boxes in the puzzle are of equal sizes

    # CODE_EXP: Getting the Columns(.shape[1]); These are total pixels, in width
    #   And then get the size(width) of one Cell by dividing the pixels by 9

    stepX = warped.shape[1] // 9
    # DEBUG: print(f"ShapeX: {warped.shape[1]} - StepX : {warped.shape[1] // 9}")

    # CODE_EXP: Getting the Rows(.shape[0]); These are total pixels, in height
    #   And then get the size(height) of one Cell by dividing the pixels by 9

    stepY = warped.shape[0] // 9
    # DEBUG: print(f"ShapeY: {warped.shape[0]} - StepY : {warped.shape[0] // 9}")

    cellLocs = []
    digitLocs = []

    # loop over the grid locations
    for y in range(0, 9):
        row = []
        for x in range(0, 9):
            # CODE_EXP: Getting the starting Pixel of the cell by x and y
            startX = x * stepX
            startY = y * stepY
            # CODE_EXP: Getting the ending Pixel of the cell by x and y
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
            # DEBUG:
            #     print(f"StartX: {startX} - startY: {startY}")
            #     print(f"endX: {endX} - endY: {endY}")
            #     print("")

            # CODE_EXP: 0 + x, x + x, x + x + x,.........
            row.append((startX, startY, endX, endY))
            cell = warped[startY:endY, startX:endX]
            # DEBUG:
            #     cv2.imshow("Cell", cell)
            #     cv2.waitKey(0)

            # CODE_EXP: Getting the digit cell
            #   There should be nothing else in the returned photo (no noise, no lines, no borders)
            #   Only the digit, in inverse thresh
            digit = extract_digit(cell)
            if digit is not None:
                # CODE_EXP: Model was trained on 28x28 size photos
                #   Numbers should be in float, below 255
                imageForModel = cv2.resize(digit, (28, 28))
                imageForModel = imageForModel.astype("float") / 255.0
                imageForModel = img_to_array(imageForModel)
                imageForModel = np.expand_dims(imageForModel, axis=0)

                predictedDigit = model.predict(imageForModel).argmax(axis=1)[0]
                print(predictedDigit)

                # CODE_EXP: Store the Predicted Digit in the board
                #   This board will be used to solve the puzzle, once filled
                #   Empty spaces in the board need to empty
                board[y, x] = predictedDigit
                digitLocs.append((startX, startY, endX, endY))
        cellLocs.append(row)

    print("[Sudoku board:")
    puzzle = Sudoku(3, 3, board=board.tolist())
    puzzle.show()

    # DEBUG: print(len(digitLocs))
    # DEBUG: print(digitLocs)

    print("Solved Sudoku Board")
    solution = puzzle.solve()
    solution.show_full()

    for (cellRow, boardRow) in zip(cellLocs, solution.board):
        for (box, digit) in zip(cellRow, boardRow):
            # DEBUG: print(box)
            if not digitLocs.__contains__(box):
                startX, startY, endX, endY = box
                textX = int((endX - startX) * 0.33)
                textY = int((endY - startY) * -0.2)
                textX += startX
                textY += endY
                cv2.putText(puzzleImage, str(digit), (textX, textY), cv2.FONT_ITALIC, 5, (0, 255, 0), 7)

    # DEBUG: cv2.imshow("Sudoku Result", puzzleImage)
    cv2.imwrite("solvedPuzzles/"+'solved_'+imagePathFromAPI, puzzleImage)
    return True

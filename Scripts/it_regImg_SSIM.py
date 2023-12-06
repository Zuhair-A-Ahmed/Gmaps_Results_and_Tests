#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import imutils
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import argparse
#from skimage.measure import compare_ssim
#featuresNumber = 75
bestSSIM=0
bestNM=0
bestFN=0
parser = argparse.ArgumentParser()
parser.add_argument('x', type=int)
parser.add_argument('y', type=int)
args = parser.parse_args()
featuresNumber= 0
for itt in range(args.x,args.y):
    featuresNumber=itt
    # Convert images to grayscale for computing the rotation via ECC method
    # Open the image files.
    for file in os.listdir():
        filename, extension  = os.path.splitext(file)
        if extension == ".pgm" and filename[0]=="g":
            extensionPGM = extension
            filenamePGM = filename

    img1_color = cv2.imread("{}.pgm".format(filenamePGM))  # Image to be aligned.
    img2_color = cv2.imread("../gt/gt.pgm")  # Reference image.

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(featuresNumber)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    # (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = bfmatcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    #matches.sort(key=lambda x: x.distance)
    matches = sorted(matches, key=lambda x: x.distance)
    # Take the top 90 % matches forward.
    matches = matches[:]
    no_of_matches = len(matches)
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2),dtype=np.float32)
    p2 = np.zeros((no_of_matches, 2),dtype=np.float32)
    #for i in range(len(matches)):
    for i, match in enumerate(matches):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    # homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    h= cv2.estimateAffinePartial2D(p1, p2, method = cv2.RANSAC)[0]
    #h= cv2.estimateAffinePartial2D(p1, p2, method = cv2.LMEDS)[0]
    # Use this matrix to transform the
    # colored image wrt the reference image.
    # transformed_img = cv2.warpPerspective(img1_color, homography, (width, height))
    transformed_img = cv2.warpAffine(img1_color, h, (width, height))
    transformed_img_Grey = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY)
    # Save the output.
    #cv2.imwrite('output.pgm', transformed_img_Grey)
    cv2.imwrite('RegImg%dof%s.pgm'%(featuresNumber,filenamePGM), transformed_img_Grey)
    #cv2.imshow('output.pgm', transformed_img_Grey)
    #print(no_of_matches)


    # Open the image files.
    imageA  = cv2.imread('../gt/gt.pgm')  # Image to be aligned.
    imageB = cv2.imread('RegImg%dof%s.pgm'%(featuresNumber,filenamePGM))  # Reference image.
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    #(score, diff) = compare_ssim(grayA, grayB, full=True)
    (score, diff) = structural_similarity(grayA, grayB,full=True)
    diff = (diff * 255).astype("uint8")
    #print(("SSIM: {}".format(score)))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # the output images
    #cv2.imwrite('Diff%d-%f-%d-%s.pgm'%(featuresNumber,score,no_of_matches,filenamePGM), diff)
    #cv2.imwrite('Thresh%dand%fof%s.pgm'%(featuresNumber,score,filenamePGM), thresh)
    print("no_of_matches: {}".format(no_of_matches))
    print("features Number: {}".format(featuresNumber))
    print(("SSIM: {}".format(score)))
    if score > bestSSIM:
            bestSSIM=score
            bestNM=no_of_matches
            bestFN=featuresNumber
    if score >= 0.87:
        cv2.imwrite('Diff%d-%f-%d-%s.pgm'%(featuresNumber,score,no_of_matches,filenamePGM), diff)
        #cv2.imwrite('Thresh%dand%fof%s.pgm'%(featuresNumber,score,filenamePGM), thresh)
    else:
        os.remove('RegImg%dof%s.pgm'%(featuresNumber,filenamePGM))
    cv2.waitKey(0)

print("Best no_of_matches: {}".format(bestNM))
print("Best features Number: {}".format(bestFN))
print(("Best SSIM: {}".format(bestSSIM)))

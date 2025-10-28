import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Expanding the program with the functions from assignment 4.

# I have chosen to use SIFT on task 2.

# There are two pictures in solutions for sift_matches. One is with the parameters
# given in the assignmetn (10 and 0.7) and the other is with unlimited features.
# The one with unlimited showed some results, while the other did not show anything.

# - Harris corner detection (line 254)
# - SIFT feature matching (line 283)


# Using reference_img.png as default
img = cv2.imread("reference_img.png")
img_align = cv2.imread("align_this.jpg")


def show_image(image, title="Picture"):
    if image is None:
        print("There is no image to show at this time. \n")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def accessPixelValue(image, row, col):
    h, w = image.shape[:2]
    if not (0 <= row < h and 0 <= col < w):
        print("\nChosen values are out of bounds \n")
        return
    px = image[row, col]
    print( px )


def padding(image, border_width):
    reflect = cv2.copyMakeBorder(image, border_width,border_width,border_width,border_width,cv2.BORDER_REFLECT)
    reflect_rgb = cv2.cvtColor(reflect, cv2.COLOR_BGR2RGB)
    plt.imshow(reflect_rgb)
    plt.title("REFLECTED")
    plt.axis("off")
    plt.show()
    cv2.imwrite("solutions/lena_reflected.png", reflect_rgb)
    return reflect


def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    plt.imshow(cropped_rgb)
    plt.title("CROPPED")
    plt.axis("off")
    plt.show()
    cv2.imwrite("solutions/lena_cropped.png", cropped_rgb)
    return cropped_rgb

def resize(image, width, height):
    resize = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    resize_rgb = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
    plt.imshow(resize_rgb)
    plt.title("RESIZED")
    plt.axis("off")
    plt.show()
    cv2.imwrite("solutions/lena_resized.png", resize_rgb)
    return resize_rgb


def grayscale(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(grayscale, cmap="gray")
    plt.title("GRAYSCALE")
    plt.axis("off")
    plt.show()
    cv2.imwrite("solutions/lena_grayscale.png", grayscale)
    return grayscale


def hsv(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    plt.imshow(hsvImage)
    plt.title("HSV")
    plt.axis("off")
    plt.show()
    cv2.imwrite("solutions/lena_hsv.png", hsvImage)
    return hsvImage


def smoothing(image):
    smooth = cv2.GaussianBlur(image, (15,15), cv2.BORDER_DEFAULT)
    smooth_rgb = cv2.cvtColor(smooth, cv2.COLOR_BGR2RGB)
    plt.imshow(smooth_rgb)
    plt.title("SMOOTH")
    plt.axis("off")
    plt.show()
    cv2.imwrite("solutions/lena_smooth.png", smooth_rgb)
    return smooth_rgb


def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180_CLOCKWISE)
    elif rotation_angle == 270:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        print("No valid angle has been chosen. \n")

    rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

    plt.imshow(rotated_rgb)
    plt.title("ROTATED")
    plt.axis("off")
    plt.show()
    cv2.imwrite("solutions/lena_rotated.png", rotated_rgb)

    return rotated_rgb


def copy_manual(image):

    height, width, channels = image.shape
    emptyPictureArray = np.zeros((height, width, channels), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = image[y, x, c]
    copied_rgb = cv2.cvtColor(emptyPictureArray, cv2.COLOR_BGR2RGB)

    plt.imshow(copied_rgb)
    plt.title("COPIED")
    plt.axis("off")
    plt.show()
    cv2.imwrite("solutions/lena_copied.png", copied_rgb)

    return copied_rgb


def hue_shifted(image, hue):

    height, width, channels = image.shape
    emptyPictureArray = np.zeros((height, width, channels), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                hueValue = int(image[y, x, c]) + int(hue)
                if hueValue > 255:
                    hueValue -= 255
                elif hueValue < 0:
                    hueValue += 255
                emptyPictureArray[y, x, c] = hueValue
    hueShift_rgb = cv2.cvtColor(emptyPictureArray, cv2.COLOR_BGR2RGB)

    plt.imshow(hueShift_rgb)
    plt.title("HueShift")
    plt.axis("off")
    plt.show()
    cv2.imwrite("solutions/lena_hueShift.png", hueShift_rgb)

    return hueShift_rgb


def sobel_edge_detection(image):
    imgOriginal = image
    imgGrayscale = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGrayscale, (3,3), 0)

    sobelxy = cv2.Sobel(src=imgBlur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

    plt.imshow(sobelxy)
    plt.title("sobelXY")
    plt.axis("off")
    plt.show()
    cv2.imwrite("solutions/lambo_sobelxy.png", sobelxy)

    return sobelxy


def canny_edge_detection(image, threshold_1, threshold_2):
    imgOriginal = image
    imgGrayscale = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGrayscale, (3,3), 0)
    edges = cv2.Canny(imgBlur, threshold_1, threshold_2)

    plt.imshow(edges)
    plt.title("Edges")
    plt.axis("off")
    plt.show()
    cv2.imwrite("solutions/lambo_edges.png", edges)

    return


def template_match(image, template):

    if template is None:
        print(f"Could not read template: {template}")
        return

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    temp = cv2.imread(template, 0)
    w, h = temp.shape[::-1]

    res = cv2.matchTemplate(img_gray, temp, cv2.TM_CCOEFF_NORMED)

    try:
        threshold = float(input("Enter threshold: ").strip())
    except:
        print("Invalid input")
        return
    
    loc = np.where( res >= float(threshold))
    img_out = image.copy()
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_out, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.title("Match")
    plt.axis("off")
    plt.show()

    cv2.imwrite("solutions/template_match.png", img_out)
    return img_out


def zoom_resize(image, scale_factor: int, up_or_down: str):
    if image is None:
        print("No image to resize")
        return
    if up_or_down != "down" or up_or_down != "up":
        print("Invalid argument for up_or_down")
        return
    
    resize = image.copy()
        
    for _ in range(scale_factor):
        rows, cols, _channels = map(int, resize.shape)
        if up_or_down == "down":
            resize = cv2.pyrDown(resize, dstsize=(2 // cols, 2 // rows))
        elif  up_or_down == "up":
            resize = cv2.pyrUp(resize, dstsize=(2 * cols, 2 * rows))
        else:
            print("Invalid argument")

    plt.imshow(cv2.cvtColor(resize, cv2.COLOR_BGR2RGB))
    plt.title("Resized")
    plt.axis("off")
    plt.show()

    cv2.imwrite("solutions/resize.png", resize)

    return resize


def harris(reference_image):
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    new_image = reference_image.copy()
    new_image[dst > 0.01 * dst.max()] = [0, 0, 255]
    
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.title("Harris Corners")
    plt.axis("off")
    plt.show()
    cv2.imwrite("solutions/harris.png", new_image)

    return new_image


def sift(image_to_align, reference_image, max_features, good_match_percent):
    img1 = reference_image.copy()
    img2 = image_to_align.copy()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures = max_features)

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < good_match_percent * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = gray1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good, None)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask = matchesMask,
                       flags = 2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.title("SIFT Matches")
    plt.axis("off")
    plt.show()
    cv2.imwrite("solutions/sift_matchesNF0.png", img3)

    return img3


    




while True:
    print("\n=== Menu ===")
    print("1. Set image")
    print("2. Show image with details")
    print("3. Access pixel value")
    print("4. Create border which reflects the edges")
    print("5. Crop image")
    print("6. Resize image")
    print("7. Grayscale image")
    print("8. HSV image")
    print("9. Smooth image")
    print("10. Rotate image")
    print("11. Copy image")
    print("12. Hue image")
    print("13. Edge detection")
    print("14. Canny edge detection")
    print("15. Template match")
    print("16. Zoom resize")
    print("17. Harris corner detection")
    print("18. SIFT feature matching (between img and img2 variables set at top of script)")
    print("0. Exit")
    print("\n")
    choice = input("Choose: ").strip()


    if choice == "1":
        fileName = input("Enter the complete file name of the image you want to work with (None for no image): ").strip()
        if fileName == "None" or fileName == "none":
            print("No file selected")
        img = cv2.imread(fileName)


    elif choice == "2":
        if img is None:
            print("No image has been set. Please set one. \n")
            continue

        print("Current picture has the following values:")
        print("rows, collums, colors")
        print(img.shape)

        print("Image size:")
        print(img.size)

        show_image(img)


    elif choice == "3":
        if img is not None:
            try:
                access_row = int(input("Enter pixel row: ").strip())
                access_col = int(input("Enter pixel col: ").strip())
            except ValueError:
                print("You must enter integer values for row and col.")
                continue
            accessPixelValue(img, access_row, access_col)
        else:
            print("No image has been set.")


    elif choice == "4":
        if img is not None:
            try:
                width = int(input("Enter the desired border pixel width: ").strip())
            except ValueError:
                print("Invalid input. Must be an integer.")
                continue
            padding(img, width)
        else:
            print("No image has been set.")
        


    elif choice == "5":
        if img is not None:
            try:
                crop_x1 = int(input("Enter x1 value: ").strip())
                crop_x2 = int(input("Enter x2 value: ").strip())
                crop_y1 = int(input("Enter y1 value: ").strip())
                crop_y2 = int(input("Enter y2 value: ").strip())
            except ValueError:
                print("Invalid input. Must be an integer.")
                continue
            crop(img, crop_x1, crop_x2, crop_y1, crop_y2)
        else:
            print("No image has been set.")
        

    
    elif choice == "6":
        if img is not None:
            try:
                resize_width = int(input("Enter width: ").strip())
                resize_height = int(input("Enter height: ").strip())
            except ValueError:
                print("Invalid input. Must be an integer.")
                continue
            resize(img, resize_width, resize_height)
        else:
            print("No image has been set.")

    
    elif choice == "7":
        if img is not None:
            grayscale(img)
        else:
            print("No image has been set.")

    
    elif choice == "8":
        if img is not None:
            hsv(img)
        else:
            print("No image has been set.")


    elif choice == "9":
        if img is not None:
            smoothing(img)
        else:
            print("No image has been set.")


    elif choice == "10":
        if img is not None:
            rotation_angle = int(input("Enter rotation angle (90, 180 or 270): ").strip())
            if rotation_angle == 90 or rotation_angle == 180 or rotation_angle == 270: 
                rotation(img, rotation_angle)
            else:
                print("Invalid input. Must enter 90, 180 or 270. \n")
        else:
            print("No image has been set.")


    elif choice == "11":
        if img is not None:
            copy_manual(img)
        else:
            print("No image has been set.")

    elif choice == "12":
        if img is not None:
            try:
                hueShift = int(input("Enter hue shift value: ").strip())
            except:
                print("Must enter an integer")
            hue_shifted(img, hueShift)

    elif choice == "13":
        if img is not None:
            sobel_edge_detection(img)

    elif choice == "14":
        if img is not None:
            try:
                threshold1 = int(input("Enter threshold1: ").strip())
                threshold2 = int(input("Enter threshold2: ").strip())
            except ValueError:
                print("Invalid input. Must be an integer.")
                continue
            canny_edge_detection(img, threshold1, threshold2)
        else:
            print("No image has been set.")

    elif choice == "15":
        if img is not None:
            template = input("Please enter template image file name (img.png): ")
            template_match(img, template)
        else:
            print("No image has been set.")

    elif choice == "16":
        if img is not None:
            up_or_down = input("Do you want to scale 'up' or 'down'? (up/down): ")
            factor = int(input("Please enter the scale factor (an integer): "))
            zoom_resize(img, factor, up_or_down)
        else:
            print("No image has been set.")

    elif choice == "17":
        if img is not None:
            harris(img)
        else:
            print("No image has been set.")

    elif choice == "18":
        if img is not None:
            percent = float(input("Enter good match percentage: ").strip())
            max_features = int(input("Enter maximum number of features to detect: ").strip())
            sift(img, img_align, max_features, percent) # img and img_align variables are set at top of script
        else:
            print("No image has been set.")



    

    elif choice == "0":
        break
    else:
        print("Unknown choice.")

    




if __name__ == "__main__":
    main()
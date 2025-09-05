import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("lena.png")


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
    print("10. Rotate Image")
    print("11. Copy Image")
    print("12. Hue Image")
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


    elif choice == "0":
        break
    else:
        print("Unknown choice.")

    
    




if __name__ == "__main__":
    main()
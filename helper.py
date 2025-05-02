import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract


def load_images(path: Path) -> list:
    images = []
    for file in path.iterdir():
        if file.is_file() and file.suffix in ['.jpg', '.png', '.jpeg']:
            # img = cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2GRAY)
            img = cv2.imread(str(file))
            images.append(img)
    return images


def filter_for_black(img: np.ndarray, lower_black: np.ndarray, upper_black: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Filter the image to keep only black pixels. Everything that is not black becomes white.

    Args:
        img (np.ndarray): The input image in BGR format.
        lower_black (np.ndarray): The lower bound for the black color in HSV.
        upper_black (np.ndarray): The upper bound for the black color in HSV.

    Returns:
        tuple[np.ndarray, int]: A tuple containing the filtered image and the count of black pixels.

        The filtered image is in grayscale format where black pixels are 0 and everything else is 255.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (7, 7), 0)
    mask = cv2.inRange(blurred, lower_black, upper_black)

    # Morphological opening to remove small black regions
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    result = cv2.bitwise_not(mask_clean)
    black_pixels_count = result.size - cv2.countNonZero(result)
    return result, black_pixels_count


def resize_image(img: np.ndarray, target_size: tuple) -> np.ndarray:
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)


def crop_image(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    return img[y:y + h, x:x + w]


def rotate_image_2(img: np.ndarray, angle: float) -> np.ndarray:
    # Pad image to prevent cropping after rotation
    h, w = img.shape[:2]
    cX, cY = w // 2, h // 2
    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    # Compute the new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # Perform the rotation
    return cv2.warpAffine(img, M, (nW, nH), borderValue=255)


def get_first_whitelisted_char(text, whitelist="ABC"):
    filtered = re.sub(f"[^{whitelist}]", "", text)
    if len(filtered) > 1:
        print(f"Recognized more than one character: {filtered}")
    return filtered[0] if filtered else ""


def do_multiple_recognition_cycles(img: np.ndarray, config: str, angles: list, lower_black: np.ndarray, upper_black: np.ndarray, debug: bool) -> str:
    """Performs multiple recognition cycles turning the image by 15 degrees each time. After each cycle the recognized text is stored in a list.
    The function returns the character with the most occurrences.

    Args:
        img (np.ndarray): The image to be processed.
        config (str): The Tesseract configuration string.
        angles (list): The list of angles to rotate the image.
        lower_black (np.ndarray): The lower bound for the black color in HSV.
        upper_black (np.ndarray): The upper bound for the black color in HSV.
        debug (bool): If True, display the images during processing.

    Returns:
        str: The recognized text.
    """
    recognized_chars = []
    for angle in angles:
        # lower resolution works better, don't know why
        resized_img = None
        if img.shape[0] > 2000:
            resized_img = resize_image(img, (img.shape[1] // 18, img.shape[0] // 18))
        else:
            resized_img = resize_image(img, (img.shape[1] // 4,
                                         img.shape[0] // 4))
        x = int(resized_img.shape[1] * 0.2)
        y = int(resized_img.shape[0] * 0.2)
        w = int(resized_img.shape[1] * 0.6)
        h = int(resized_img.shape[0] * 0.4)
        croped_img = crop_image(resized_img, x, y, w, h)
        filtered_img, black_pixels_count = filter_for_black(croped_img, lower_black, upper_black)
        if black_pixels_count < 500:
            print(f"Not enough black pixels at {angle} degrees: {black_pixels_count}")
            print(f"Breaking out of the loop.")
            break
        rotated_img = rotate_image_2(filtered_img, angle)
        if debug:
            display_image(rotated_img)
        text = pytesseract.image_to_string(rotated_img, config=config)
        if not text.strip():
            continue
        char = get_first_whitelisted_char(text.strip())
        recognized_chars.append(char)
        print(f"Recognized text at {angle} degrees: {text.strip()}")
    if not recognized_chars:
        print("No text recognized.")
        return ""
    if 'A' in recognized_chars or 'B' in recognized_chars:
        recognized_chars = [char for char in recognized_chars if char in ['A', 'B']]
    most_common_text = max(set(recognized_chars), key=recognized_chars.count)
    print(f"Most common char occurrences: {recognized_chars.count(most_common_text)}")
    return most_common_text


def recognize_text(img, lower_black, upper_black, debug) -> str:
    custom_config = r'--psm 10 --oem 3 -c tessedit_char_whitelist=ABC'
    angles = [i for i in range(0, 360, 12)]
    # angles = [i for i in range(0, 30, 10)] + [i for i in range(330, 360, 10)]
    return do_multiple_recognition_cycles(img, custom_config, angles, lower_black, upper_black, debug)


def display_image(img):
    # rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

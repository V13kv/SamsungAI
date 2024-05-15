import subprocess
import printy
import random
import os
import cv2
from pathlib import Path
import numpy as np



# Used:
# https://imagemagick.org/script/convert.php
class TtfFonts:
    """
    Class that organizes the work with ttf fonts
    """
    BAD_FONTS = (b'cst', b'lklug', b'ani', b'rsfs', b'esint', b'cmex', b'msam', b'EagleLake')  # excluding bad fonts (do not have numbers or bad numbers font)

    def __init__(self, outputFolder = "fonts", numberOfFonts = None): # number of fonts are not linear, they are picked randomly
        self.outputFolder = outputFolder
        self.numberOfFonts = numberOfFonts
        self.ttf_fonts = self.__getFonts()

    def fontsToPng(self):
        # Creating output directory with fonts converted to .png format
        if not os.path.isdir(self.outputFolder):
            os.mkdir(self.outputFolder)
        
        for ttf in self.ttf_fonts:
            self.__ttfToPng(ttf)
        
        printy("[nB][+]@ Digits are successfully generated and converted to png-format")

    def __getFonts(self):
        # Read all available fonts via executing subprocess
        process = subprocess.Popen(['convert', '-list', 'font'], stdout=subprocess.PIPE)    # execute "./convert -list font", program output to PIPE
        out = process.communicate()[0]  # getting stdout of executed program (in bytes), ignoring stderr

        # Choose good fonts
        fonts = list()
        for fontInfo in out.split():
            if fontInfo.find(b'ttf') != -1 and not any(map(lambda badFontSubStr: fontInfo.find(badFontSubStr) != -1, self.BAD_FONTS)):   # if retrieved output is ttf (hence, a font) then check it for not being in a BAD_FONTS list 
                fonts.append(fontInfo.decode("utf-8"))  # forcibly convert or perceive the output text in utf-8 format
        
        if self.numberOfFonts is not None and self.numberOfFonts <= len(fonts):
            return random.sample(fonts, self.numberOfFonts) # choose random fonts
        else:
            return fonts

    def __ttfToPng(self, ttf_font_path, ttf_font_size = 500):
        """
        Generate png picture of digits with a font from ttf_font_path.
        """
        ttf_font_name = os.path.splitext(os.path.basename(ttf_font_path))[0] # getting font name WO an extension
        for i in range(10):
            output_png_path = os.path.join(self.outputFolder, str(i) + "_" + ttf_font_name + ".png")
            subprocess.call(["convert-im6.q16", "-font", ttf_font_path, "-pointsize", str(ttf_font_size), "-background",
                            "#FFFFFF", "label:" + str(i), output_png_path])


class PNGNormalize:
    """
    Normalizes the height and width of the png pictures located in folder directory
    Border is an identation between contour of digit and a border of an image that will be generated => fit digit
    """
    def __init__(self, folder, border = 1.5):
        self.folder = folder    # input folder, where png's are located (the png's will be overwritten with compressed ones)
        self.border = border
    
    def normalize(self, verbose = False):
        printy(f"[yB][INFO]@ Normalizing pictures in {self.folder} with {self.border} border")

        files = os.listdir(self.folder)
        for file in files:
            fileName, fileExt = os.path.splitext(file)
            if fileExt == ".png":
                self.__normalize(fileName, verbose)
        
        printy(f"[nB][+]@ All pictures in {self.folder} are successfully normalized (compressed)")
    
    def __normalize(self, pictureName, verbose: bool):
        # See conducted research: "CompressImage.ipynb" file
        img = cv2.imread(os.path.join(self.folder, pictureName + '.png')) # reading white-black image (white background)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, 0, 255, 0)   # will get black-white image (black background), hence binary image

        # Finding contours (see docs: https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # finding binary image contours
        try:
            # contours[0] - first in a hierarchy contour (contour is a list of points (pairs, e.g. (x, y))) (in our case first contour is outer contour)
            # getting (minX, minY) and (maxX, maxY) coordinates in order to norm
            x0 = (min(contours[0][i][0][0] for i in range(contours[0].shape[0])), min(contours[0][i][0][1] for i in range(contours[0].shape[0])))
            x1 = (max(contours[0][i][0][0] for i in range(contours[0].shape[0])), max(contours[0][i][0][1] for i in range(contours[0].shape[0])))
        except IndexError:
            # if we didn't fund contours then there is a bad digit in .png file, hence remove it
            print("[!] Error to norm ", pictureName, end='; ')
            Path.unlink(os.path.join(self.folder, pictureName + '.png'))
            print("Successfully removed")
            return

        # if a blank area does not exist => we don't need this font
        if int(max(x1[0] - x0[0], x1[1] - x0[1])) == 0:
            print("[!] Error to norm (Bad Image) ", pictureName, end='; ')
            Path.unlink(os.path.join(self.folder, pictureName + '.png'))
            print("Successfully removed")

        # Drawing blank area for future compressed, in terms of dimensions, image
        width = height = int(max(x1[0] - x0[0], x1[1] - x0[1]) * self.border)
        new_img = np.zeros((height, width, 3), np.uint8)    # 3 channels (RGB)
        cv2.rectangle(new_img, (0, 0), (width, height), (255, 255, 255), -1)    # -1 -- fill rectangle with WHITE color (specified)

        # Filling newly allocated blank area
        dx = int(width / 2 - (x1[0] - x0[0]) / 2)
        dy = int(height / 2 - (x1[1] - x0[1]) / 2)
        new_img[dy:x1[1] - x0[1] + dy, dx:x1[0] - x0[0] + dx] = img[x0[1]:x1[1], x0[0]:x1[0]]

        # Saving new image
        cv2.imwrite(os.path.join(self.folder, pictureName + ".png"), new_img)

        if verbose:
            printy(f"[nB][+]@ {self.folder}/{pictureName}.png")


if __name__ == '__main__':
    folderName = "fonts"

    # Getting ttf fonts in png format
    TtfFonts(folderName, 30).fontsToPng()

    # Compress (or normalize) png pictures (in our case - digits)
    PNGNormalize(folderName).normalize()
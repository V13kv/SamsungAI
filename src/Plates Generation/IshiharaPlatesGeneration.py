import os
import random
import math
from time import time
from PIL import Image, ImageDraw
from printy import printy


color = lambda c: ((c >> 16) & 255, (c >> 8) & 255, c & 255)
colorToHex = lambda tuple: hex(tuple[0] << 16 | tuple[1] << 8 | tuple[2])


class ColorThemes:
    THEME_NAME_PREFIX_LENGTH = 12
    THEMES = {
        "COLOR_THEME_BROWN_GREY_YELLOW": {
            "COLORS_ON": [color(0xEBA170), color(0xF9BB82), color(0xFCCD84), color(0xffc68c), color(0xef845a)],
            "COLORS_OFF": [color(0x9CA594), color(0xD7DAAA), color(0xACB4A5), color(0xBBB964), color(0xE5D57D), color(0xD1D6AF), 
                        color(0xb6b87c), color(0xe3da73), color(0xb0ab60), color(0xfff36b), color(0xffbd52)]
        },
        "COLOR_THEME_RED_GRAY": {
            "COLORS_ON": [color(0xf26969), color(0xd8859d), color(0xf79087)],
            "COLORS_OFF": [color(0x5a4e46), color(0x7b6b63), color(0x9c9c84)]
        },
        "COLOR_THEME_RED_GREEN": {
            "COLORS_ON": [color(0x9d3c2b), color(0xea452f)],
            "COLORS_OFF": [color(0x2d6524), color(0x418C22)]
        },
        "COLOR_THEME_GREEN_BROWN": {
            "COLORS_ON": [color(0x957544), color(0x6f4a13)],
            "COLORS_OFF": [color(0x448B24), color(0x2c6224)]
        },
        "COLOR_THEME_GREEN_BLUE": {
            "COLORS_ON": [color(0x4873A6), color(0x1F2e69)],
            "COLORS_OFF": [color(0x418d2b), color(0x2b6322)]
        },
        "COLOR_THEME_BLUE_GRAY": {
            "COLORS_ON": [color(0x6e97c5), color(0x1f2e69)],
            "COLORS_OFF": [color(0x848685), color(0x4c4a4b)]
        },
        "COLOR_THEME_BLUE_PURPLE": {
            "COLORS_ON": [color(0x2383f1), color(0x1c33a7)],
            "COLORS_OFF": [color(0x8c52fc), color(0x562a8b)]
        },
        "COLOR_THEME_GREEN_GRAY": {
            "COLORS_ON": [color(0x969696), color(0x39373c)],
            "COLORS_OFF": [color(0x549a5b), color(0x2e6933)]
        },
        "COLOR_THEME_GREEN_BLACK": {
            "COLORS_ON": [color(0x030303)],
            "COLORS_OFF": [color(0x709A74), color(0x5ec751)]
        },
    }


# TODO: add support to not generate plates with themes that are already generated, hene generate only new ones and create folder for new ones
# TODO: add support to generate not only digits, but numbers > 10
# TODO: add color gradient (randomly choose colors simillar to one chosen)
class IshiharaPlatesGenerator(ColorThemes):
    def __init__(self, fontFolder, outputFolder, createThemeFolders = True):
        self.fontFolder = fontFolder
        self.outputFolder = outputFolder

        # instances used in Ishihara Plates generation
        self.fontBackgroundColor = None
        self.density = None
        self.min_radius_factor = None
        self.max_radius_factor = None
        self.min_radius = None
        self.max_radius = None

        # Create dirs for Ishihara plates and each color theme
        if not os.path.isdir(self.outputFolder):
            os.mkdir(self.outputFolder)

        for themeFolder in self.THEMES.keys():
            if not os.path.isdir(os.path.join(self.outputFolder, themeFolder[self.THEME_NAME_PREFIX_LENGTH:])):
                os.mkdir(os.path.join(self.outputFolder, themeFolder[self.THEME_NAME_PREFIX_LENGTH:]))
    

    def generate(self, verbose = False, fontBackgroundColor = color(0xFFFFFF), density: float = 1, min_radius_factor: int = 200, max_radius_factor: int = 75):
        self.fontBackgroundColor = fontBackgroundColor
        self.density = density
        self.min_radius_factor = min_radius_factor
        self.max_radius_factor = max_radius_factor
        
        count = 0
        totalTime = 0
        files = os.listdir(self.fontFolder)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == ".png": 
                # Measure time needed for generating one plate with specified font
                start = time()

                self._generate(name, verbose)

                end = time()
                totalTime += end - start

                print(f"[{count + 1}/{len(files)}]", end=' ')
                printy(f"[nB][+]@ {name} time elapsed: {end - start}s")
  
            count += 1
        
        printy(f"[nB][+]@ Ishihara Plates with '{self.fontFolder}' fonts are successfully generated in {self.outputFolder} folder!")
        print(f"Total time elapsed: {totalTime:.3f}s")
        

    def _generateCircle(self, main_x_center, main_y_center, main_radius):
        radius = random.triangular(self.min_radius, self.max_radius, self.max_radius * 0.8 + self.min_radius * 0.2) # third argument is adjustable parameter
        distanceFromOrigin = main_radius * math.sqrt(random.random())
        angle = 2 * math.pi * random.random()

        return main_x_center + distanceFromOrigin * math.cos(angle), main_y_center + distanceFromOrigin * math.sin(angle), radius
    
    
    def _circleIntersection(self, circle1, circle2):
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2

        return (x2 - x1)**2 + (y2 - y1)**2 < (r2 + r1)**2


    def _overlaps(self, circle, fontImg):
        x, y, r = circle
        pi54 = 5 * math.pi / 4
        pi34 = 3 * math.pi / 4
        pi4 = math.pi / 4
        pointsOfInterest = [
            (x, y), (x, y-r), (x, y+r), (x-r, y), (x+r, y),
            (int(x - r * math.cos(pi54)), int(y + r * math.sin(pi54))),
            (int(x - r * math.cos(pi34)), int(y - r * math.sin(pi34))),
            (int(x + r * math.cos(pi4)), int(y - r * math.sin(pi4))),
            (int(x + r * math.cos(-pi4)), int(y + r * math.sin(-pi4)))
        ]

        # For each edge point of circle
        for point in pointsOfInterest:
            # if pixel value (RGB color) at edge point of circle is not background color (hence the place is occupied)
            try:
                if fontImg.getpixel(point)[:3] != self.fontBackgroundColor:
                    # then there is an overlap motive
                    return True
            except:
                # print(f"Exception cought on point=({point[0], point[1]}) with center=({x, y}), image size: (w, h)={fontImg.size}")
                pass

            # otherwise the place for circle point is free so check other

        return False
        

    def _generate(self, name, verbose):
        if verbose: print(f"Generate Ishihara Plate for {name} font:")
        
        fontImg = Image.open(os.path.join(self.fontFolder, name + ".png"))
        width, height = fontImg.size
        self.min_radius = min(width, height) / self.min_radius_factor
        self.max_radius = min(width, height) / self.max_radius_factor
        # print(f"width: {width}, height: {height}\nradius=[{min_radius},{max_radius}]")

        main_radius = min(width, height) // 2
        main_x_center = width // 2
        main_y_center = height // 2

        N = int(1500 * self.density)

        # Generate N non-overlapping circles
        if verbose: printy(f"\t[yB][INFO]@ Generating {N} circles...")

        circle = self._generateCircle(main_x_center, main_y_center, main_radius)
        circles = [circle]
        for _ in range(N):
            attempts = 0
            while any(self._circleIntersection(circle, circle2) for circle2 in circles):
                attempts += 1
                circle = self._generateCircle(main_x_center, main_y_center, main_radius)

            circles.append(circle)

        if verbose: printy(f"\t[nB][+]@ {N} circles are successfully generated!\n")

        getX = lambda circle: circle[0]
        getY = lambda circle: circle[1]
        getR = lambda circle: circle[2]

        # Generate all themes of current file
        for theme in self.THEMES.keys():
            for color_on in self.THEMES[theme]["COLORS_ON"]:
                for color_off in self.THEMES[theme]["COLORS_OFF"]:
                    # For each pair (color_on, color_off) generate Ishihara plate
                    result = Image.new("RGB", fontImg.size, self.fontBackgroundColor)
                    drawer = ImageDraw.Draw(result)

                    # Draw N non-overlapping circles
                    if verbose: printy(f"\t[yB][INFO]@ Drawing {N} circles with colors ({color_on, color_off})...")

                    for circle in circles:
                        if self._overlaps(circle, fontImg):
                            fillColor = color_on
                        else:
                            fillColor = color_off
                        
                        drawer.ellipse((getX(circle) - getR(circle),
                                        getY(circle) - getR(circle),
                                        getX(circle) + getR(circle),
                                        getY(circle) + getR(circle)),
                                        fill=fillColor,
                                        outline=fillColor)
                        
                    if verbose: printy(f"\t[nB][+]@ {N} circles are successfully drawn!\n")

                    # print(f"theme: {theme}, color_on: {colorToHex(color_on)}, color_off: {colorToHex(color_off)}")
                    nameOut = name + "theme_" + theme[self.THEME_NAME_PREFIX_LENGTH:] + "(" + colorToHex(color_on) + "," + colorToHex(color_off) + ")" + ".png"
                    result.save(os.path.join(self.outputFolder, theme[self.THEME_NAME_PREFIX_LENGTH:], nameOut), "PNG")


def main():
    # Ishihara Plates Generation
    IshiharaPlatesGenerator("fonts", "ishiharaPlates").generate()



if __name__ == '__main__':
    main()    

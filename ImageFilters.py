import cv2 as cv
import numpy as np
import sys
from scipy.interpolate import splrep, splev


class ValueNotInRange(Exception):
    def __init__(self, message):
        super().__init__(message)


class NotSingleChannel(Exception):
    def __init__(self, message):
        super().__init__(message)


class filters:
    """Filters Object class"""

    # Filter color maps
    CMAP_AUTMN = cv.COLORMAP_AUTUMN
    CMAP_BONE = cv.COLORMAP_BONE
    CMAP_JET = cv.COLORMAP_JET
    CMAP_WINTER = cv.COLORMAP_WINTER
    CMAP_RAINBOW = cv.COLORMAP_RAINBOW
    CMAP_OCEAN = cv.COLORMAP_OCEAN
    CMAP_SUMMER = cv.COLORMAP_SUMMER
    CMAP_SPRING = cv.COLORMAP_SPRING
    CMAP_COOL = cv.COLORMAP_COOL
    CMAP_HSV = cv.COLORMAP_HSV
    CMAP_PINK = cv.COLORMAP_PINK
    CMAP_HOT = cv.COLORMAP_HOT

    def __init__(
        self,
    ):
        pass

    def gray_to_threechannel(self, img):
        """Streamlit does not display single channel image, this method is used to convert single channel to three channel by stacking same image 3 times"""

        if len(img.shape) > 2:
            raise NotSingleChannel("Image is not a single channel image")

        img = cv.merge([img, img, img])

        return img

    def sketch(
        self,
        img,
        invert=True,
        detail=0.01,
        three_channel=False,
        auto=True,
        thresh1=None,
        thresh2=None,
        size=3,
    ):
        """
        detail should be between 0-1
        sketch like effect filter
        """

        clip_limit = 2.0
        blur = 3

        if detail < 0 or detail > 1:
            raise ValueNotInRange("d not in range 0 to 1")

        detail = max(0.005, detail)
        detail = 1 / detail

        a = int(np.ceil(detail * ((img.shape[0] + img.shape[1]) / 10000)) // 2 * 2 + 1)

        # TODO: for some reason using from interface is using interface.hist_equalize and gblur and not self.hist_equalize and self.gblur, needs fix
        # img = self.hist_equalize(img, clip_limit=2., ksize=(a, a))
        # img = self.gblur(img, ksize=(a, a), blur=blur)

        #  for now pasting same method here
        L, A, B = cv.split(cv.cvtColor(img, cv.COLOR_BGR2LAB))

        # Histogram Eq
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(a, a))
        L = clahe.apply(L)

        img = cv.cvtColor(cv.merge((L, A, B)), cv.COLOR_LAB2BGR)

        img = cv.GaussianBlur(img, (a, a), blur)

        if auto:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            th, _ = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            img = cv.Canny(img, th / 2, th)

        else:
            if thresh1 or thresh2 is None:
                thresh1 = 90
                thresh2 = 90
                print(
                    f"Using default values: thresh1 = {thresh1}, thresh2: {thresh2}, size: {size}"
                )

            img = cv.Canny(
                img, threshold1=thresh1, threshold2=thresh2, apertureSize=size
            )

        if three_channel:
            if invert:
                return self.gray_to_threechannel(img)
            else:
                return self.gray_to_threechannel(cv.bitwise_not(img))

        else:
            if invert:
                return img
            else:
                return cv.bitwise_not(img)

    def gblur(self, img, ksize: tuple = (5, 5), blur: int = 5):
        """
        Make sure ksize is odd numbers
        GaussianBlur filter
        """
        return cv.GaussianBlur(img, ksize, blur)

    def blur(self, img, ksize: tuple = (5, 5)):
        """
        Blur by using mean
        """
        return cv.blur(img, ksize)

    def cartoon(
        self,
        img,
        sigma_color: float,
        sigma_space: float,
        ksize: int = 5,
        iterations: int = 1,
    ):
        for _ in range(iterations):
            img = cv.bilateralFilter(
                img, d=ksize, sigmaColor=sigma_color, sigmaSpace=sigma_space
            )

        return img

    def gray(self, img, three_channel=False):
        """
        Gray scale image
        """
        if three_channel:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            return self.gray_to_threechannel(img)

        else:
            return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    def emboss(self, img, ksize):
        """
        Using Sobel operator to create emboss effect
        """

        sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=ksize)
        sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=ksize)
        sobel = cv.addWeighted(src1=sobelx, alpha=0.5, src2=sobely, beta=0.5, gamma=0)

        # normalise
        sobel = (sobel - np.min(sobel)) / (np.max(sobel) - np.min(sobel))
        return sobel

    def invert(self, img):
        """
        Inverting Filter
        """
        return cv.bitwise_not(img)

    def hist_equalize(self, img, clip_limit: float = 40.0, ksize: tuple = (8, 8)):
        """
        Histogram Equalization Filter
        """
        L, a, b = cv.split(cv.cvtColor(img, cv.COLOR_BGR2LAB))

        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=ksize)
        L = clahe.apply(L)

        return cv.cvtColor(cv.merge((L, a, b)), cv.COLOR_LAB2BGR)

    def adjust_brightness(self, img, param):
        """
        Gammma correction filter
        """

        invGamma = 1.0 / param
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
        return cv.LUT(img.astype(np.uint8), table.astype(np.uint8))

    def color_filters(self, img, color_map):
        """
        Expects cv.ColormapTypes
        uses various builtin color map types from opencv
        """
        return cv.applyColorMap(img, color_map)

    def array_LUT(
        self,
        img,
        x: list = [0, 64, 128, 192, 255],
        y: list = [0, 45, 128, 215, 255],
        bitdepth: int = 8,
    ):
        """
        Barebones version of 'Curves' from photoshop
        accepts input coordinate list[x] with corresponding output coordinate list [y]
        """

        length = pow(2, bitdepth)
        table = self.__interpolate(x=x, y=y, bitdepth=bitdepth)
        print(len(table))
        return cv.LUT(img, table)

    def __interpolate(self, x: list, y: list, bitdepth: int = 8):
        """
        Private interpolation method used for array_LUT method
        """
        length = pow(2, bitdepth)
        spl = splrep(x, y)
        f = splev(np.arange(length), spl)

        return np.array(
            [0 if i < 0 else 255 if i > length else i for i in f], dtype="uint8"
        )

    def hdr(self, img, sigma_s, sigma_r):
        """
        HDR effect
        sigma_s: 0-200
        sigma_r: 0-1
        """
        if sigma_r < 0 or sigma_r > 1:
            raise ValueNotInRange("sigma_r not in range 0-1")

        return cv.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)

    def show(self, img, win_name=None):
        """
        show method to show output
        """
        if win_name is None:
            win_name = str(np.random.choice(a=10, size=1, replace=False)[0])

        cv.namedWindow(win_name, cv.WINDOW_NORMAL)
        cv.imshow(win_name, img)

        key = cv.waitKey(0) & 0xFF
        if key == ord("s"):
            cv.imwrite(
                win_name + ".jpg",
                img,
            )
        if key == ord("q"):
            cv.destroyWindow(win_name)

        # while True:
        #     # if cv.getWindowProperty(win_name, cv.WND_PROP_VISIBLE) < 1:
        #     #     cv.destroyWindow(win_name)
        #     #     break
        #
        #     key = cv.waitKey(0) & 0xFF
        #     if key == ord("s"):
        #         cv.imwrite(
        #             win_name + ".jpg",
        #             img,
        #         )
        #
        #     if key == ord("q"):
        #         cv.destroyWindow(win_name)
        #         break
        #
        #     # if key :
        #     #     break

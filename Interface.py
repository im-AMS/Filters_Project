import ImageFilters
import cv2 as cv
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np


class interface(ImageFilters.filters):
    functions = [
        "Introduction",
        "Blur",
        "Gaussian Blur",
        "Gray",
        "HDR",
        "Emboss",
        "Invert",
        "Histogram_equalize",
        "Adjust_brightness",
        "Color_filters",
        "Sketch",
        "Cartoon",
    ]

    def __init__(self, mode="image"):

        super().__init__()

        st.title("Image Filters using OpenCV")

        self.mode = mode
        self.__default_image = "./assets/default.jpg"
        self.__kid_image = "./assets/kid.png"

        self.upload_handler()
        self.selector()

    def upload_handler(self):
        if self.mode == "image":
            uploaded = st.sidebar.file_uploader(
                "Upload a image",
                type=["png", "jpg", "jpeg"],
                help="Default image is chosen if not uploaded",
            )
            if uploaded is not None:
                self.img = cv.cvtColor(np.array(Image.open(uploaded)), cv.COLOR_RGB2BGR)

            else:
                self.img = cv.imread(self.__default_image)

        elif self.mode == "video":
            pass

    def selector(self):
        func = st.sidebar.selectbox("Select functions", options=self.functions, index=0)

        if func == "Introduction":
            self.introduction()

        elif func == "Blur":
            st.subheader("Blur Filter")
            self.blur()

        elif func == "Gaussian Blur":
            st.subheader("Gaussian Filter")
            self.gblur()

        elif func == "Gray":
            st.subheader("Gray Filter")
            self.gray()

        elif func == "Emboss":
            st.subheader("Emboss Filter")
            self.emboss()

        elif func == "Invert":
            st.subheader("Invert Filter")
            self.invert()

        elif func == "Histogram_equalize":
            st.subheader("Histogram Equalize")
            self.hist_equalize()

        elif func == "Adjust_brightness":
            st.subheader("Adjust brightness/Gamma")
            self.adjust_brightness()

        elif func == "Color_filters":
            st.subheader("Color Filters")
            self.color_filters()

        elif func == "Sketch":
            st.subheader("Sketch Filter")
            self.sketch()

        elif func == "Cartoon":
            st.subheader("Cartoon Filter")
            self.cartoon()

        elif func == "HDR":
            st.subheader("HDR Filter")
            self.hdr()

    def introduction(self):
        st.markdown(
            """
            This is a webapp which allows you to apply multiple filters to your images.

            If no image is uploaded, it uses a default image, you can upload your own too.

            ---

            # About me
            I am currently engineering student, aspiring Data Scientist.
            Feel free to suggest changes.

            I am also open for **collaborations** and **interships**.

            Here are my profiles

            [LinkedIn](www.linkedin.com/in/aditya-ms)

            [Kaggle](https://www.kaggle.com/imams2000)

            [GitHub](https://github.com/im-AMS)

            [Resume](https://drive.google.com/drive/folders/1MzX_eL5B40HFEXgG1c1e7b1ukBbRa1Ha?usp=sharing)


        """,
            unsafe_allow_html=True,
        )

    def show(self, file):
        if self.mode == "image":
            st.image(
                image=file,
                channels="BGR",
            )

        elif self.mode == "video":
            pass

    def blur(self, img=None):

        if img is None:
            img = self.img

        enable = st.sidebar.checkbox("Enable effect", value="True")
        st.sidebar.title("Params for **Regular Blur**")
        a = st.sidebar.slider(
            "kernel size", min_value=1, max_value=50, step=1, value=20
        )

        if enable:
            self.show(file=super().blur(img=img, ksize=(a, a)))

        else:
            self.show(img)

    def gblur(self, img=None):
        if img is None:
            img = self.img

        st.sidebar.title("Params for **Gaussian Blur**")
        enable = st.sidebar.checkbox("Enable effect", value="True")
        a = st.sidebar.slider(
            "kernel size", min_value=3, max_value=49, step=2, value=11
        )
        blur = st.sidebar.slider(
            "Blur strength", min_value=1.0, max_value=20.0, step=0.1, value=11.0
        )

        if enable:
            self.show(file=super().gblur(img=img, ksize=(a, a), blur=blur))
        else:
            self.show(file=img)

    def sketch(self, img=None):

        if img is None:
            img = self.img

        st.sidebar.title("Params for **Sketch**")

        auto = st.sidebar.checkbox("Auto Mode", value="True")
        enable = st.sidebar.checkbox("Enable effect", value="True")
        invert = st.sidebar.checkbox("Invert", value="True")
        detail = st.sidebar.slider(
            "detail", min_value=0.005, max_value=1.0, step=0.005, value=0.13
        )

        if auto:
            file = super().sketch(
                img=img, detail=detail, invert=invert, three_channel=True, auto=True
            )

        else:
            thresh1, thresh2 = st.sidebar.slider(
                "threshold value", value=[42, 200], min_value=0, max_value=255, step=1
            )
            size = st.sidebar.slider("Size", min_value=3, max_value=7, step=2, value=3)

            file = super().sketch(
                img=img,
                thresh1=thresh1,
                thresh2=thresh2,
                size=size,
                detail=detail,
                invert=invert,
                three_channel=True,
                auto=False,
            )

        if enable:
            self.show(file=file)

        else:
            self.show(file=img)

    def gray(self, img=None):
        if img is None:
            img = self.img

        st.sidebar.title("Params for **Gray**")
        enable = st.sidebar.checkbox("Enable effect", value="True")
        st.sidebar.markdown("**None**")

        if enable:
            self.show(file=super().gray(img=img, three_channel=True))
        else:
            self.show(file=img)

    def hdr(self, img=None):
        if img is None:
            img = self.img

        st.sidebar.title("Params for **HDR**")
        enable = st.sidebar.checkbox("Enable effect", value="True")
        param1 = st.sidebar.slider(
            "param1", min_value=0.0, max_value=200.0, step=0.5, value=9.
        )
        param2 = st.sidebar.slider(
            "param2", min_value=0.0, max_value=1.0, step=0.05, value=0.1
        )

        if enable:
            self.show(file=super().hdr(img=img, sigma_s=param1, sigma_r=param2))

        else:
            self.show(file=self.img)

    def emboss(self, img=None):
        if img is None:
            img = self.img

        st.sidebar.title("Params for **Emboss**")
        enable = st.sidebar.checkbox("Enable effect", value="True")
        a = st.sidebar.slider("kernel size", min_value=3, max_value=29, step=2)

        if enable:
            self.show(
                file=super().emboss(
                    img=img,
                    ksize=a,
                )
            )

        else:
            self.show(file=img)

    def invert(self, img=None):

        if img is None:
            img = self.img

        st.sidebar.title("Params for **Invert**")
        st.sidebar.markdown("**None**")

        self.show(file=super().invert(img=img))

    def hist_equalize(self, img=None):

        if img is None:
            img = self.img

        st.sidebar.title("Params for **Histogram Equalize**")

        enable = st.sidebar.checkbox("Enable effect", value="True")
        enable_hist = st.sidebar.checkbox("Show Histogram", value="False")
        a = st.sidebar.slider(
            "kernel size", min_value=3, max_value=99, step=2, value=50
        )
        clip = st.sidebar.slider(
            "Clip limit", min_value=0.0, max_value=200.0, step=0.1, value=3.0
        )

        backend = st.sidebar.selectbox(
            "Select Plotting backend", ["Matplotlib", "Altair"]
        )

        if enable:
            file = super().hist_equalize(img=img, ksize=(a, a), clip_limit=clip)

        else:
            file = self.img

        self.show(file=file)
        if enable_hist:
            self.plot_histogram(img=file, backend=backend)

    def adjust_brightness(self, img=None):

        if img is None:
            img = self.img

        st.sidebar.title("Params for **Adjust Brightness**")
        enable = st.sidebar.checkbox("Enable effect", value="True")
        gamma = st.sidebar.slider(
            "Gamma", min_value=0.01, max_value=10.0, step=0.1, value=0.8
        )
        if enable:
            self.show(file=super().adjust_brightness(img=img, param=gamma))

        else:
            self.show(file=img)

    def color_filters(self, img=None):

        if img is None:
            img = self.img

        cmaps = {
            "AUTMN": self.CMAP_AUTMN,
            "BONE": self.CMAP_BONE,
            "JET": self.CMAP_JET,
            "WINTER": self.CMAP_WINTER,
            "RAINBOW": self.CMAP_RAINBOW,
            "OCEAN": self.CMAP_OCEAN,
            "SUMMER": self.CMAP_SUMMER,
            "SPRING": self.CMAP_SPRING,
            "COOL": self.CMAP_COOL,
            "HSV": self.CMAP_HSV,
            "PINK": self.CMAP_PINK,
            "HOT": self.CMAP_HOT,
        }

        st.sidebar.title("Params for **Color Filters**")
        enable = st.sidebar.checkbox("Enable effect", value="True")
        color = st.sidebar.selectbox(
            "Color Maps",
            cmaps,
        )

        if enable:
            self.show(file=super().color_filters(img=img, color_map=cmaps[color]))

        else:
            self.show(file=img)

    def cartoon(self, img=None):
        if img is None:
            img = cv.imread(self.__kid_image)

        st.sidebar.title("Params for **Cartoon Filter**")
        enable = st.sidebar.checkbox("Enable effect", value="True")
        enable_uploaded = st.sidebar.checkbox("Use uploaded Image", value="false")
        sigma_color = st.sidebar.slider(
            "Sigma Color", min_value=0.01, max_value=90.0, step=0.1, value=18.0
        )
        sigma_space = st.sidebar.slider(
            "Sigma Space", min_value=0.01, max_value=90.0, step=0.1, value=10.0
        )
        a = st.sidebar.slider("kernel size", min_value=2, max_value=20, step=1, value=8)
        iterations = st.sidebar.slider(
            "Number of Passes", min_value=1, max_value=10, step=1, value=8
        )

        if enable:
            if enable_uploaded:
                self.show(
                    file=super().cartoon(
                        img=self.img,
                        sigma_color=sigma_color,
                        sigma_space=sigma_space,
                        ksize=a,
                        iterations=iterations,
                    )
                )
            else:
                self.show(
                    file=super().cartoon(
                        img=img,
                        sigma_color=sigma_color,
                        sigma_space=sigma_space,
                        ksize=a,
                        iterations=iterations,
                    )
                )

        else:
            self.show(file=img)

    def plot_histogram(self, img=None, backend="Matplotlib"):
        """backend = 'Altair' or 'MPL'"""

        if img is None:
            img = self.img

        b, g, r = cv.split(img)

        if backend == "Altair":
            source = pd.DataFrame(
                {
                    "R": r.ravel(),
                    "G": g.ravel(),
                    "B": b.ravel(),
                }
            )

            # TODO: assign right color to channels
            c = (
                alt.Chart(source)
                .transform_fold(["R", "G", "B"], as_=["Channel", "Range"])
                .mark_area(opacity=0.7)
                .encode(
                    alt.X("Range:Q", bin=alt.Bin(maxbins=256), axis=None),
                    alt.Y(
                        "count()", stack=None, axis=None, scale=alt.Scale(type="log")
                    ),
                    alt.Color("Channel:N"),
                )
            )

            st.altair_chart(c, use_container_width=True)

        elif backend == "Matplotlib":
            fig, ax = plt.subplots()
            ax.set_axis_off()
            ax.set_yscale("log")
            st.markdown("**Y axis is semilog**")
            ax.hist(r.ravel(), 256, [0, 256], "r", alpha=0.6)
            ax.hist(g.ravel(), 256, [0, 256], "g", alpha=0.6)
            ax.hist(b.ravel(), 256, [0, 256], "b", alpha=0.6)
            st.pyplot(fig)


if __name__ == "__main__":
    ui = interface()

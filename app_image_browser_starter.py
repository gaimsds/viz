# app_image_browser_starter.py — Week 8: image browser + stats
# Run with:  streamlit run app_image_browser_starter.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import base64
import io
import streamlit.components.v1 as components
from PIL import Image, ImageFilter, ImageOps


def compare(left, right, label_left="original", label_right="processed"):
    """A before/after slider: `left` on the left, `right` on the right.

    streamlit-image-comparison would do this, but it loads juxtapose from
    cdn.knightlab.com/libs/juxtapose/LATEST while the package itself is pinned
    at 0.0.4, and current juxtapose collapses its container to 100x100. This is
    the same thing in plain CSS, with nothing fetched from a CDN.
    """
    def b64(im):
        buf = io.BytesIO(); im.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    right = right.resize(left.size)
    w, h = left.size
    shown_h = int(700 * h / w)
    cap = ("position:absolute;top:8px;background:rgba(0,0,0,.6);color:#fff;"
           "padding:2px 8px;font:13px sans-serif;border-radius:3px")
    components.html(f"""
      <div style="position:relative;max-width:700px">
        <img src="data:image/png;base64,{b64(left)}" style="width:100%;display:block">
        <img id="top" src="data:image/png;base64,{b64(right)}"
             style="width:100%;position:absolute;top:0;left:0;clip-path:inset(0 0 0 50%)">
        <span style="{cap};left:8px">{label_left}</span>
        <span style="{cap};right:8px">{label_right}</span>
        <input type="range" min="0" max="100" value="50" style="width:100%;margin-top:6px"
               oninput="document.getElementById('top').style.clipPath=
                        'inset(0 0 0 '+this.value+'%)'">
      </div>
    """, height=shown_h + 50)

st.title("Image Browser")

tab1, tab2, tab3 = st.tabs(["Digits dataset", "Upload your own", "Before / after"])

with tab1:
    digits = load_digits()
    cls = st.selectbox("Class", sorted(np.unique(digits.target)))
    idxs = np.where(digits.target == cls)[0]
    i = st.slider("Image #", 0, len(idxs) - 1, 0)
    img = digits.images[idxs[i]]
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(); ax.imshow(img, cmap="gray"); ax.axis("off")
        st.pyplot(fig)
    with c2:
        st.metric("Mean intensity", f"{img.mean():.1f}")
        st.metric("Ink (nonzero px)", int((img > 0).sum()))

with tab2:
    up = st.file_uploader("PNG/JPG", type=["png", "jpg", "jpeg"])
    if up:
        pil = Image.open(up)
        arr = np.array(pil)
        st.image(pil, caption=f"shape: {arr.shape}")
        if arr.ndim == 3:
            st.bar_chart({"R": [arr[...,0].mean()], "G": [arr[...,1].mean()], "B": [arr[...,2].mean()]})

with tab3:
    st.caption("Original on the left, processed on the right. Drag the handle.")

    up2 = st.file_uploader("PNG/JPG", type=["png", "jpg", "jpeg"], key="cmp")
    if up2:
        original = Image.open(up2).convert("RGB")
    else:
        # No upload yet: fall back to a digit, scaled up so it is visible.
        st.caption("No image uploaded, so this is a digit from the dataset.")
        d = load_digits().images[0]
        a = (np.clip(d / 16.0, 0, 1) * 255).astype(np.uint8)
        original = Image.fromarray(a).convert("RGB").resize((704, 704), Image.NEAREST)

    step = st.selectbox("Processing", ["Grayscale", "Blur", "Edges", "Invert"])
    if step == "Grayscale":
        processed = ImageOps.grayscale(original).convert("RGB")
    elif step == "Blur":
        processed = original.filter(ImageFilter.GaussianBlur(4))
    elif step == "Edges":
        processed = original.filter(ImageFilter.FIND_EDGES)
    else:
        processed = ImageOps.invert(original)

    compare(original, processed, "original", step.lower())

# ------------------------------------------------------------------
# TODO (pick one):
# 1. Add the class MEAN image next to the selected image
# 2. Add an intensity histogram (np.histogram -> st.bar_chart)
# 3. Channel histograms for uploads (per-channel, overlaid)
# 4. Add a saliency tab: train a classifier once (cache it with
#    @st.cache_resource), then show the occlusion map for the selected image
#    beside the image itself. Two gotchas from the demo -- hide a 3x3 PATCH,
#    not one pixel, or the map comes out blank; and draw it with
#    cmap="inferno", because "hot" tops out at white and disappears.
# ------------------------------------------------------------------

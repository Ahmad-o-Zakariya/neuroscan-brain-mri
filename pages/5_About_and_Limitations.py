import streamlit as st
st.set_page_config(page_title="About | NeuroScan",layout="wide");st.title("About and limitations")
st.markdown("""
### Method
The research pipeline uses an EfficientNet-B0 four-class screening model followed by an independent Attention U-Net binary mask model. The classifier gate decides whether localization runs. The segmentation output is a predicted 2D pixel mask; boxes and centroids are pixel coordinates only.

### Dataset
BRISC 2025: [Kaggle dataset](https://www.kaggle.com/datasets/briscdataset/brisc2025), distributed under CC BY 4.0. Please cite the BRISC dataset authors and retain attribution when using the data. `non_tumorous` may include non-neoplastic space-occupying lesions and is not a guarantee of health.

### Limitations
Images are single-slice contrast-enhanced T1-weighted MRI in axial, coronal, or sagittal planes. The application does not perform 3D volumetric segmentation, identify anatomical lobes, calculate physical dimensions, or provide a diagnosis. Patient IDs were unavailable in the source collections, so patient-level independence cannot be guaranteed. This is research/educational software, not a medical device and not validated for clinical use.
""")

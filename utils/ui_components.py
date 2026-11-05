"""
UI styling and cached model utilities for PhotoRestore AI.
Clean, modern, and colorful styling without emojis or decorative clutter.
"""

from typing import Tuple, Optional
import streamlit as st
import torch

from config import DEVICE
from utils.inference import load_trained_model, check_checkpoint_status


def apply_custom_css():
    """Injects clean, colorful, and minimal CSS for the single-page application."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        /* Base Typography & Background */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F6 100%);
            color: #0F172A;
        }

        /* Hide Streamlit default sidebar and footer clutter */
        [data-testid="stSidebar"] {
            display: none;
        }
        #MainMenu, footer, header {
            visibility: hidden;
        }

        /* Hero Header */
        .restore-hero {
            text-align: center;
            padding: 32px 16px 20px 16px;
            margin-bottom: 24px;
        }

        .restore-title {
            font-size: 38px;
            font-weight: 800;
            letter-spacing: -0.8px;
            margin: 0 0 10px 0;
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.2;
        }

        .restore-subtitle {
            font-size: 16px;
            font-weight: 500;
            color: #64748B;
            margin: 0 auto;
            max-width: 600px;
            line-height: 1.5;
        }

        /* Model Selector Container */
        .model-select-wrapper {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }

        /* Primary Action Button */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #4F46E5 0%, #6366F1 50%, #8B5CF6 100%) !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 24px !important;
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
        }

        div.stButton > button:first-child:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
        }

        /* Download Button */
        div.stDownloadButton > button:first-child {
            background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 24px !important;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
        }

        div.stDownloadButton > button:first-child:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 18px rgba(16, 185, 129, 0.4) !important;
        }

        /* File Uploader */
        [data-testid="stFileUploader"] {
            background: #FFFFFF;
            border: 2px dashed #CBD5E1;
            border-radius: 12px;
            padding: 18px;
            transition: border-color 0.2s ease;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #6366F1;
        }

        /* Image Containers */
        [data-testid="stImage"] img {
            border-radius: 10px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        }

        /* Result Details Badge */
        .status-badge {
            display: inline-block;
            background: #EFF6FF;
            color: #2563EB;
            border: 1px solid #BFDBFE;
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 13px;
            font-weight: 600;
        }

        .status-details {
            font-size: 13px;
            color: #64748B;
            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


@st.cache_resource(show_spinner="Loading trained neural network...")
def get_cached_model(model_type: str) -> Tuple[Optional[torch.nn.Module], str]:
    """Caches loaded model weights to eliminate redundant disk I/O across reruns."""
    return load_trained_model(model_type=model_type, device=DEVICE)


def show_missing_checkpoint_warning(model_type: str):
    """Displays informative guidance when a required model checkpoint is missing."""
    st.error(f"The trained {model_type.upper()} model checkpoint was not found on disk.")
    st.info("Train the model using: python training/train.py --model " + model_type)

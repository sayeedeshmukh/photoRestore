"""
PhotoRestore AI: Single-Page Streamlit Application.
Restores damaged photographs using Deep Learning:
- Pretrained LaMa Inpainting (scratches, cracks, stains, tears)
- Pretrained GFPGAN Facial Prior (reconstructs facial geometry so no face remains empty)
- Pretrained Deep Learning Colorization (infers natural skin tones and clothing colors)
- In-House Trained Convolutional Autoencoder (CAE) & Variational Autoencoder (VAE)
"""

import streamlit as st
from PIL import Image

from config import DEVICE
from utils.ui_components import (
    apply_custom_css,
    get_cached_model,
    show_missing_checkpoint_warning,
)
from utils.image_processing import (
    validate_image_file,
    load_image_to_pil,
    image_to_bytes,
)
from utils.inference import restore_image, check_checkpoint_status


def main():
    st.set_page_config(
        page_title="PhotoRestore AI",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    apply_custom_css()

    # 1. Landing Page Header
    st.markdown(
        """
        <div class="restore-hero">
            <h1 class="restore-title">PhotoRestore AI</h1>
            <p class="restore-subtitle">Restore damaged photographs using Deep Learning.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Checkpoint and weights status verification
    status = check_checkpoint_status()

    # Model & Restoration Pipeline Selection
    options = [
        "Full Restoration (Damage Inpainting + Face Recovery + Colorization + Substrate CAE)",
        "Damage & Face Restoration (Authentic Monochrome - No Colorization)",
        "In-House Convolutional Autoencoder (CAE) Only",
        "In-House Variational Autoencoder (VAE) Only",
    ]
    
    pipeline_choice = st.radio(
        "Select Restoration Pipeline:",
        options=options,
        index=0,
        horizontal=False,
        label_visibility="collapsed"
    )

    # Configure pipeline stages based on selection
    if "Full Restoration" in pipeline_choice:
        run_damage_removal = True
        run_face_restoration = True
        run_colorization = True
        run_substrate_cae = True
        base_model_type = "cae"
        attribution_text = "<b>Multi-Stage Deep Learning:</b> Face-safe LaMa (scratch/crack inpainting) &bull; GFPGAN facial prior (reconstructs all facial structures) &bull; Deep Colorization (skin tones & clothing) &bull; In-house CAE (substrate regularization)."
    elif "Damage & Face" in pipeline_choice:
        run_damage_removal = True
        run_face_restoration = True
        run_colorization = False
        run_substrate_cae = True
        base_model_type = "cae"
        attribution_text = "<b>Monochrome Restoration:</b> Face-safe LaMa scratch removal &bull; GFPGAN facial reconstruction &bull; In-house CAE regularization (preserves original monochrome)."
    elif "CAE Only" in pipeline_choice:
        run_damage_removal = False
        run_face_restoration = False
        run_colorization = False
        run_substrate_cae = True
        base_model_type = "cae"
        attribution_text = "<b>In-House Model:</b> Convolutional Autoencoder trained for patch-based substrate and noise attenuation."
    else: # VAE Only
        run_damage_removal = False
        run_face_restoration = False
        run_colorization = False
        run_substrate_cae = True
        base_model_type = "vae"
        attribution_text = "<b>In-House Model:</b> Variational Autoencoder trained for latent distribution reconstruction."

    st.markdown(
        f"""
        <div style="font-size: 12px; color: #64748B; margin-top: -6px; margin-bottom: 16px; padding: 6px 12px; background: #F1F5F9; border-radius: 6px; border-left: 3px solid #6366F1;">
            {attribution_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Image Upload Option (JPG, JPEG, PNG)
    uploaded_file = st.file_uploader(
        "Upload a damaged photograph (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        help="Accepts RGB and monochrome photographs of any dimension. Preserves original proportions."
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        is_valid, err_msg = validate_image_file(file_bytes)

        if not is_valid:
            st.error(f"Image validation error: {err_msg}")
            return

        input_pil = load_image_to_pil(file_bytes)
        w_orig, h_orig = input_pil.size
        state_key = f"restored_{uploaded_file.name}_{pipeline_choice.split()[0]}"

        # 3. Preview of Uploaded Image
        st.markdown(f"<div style='font-size: 14px; font-weight: 600; color: #334155; margin-bottom: 6px;'>Uploaded Photograph ({w_orig} × {h_orig} px)</div>", unsafe_allow_html=True)
        st.image(input_pil, use_container_width=True)

        # 4. Prominent Restore Photograph Button
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        restore_clicked = st.button("Restore Photograph", type="primary", use_container_width=True)

        # Execute restoration with loading indicator
        if restore_clicked:
            with st.spinner("Restoring photograph with deep learning pipeline..."):
                model = None
                if run_substrate_cae:
                    model, load_msg = get_cached_model(base_model_type)
                    if model is None:
                        st.error(f"Base model failed to load: {load_msg}")
                        return

                restored_pil, latency_ms, faces_count, dmg_pct, was_colorized = restore_image(
                    model=model,
                    image=input_pil,
                    device=DEVICE,
                    restore_original_aspect=True,
                    run_damage_removal=run_damage_removal,
                    run_face_restoration=run_face_restoration,
                    run_colorization=run_colorization,
                    run_substrate_cae=run_substrate_cae,
                    fidelity_weight=0.5,
                )
                st.session_state[state_key] = (restored_pil, latency_ms, faces_count, dmg_pct, was_colorized)

        # 5. Display Restored Photograph & Download Button
        if state_key in st.session_state:
            restored_pil, latency_ms, faces_count, dmg_pct, was_colorized = st.session_state[state_key]

            st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
            
            details = [f"Inference: {latency_ms:.1f} ms", f"Output: {restored_pil.size[0]} × {restored_pil.size[1]} px"]
            if dmg_pct > 0:
                details.append(f"Damage Repaired: {dmg_pct:.1f}%")
            if faces_count > 0:
                details.append(f"Faces Restored: {faces_count}")
            if was_colorized:
                details.append("Colorized: Yes")
                
            details_str = " | ".join(details)
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="status-badge">Restoration Completed</span>
                    <span class="status-details">{details_str}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div style='font-size: 13.5px; font-weight: 600; color: #64748B; margin-bottom: 6px;'>Damaged Photograph</div>", unsafe_allow_html=True)
                st.image(input_pil, use_container_width=True)
            with col2:
                st.markdown("<div style='font-size: 13.5px; font-weight: 600; color: #4F46E5; margin-bottom: 6px;'>Restored Photograph</div>", unsafe_allow_html=True)
                st.image(restored_pil, use_container_width=True)

            # 6. Prominent Download Button
            st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
            png_bytes = image_to_bytes(restored_pil, format="PNG")
            base_name = uploaded_file.name.rsplit(".", 1)[0]
            st.download_button(
                label="Download Restored Photograph",
                data=png_bytes,
                file_name=f"restored_{base_name}.png",
                mime="image/png",
                use_container_width=True
            )


if __name__ == "__main__":
    main()

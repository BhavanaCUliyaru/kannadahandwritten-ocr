import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from scipy import ndimage
import re  # Add this import at the top of the file

def preprocess_image(image, contrast=1.5, brightness=10, binarize=True, denoise=True, deskew=True):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    if deskew:
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        (h, w) = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if binarize:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 11)

    return gray


def recognize_text_tesseract(image):
    custom_config = r'--oem 3 --psm 6 -l kan+eng'
    return pytesseract.image_to_string(image, config=custom_config)

def recognize_text_easyocr(image, reader):
    result = reader.readtext(np.array(image))
    raw_text = ' '.join([text for _, text, _ in result])
    
    # Show raw OCR output for debugging
    st.write("🔍 Raw EasyOCR Output:", raw_text)
    
    # Filter only Kannada characters (Unicode range: U+0C80–U+0CFF)
    kannada_only = ''.join([
    c for c in raw_text
    if (('\u0c85' <= c <= '\u0cb9') or  # Kannada letters (ಅ to ಹ)
        ('\u0cde' == c) or              # Kannada letter ೞ
        ('\u0cbc' <= c <= '\u0ccd') or  # Kannada diacritics (virama, etc.)
        ('\u0ce0' <= c <= '\u0ce1') or  # Kannada vowels ಋ ೠ
        ('\u0cf1' <= c <= '\u0cf2') or  # Rare letters
        c.isspace())
])

    
    # Show filtered Kannada output
    st.write("✅ Kannada-only Output:", kannada_only)
    return kannada_only

def recognize_text_trocr(image, processor, model):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

@st.cache_resource
def load_easyocr_reader():
    return easyocr.Reader(['kn'])
      # 'kn' is the language code for Kannada
    
 # Added English for better recognition of mixed text

@st.cache_resource
def load_trocr_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")  # Using large model for better accuracy
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    return processor, model

def segment_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)

def search_keyword(text, keyword):
    # Convert both text and keyword to lowercase for case-insensitive search
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    # Find all occurrences of the keyword
    matches = list(re.finditer(re.escape(keyword_lower), text_lower))
    
    return matches

def highlight_keyword(text, matches):
    # Convert the text to a list of characters for easier manipulation
    text_chars = list(text)
    
    # Sort matches in reverse order to avoid index issues when adding highlighting
    for match in reversed(matches):
        start, end = match.span()
        text_chars.insert(end, '</span>')
        text_chars.insert(start, '<span style="background-color: blue;">')
    
    # Join the characters back into a string
    return ''.join(text_chars)

def extract_matches(text, matches):
    return [text[match.start():match.end()] for match in matches]

def main():
    st.set_page_config(page_title="Kannada Handwritten Text Recognition and Search", layout="wide")
    st.title("Kannada Handwritten Text Recognition and Search")

    easyocr_reader = load_easyocr_reader()
    trocr_processor, trocr_model = load_trocr_model()

    uploaded_file = st.file_uploader("Choose an image file with Kannada handwritten text", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.header("Image Preprocessing")
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0)
            brightness = st.slider("Brightness", -50, 50, 0)
            binarize = st.checkbox("Binarize Image")
            denoise = st.checkbox("Apply Denoising")
            deskew = st.checkbox("Deskew Image")

            st.header("OCR Options")
            ocr_option = st.radio("Select OCR Method", ["EasyOCR"])
            segment = st.checkbox("Segment Image")

        with col2:
                    if st.button("Perform Text Recognition", key="recognize"):
                        with st.spinner("Processing image..."):
                            preprocessed_image = preprocess_image(image, contrast, brightness, binarize, denoise, deskew)
                            st.image(preprocessed_image, caption="Preprocessed Image", use_column_width=True)

        # 🔍 Perform EasyOCR
                            full_text = recognize_text_easyocr(preprocessed_image, easyocr_reader)
                            st.session_state.full_text = full_text  # ✅ Store filtered Kannada text


        # 📝 Show Kannada text
                            st.subheader("Extracted Kannada Text")
                            st.text_area("Kannada Output", value=full_text, height=200)

        # 💬 Optional feedback
                            st.subheader("Confidence Feedback")
                            confidence = st.slider("How accurate was the recognition? (0-100)", 0, 100, 50)
                            if st.button("Submit Feedback"):
                                st.success("Thank you for your feedback! This will help improve future recognition.")


        # Keyword search section
        st.header("Keyword Search")
        search_keyword_input = st.text_input("Enter a keyword or phrase to search:")
        if search_keyword_input and 'full_text' in st.session_state:
            matches = search_keyword(st.session_state.full_text, search_keyword_input)
            if matches:
                st.success(f"Found {len(matches)} occurrence(s) of '{search_keyword_input}'")
                
                # Display original text with highlights
                st.subheader("Original Text with Highlights:")
                highlighted_text = highlight_keyword(st.session_state.full_text, matches)
                st.markdown(highlighted_text, unsafe_allow_html=True)
                
                # Display matched portions separately
                st.subheader("Matched Portions:")
                matched_texts = extract_matches(st.session_state.full_text, matches)
                for i, match in enumerate(matched_texts, 1):
                    st.markdown(f"{i}. {match}")
            else:
                st.warning(f"No occurrences of '{search_keyword_input}' found in the text.")
                st.markdown(st.session_state.full_text)  # Display the original text if no matches found

    st.sidebar.header("Tips for Better Results")
    st.sidebar.markdown("""
    1. Ensure good lighting and contrast in the image.
     2. Try different preprocessing settings, especially binarization and deskewing.
     3. For large handwritten files, use the 'Segment Image' option to process text in smaller chunks.
     4. EasyOCR often performs well for Indic scripts like Kannada.
     5. Experiment with denoising for images with background noise.
     6. If results are poor, try adjusting the image before uploading (e.g., increase contrast, convert to grayscale).
     7. For mixed Kannada and English text, the system now supports both languages.
     8. When searching for keywords, try variations of the word to account for potential OCR errors.
    """)

    st.sidebar.header("About")
    st.sidebar.info("""
    This app is optimized for recognizing Kannada handwritten text from images, including large files. 
    It offers three OCR methods:
    1. EasyOCR: Generally accurate for handwritten Kannada text.
    2. Tesseract: May work better for printed Kannada text.
    3. TrOCR: A deep learning model for handwritten text recognition.
    The app now includes image segmentation for better handling of large files and supports mixed Kannada-English text.
    New feature: Keyword search allows you to find specific words or phrases within the recognized text.
    """)

if __name__ == "__main__":
    main()

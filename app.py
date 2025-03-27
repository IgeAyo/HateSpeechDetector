import string
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from lime import lime_text
import streamlit.components.v1 as components
from PIL import Image
import pytesseract

# Configure Tesseract path (update this path as per your installation)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load your trained model
model_path = "best_model_rf.joblib"  # Replace with your actual model path
model = joblib.load(model_path)

# Define a function to preprocess text
def preprocess_text(text):
    text = text.lower()
    allowed_chars = "@$%&"
    table = str.maketrans('', '', ''.join(char for char in string.punctuation if char not in allowed_chars))
    text = text.translate(table)
    tokens = TweetTokenizer().tokenize(text)
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Define custom CSS for the logo and the app
def inject_custom_css():
    custom_css = """
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
            color: #333;
        }
        .stMarkdown span {
            font-size: 18px;
            color: #2e2e2e;
        }
        .stMarkdown h1 {
            color: #2e2e2e;
        }
        .logo-container {
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 1000;
        }
        .logo-container img {
            width: 100px;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Function to extract text from an image
def extract_text_from_image(uploaded_image):
    image = Image.open(uploaded_image)
    text = pytesseract.image_to_string(image)
    return text

# Define the Streamlit app
def main():
    inject_custom_css()  # Inject custom CSS

    # Add Nigerian logo to the top left corner
    logo_url = r"C:\Users\igeay\Downloads\ng-circle-01.png"  # Replace with the URL or local path to your logo
    st.markdown(
        f"""
        <div class="logo-container">
            <img src="{logo_url}" alt="Nigerian Logo">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.title('Hate Speech Detection App')
    st.subheader('Enter text to classify, upload a file, or extract text from an image:')

    # Option to choose input method
    text_input_option = st.selectbox("Input method:", ("Text Area", "Upload File", "Upload Image"))

    # Initialize user_input variable
    user_input = ""

    # Input text box for user input
    if text_input_option == "Text Area":
        user_input = st.text_area("Input text here:", "Type Here")
    elif text_input_option == "Upload File":
        uploaded_file = st.file_uploader("Choose a text file:", type="txt")
        if uploaded_file is not None:
            user_input = str(uploaded_file.read(), 'utf-8')
    elif text_input_option == "Upload Image":
        uploaded_image = st.file_uploader("Choose an image file:", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            user_input = extract_text_from_image(uploaded_image)
            st.text_area("Extracted Text:", user_input, height=200)

    if user_input:
        # Preprocess the user input
        processed_input = preprocess_text(user_input)
        # Option to explain the prediction
        explain_prediction = st.checkbox("Explain Prediction")

        # Button to classify text
        if st.button('Detect Hate Speech'):
            # Make prediction and show result
            prediction = model.predict([processed_input])[0]
            prediction_proba = model.predict_proba([processed_input])[0]

            # Define the display styles
            if prediction == 0:
                display_text = f"<span style='color:green; font-weight:bold;'>Non-offensive Speech</span>"
            elif prediction == 1:
                display_text = f"<span style='color:orange; font-weight:bold;'>Offensive Speech</span>"
            else:
                display_text = f"<span style='color:red; font-weight:bold;'>Hate Speech</span>"

            # Display the prediction with styling
            st.markdown(display_text, unsafe_allow_html=True)

            # Show confidence scores
            st.write("Confidence Score (Probability):")
            st.write(f"**Non-offensive Speech (0)**: {prediction_proba[0]:.2f}")
            st.write(f"**Offensive Speech (1)**: {prediction_proba[1]:.2f}")
            st.write(f"**Hate Speech (2)**: {prediction_proba[2]:.2f}")
# Check if probabilities for offensive or hate speech are close to non-offensive speech
            threshold = 0.2  # You can adjust this threshold based on your specific needs
            if abs(prediction_proba[1] - prediction_proba[0]) < threshold or abs(prediction_proba[2] - prediction_proba[0]) < threshold:
                st.markdown("<span style='color:red; font-weight:bold;'>Warning: Probabilities are close, potential hate speech.</span>", unsafe_allow_html=True)

            # Provide explanation using LIME
            if explain_prediction:
                explainer = lime_text.LimeTextExplainer(class_names=["Non-offensive", "Offensive", "Hate Speech"])
                explanation = explainer.explain_instance(processed_input, model.predict_proba, num_features=10)
                # Validate labels in the explanation
                valid_labels = explanation.available_labels()

                offensive_weight = 0
                non_offensive_weight = 0

                if 1 in valid_labels:
                    offensive_weight = sum([weight for term, weight in explanation.as_list(label=1) if weight > 0])
                if 0 in valid_labels:
                    non_offensive_weight = sum([weight for term, weight in explanation.as_list(label=0) if weight > 0])
                    # Check if offensive terms are more prevalent and display a warning
                    # Calculate the difference
                    weight_difference = offensive_weight - non_offensive_weight

                    # Set a threshold for flagging
                    weight_threshold = 0.5  # You can adjust this threshold based on your specific needs

                    if weight_difference > weight_threshold:
                        warning_message = f"<span style='color:red; font-weight:bold;'>Warning: could potentially contain hate speech (Offensive Weight: {offensive_weight:.2f}, Non-offensive Weight: {non_offensive_weight:.2f}, Difference: {weight_difference:.2f})</span>"
                    else:
                        warning_message = f"<span style='color:green; font-weight:bold;'>No significant indicators of hate speech (Offensive Weight: {offensive_weight:.2f}, Non-offensive Weight: {non_offensive_weight:.2f}, Difference: {weight_difference:.2f})</span>"

                    st.markdown(warning_message, unsafe_allow_html=True)

                # Customize the explanation HTML
                explanation_html = explanation.as_html()

                # Customize the explanation HTML
                explanation_html = explanation.as_html()

                # Inject custom CSS into LIME HTML
                custom_style = """
                <style>
                    .lime-explanation {
                        font-family: Arial, sans-serif;
                        color: #2e2e2e;
                    }
                    .lime-explanation .lime {
                        color: #0072b5;
                        font-weight: bold;
                    }
                    .lime-explanation .positive {
                        color: green;
                    }
                    .lime-explanation .negative {
                        color: red;
                    }
                    .lime-explanation h2, .lime-explanation h4 {
                        color: #2e2e2e;
                    }
                </style>
                """

                # Inject the custom style into the LIME HTML
                styled_explanation_html = f"<div class='lime-explanation'>{custom_style}{explanation_html}</div>"

                st.write("Prediction Explanation:")
                components.html(styled_explanation_html, height=800, scrolling=True)
if __name__ == '__main__':
    main()

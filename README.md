# Real-Time License Plate Detection

This project implements a binary classification model to detect the **presence** of a license plate in an image (or video stream). The solution is split into two parts: a **Python script** for model training and dataset preparation, and a **TensorFlow.js web application** for real-time inference using a webcam.

The classification model is based on **MobileNetV2**, trained on custom-cropped images of license plates (`plate` class) and non-plate background images (`no_plate` class).

## Key Features

* **Custom Dataset Generation:** The Python script automatically downloads the source dataset, extracts license plate crops, generates background/negative crops, and splits them into training and validation directories.
* **Transfer Learning:** Uses **MobileNetV2** as a base for fast and accurate binary classification.
* **Web-based Real-Time Detection:** A responsive HTML/JavaScript application using **WebRTC** to access the camera and **TensorFlow.js** (or a simulation) for on-the-fly inference.
* **Responsive UI:** Styled with **Tailwind CSS** and includes a **Dark/Light Mode** toggle.

---

## Project Structure

The repository contains two main files and the resulting dataset/model files:

| File Name | Description |
| :--- | :--- |
| `lpd.py` | **Python** script to download data, pre-process images, and train the MobileNetV2 classification model using Keras/TensorFlow. |
| `minip_webapp.html` | **HTML/JavaScript** file containing the complete web application for real-time webcam detection (currently in simulation mode, awaiting TF.js model conversion). |
| `dataset/` | *Generated directory* containing `train` and `val` folders with `plate` and `no_plate` image crops. |
| `plate_classifier_mobilenetv2.h5` | *Generated file* - The trained Keras model, saved after the training step. |

---

## Setup and Installation

### 1. Python (Training) Setup

The `lpd.py` script is designed to run in a Google Colab environment.

1.  **Prerequisites:** You need a **Kaggle API key** (`kaggle.json`) to download the dataset.
2.  **Dependencies:** Install required libraries (already done in a standard Colab environment):
    ```bash
    pip install tensorflow keras opencv-python-headless kaggle tqdm scikit-learn
    ```
3.  **Run `lpd.py`:**
    * Upload `lpd.py` and your `kaggle.json` to your Colab session.
    * Run the script. It will automatically:
        * Configure the Kaggle API.
        * Download the `andrewmvd/car-plate-detection` dataset.
        * Extract plate crops and generate random non-plate crops.
        * Split data into `train/val` (80/20 split).
        * Train the **MobileNetV2** model.
        * Save the final model as `plate_classifier_mobilenetv2.h5`.

### 2. Web Application Setup

The web application runs entirely in the browser.

1.  **Model Conversion (Crucial Step):** The trained Keras model (`plate_classifier_mobilenetv2.h5`) must be converted to the **TensorFlow.js Layers format**.
    ```bash
    # Install the converter tool
    pip install tensorflowjs
    
    # Run the conversion
    tensorflowjs_converter --input_format=keras \
      plate_classifier_mobilenetv2.h5 \
      ./tfjs_model
    ```
2.  **Hosting:** Upload the `minip_webapp.html` file and the entire converted `tfjs_model/` folder to a simple web server (e.g., GitHub Pages, Vercel, or a local server using Python's `http.server`).
3.  **Configuration:** Update the `MODEL_URL` variable inside `minip_webapp.html` to point to the `model.json` file within your hosted `tfjs_model` directory.
    ```javascript
    // Inside minip_webapp.html
    // const MODEL_URL = '[http://your-server.com/tfjs_model/model.json](http://your-server.com/tfjs_model/model.json)'; // CHANGE THIS URL
    ```
4.  **Enabling Detection:** Uncomment the three commented-out blocks (`// 1.`, `// 2.`, and `// 3. Optional:`) in the `<script>` section of `minip_webapp.html` and implement the TF.js prediction logic within the `detectPlate` function.

---

## Web App Usage

1.  Open `minip_webapp.html` in your browser.
2.  Click the **"Start Camera"** button.
3.  Grant camera permission when prompted.
4.  The application will start the real-time detection loop, displaying **"PLATE = 1"** or **"NO PLATE = 0"** based on the model's prediction of the video stream.
5.  Click the **"Stop Camera"** button to end the stream.

---

## Contact

This project was developed by:

* **Bushra Farhad:** [linkedin.com/in/bushra-farhad-518a52245/](https://www.linkedin.com/in/bushra-farhad-518a52245/)

Feel free to connect or reach out with any questions!

## Future Enhancements

- Integrate the trained TF.js model for real-time plate detection
- Add bounding box drawing in live view
- Extend model for OCR (Optical Character Recognition) to extract license numbers
- Deploy web app using GitHub Pages or Streamlit Cloud

## License

This project is released under the MIT License.
Feel free to use, modify, and share with proper attribution.

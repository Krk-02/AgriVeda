import os
import openai
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
# import pinecone
from pinecone import Pinecone,ServerlessSpec
# import pinecone
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile

def predict_disease(image_path):
    model = tf.keras.models.load_model(r'.\trained_model (1).keras')
    # model.summary()  # Verify input shape is (None, 128, 128, 3)

    # Load and preprocess image
    image_pth = image_path  # Path to your image
    image = tf.keras.preprocessing.image.load_img(image_pth, target_size=(128, 128))  # Resize to 128x128

    # Convert to array and add batch dimension
    input_arr = tf.keras.preprocessing.image.img_to_array(image)  # Shape: (128, 128, 3)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dim → (1, 128, 128, 3)

    # Predict
    prediction = model.predict(input_arr)
    # Now compatible shape

    result_index = np.argmax(prediction)
    # print(prediction[0][result_index])
    # # print(result_index)
    # if prediction[0][result_index] < 0.99 :
    #     print("Not a disease/ Not able to predict")
    CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
    ]
    # print(CLASS_NAMES[result_index])
    return CLASS_NAMES[result_index], prediction[0][result_index]   

load_dotenv()
# Initialize APIs
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Configuration
INDEX_NAME = "crop-rag-openai"
CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
def crop_chat(query):
    """Enhanced RAG query with OpenAI GPT"""
    index = pc.Index(INDEX_NAME)
    try:
        # Generate query embedding
        # expanded_queries = expand_query(query)
        response = openai.embeddings.create(
            input=[query],
            model=EMBEDDING_MODEL
        )
        query_embedding = response.data[0].embedding  
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter={"source": {"$exists": True}}
        )
        # embeddings = []
        # for q in expanded_queries:
        #     response = openai.embeddings.create(
        #         input=[q],
        #         model=EMBEDDING_MODEL
        #     )
        #     embeddings.append(response.data[0].embedding)
        
        # # Average the embeddings
        # avg_embedding = [sum(values)/len(values) for values in zip(*embeddings)]
        
        # Search with averaged embedding
        # results = index.query(
        #     vector=avg_embedding,
        #     top_k=5,
        #     include_metadata=True,
        #     filter={
        #         "source": {"$exists": True}
        #     }
        # )
        if not results.matches:
            return "No relevant information found. Please ask another question about crop."
        
        # Build context
        context = []
        for match in results.matches:
            context.append(
                f"SOURCE: {match.metadata['source']}\n"
                f"CONTENT: {match.metadata['text']}\n"
                f"RELEVANCE: {match.score:.2f}"
            )
        context_str = "\n\n".join(context)
        
        # Generate response
        response = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a knowledgeable crop expert assistant. Use the context to answer questions in a friendly, engaging manner, also if there is a disease detected in the image, you must mention it in local indian languages "
                "Prompt the user for location if required and based on this respond with the local name"},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error processing your request: {str(e)}"

app = Flask(__name__)
CORS(app)  # Enable CORS if needed
app.config['UPLOAD_FOLDER'] = r'.\uploads'  # Folder to save uploaded images

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        # Handle form data
        text_input = request.form.get('text', '')
        image_file = request.files.get('image')
        
        # Initialize query
        final_query = text_input
        
        # Process image if provided
        disease = None
        if image_file:
            # Validate image
            if not image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                return jsonify({
                    "status": "error",
                    "message": "Invalid image format. Only PNG, JPG, JPEG allowed."
                }), 400
            
            # Save temporary image
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(temp_path)
            print(temp_path)
            try:
                # Get disease prediction
                disease, confidence = predict_disease(temp_path)
                final_query += f" | Detected Disease: {disease} (Confidence: {confidence*100:.1f}%)"
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": f"Image processing failed: {str(e)}"
                }), 500
            # finally:
                # Clean up temp file
                # if os.path.exists(temp_path):
                #     os.remove(temp_path)
        
        # Process query through RAG pipeline
        if not final_query.strip():
            return jsonify({
                "status": "error",
                "message": "No input provided. Please send text or image."
            }), 400
            
        try:
            response = crop_chat(final_query)
            return jsonify({
                "status": "success",
                "message": response,
                "detected_disease": disease if disease else None
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Query processing failed: {str(e)}"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

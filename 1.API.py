#A python implementation to setup an endpoint serving and embedding for the asked query.



from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load a pre-trained model (you can choose a model from Hugging Face Sentence Transformers)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

@app.route('/get_embeddings', methods=['POST'])
def get_embeddings():
    try:
        # Get input text from the request
        data = request.get_json()
        text = data['text']

        # Generate embeddings for the input text
        embeddings = model.encode(text)

        # Convert the embeddings to a list for JSON serialization
        embeddings_list = embeddings.tolist()

        # Prepare the response
        response = {
            'embeddings': embeddings_list
        }

        return jsonify(response)

    except Exception as e:
        error_message = {'error': str(e)}
        return jsonify(error_message), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0")

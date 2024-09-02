from flask import Flask, request, jsonify, render_template
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load Excel data
df = pd.read_excel("BLEH.xlsx")
teacher_data = []
for index, row in df.iterrows():
    teacher_data.append({
        "name": row["Faculty name"],
        "department": row["Dpartments"],
        "email": row["E-mail id"],
        "room": row["ROOM NO"]
    })

# Preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Teacher information lookup function
def find_teacher(user_query):
    user_query_processed = preprocess_text(user_query)
    vectorizer = TfidfVectorizer()
    teacher_names = [teacher["name"] for teacher in teacher_data]
    vectorizer.fit(teacher_names)
    user_vector = vectorizer.transform([user_query_processed])
    teacher_vectors = vectorizer.transform(teacher_names)
    similarity_scores = cosine_similarity(user_vector, teacher_vectors)
    most_similar_index = similarity_scores.argmax()
    return teacher_data[most_similar_index]

# Format teacher information for response
def get_teacher_info(teacher):
    response = f"*{teacher['name']}*\n"
    response += f"Department: {teacher['department']}\n"
    response += f"Email: {teacher['email']}\n"
    response += f"Room: {teacher['room']}"
    return response

# Load language model for general question answering
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)

def answer_question(user_query):
    input_ids = tokenizer.encode("answer: " + user_query, return_tensors="pt")
    output = model.generate(input_ids, max_length=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Determine the type of query
def interpret(user_query):
    keywords = ["teacher", "faculty", "professor"]
    for keyword in keywords:
        if keyword in user_query.lower():
            return "teacher"
    return "general"

# Route to render the landing page
@app.route('/')
def front():
    return render_template('front.html')

# Route to render the chatbot interface
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# Route to process user queries
@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.json
    user_query = data.get('message')
    
    intent = interpret(user_query)
    if intent == "teacher":
        teacher = find_teacher(user_query)
        response = get_teacher_info(teacher)
    else:
        response = answer_question(user_query)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def get_somaiya_response(message):
    message = message.lower()
    if "hello" in message or "hi" in message:
        return "Hello! I'm somAIya, your college chatbot. How can I assist you today?"
    elif "majors offered" in message:
        return "SomAIya offers a variety of majors including Computer Science, Information Technology, Electronics, Mechanical Engineering, and more."
    elif "campus size" in message:
        return "The SomAIya campus spans over 85 acres, providing state-of-the-art facilities and green spaces for students."
    elif "goodbye" in message or "bye" in message:
        return "Goodbye! Have a great day. If you have more questions, feel free to come back!"
    else:
        return "I'm somAIya! Ask me about majors offered, campus size, or anything else about our college."

@app.route('/')
def home():
    return ('HI')
 
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/front')
def front():
    return render_template('front.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json.get("message")
    bot_response = get_somaiya_response(user_message)
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(debug=True)

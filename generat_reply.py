import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from transformers import AutoModelForCausalLM, AutoTokenizer, Conversation, ConversationalPipeline

# Load Q&A dataset
with open("bot_faq.json", "r") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]
# answers = [item["answer"] for item in faq_data]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer().fit(questions)
question_vectors = vectorizer.transform(questions)

# Load DialoGPT for fallback
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
# chatbot = ConversationalPipeline(model=model, tokenizer=tokenizer)

# def get_faq_response(user_text):
#     user_vec = vectorizer.transform([user_text])
#     sim = cosine_similarity(user_vec, question_vectors)
#     idx = sim.argmax()
#     if sim[0][idx] > 0.5:  # threshold for similarity
#         return answers[idx]
#     return None

# import random

# def get_random_answer(question, faq_data):
#     for item in faq_data:
#         if question.lower() == item["question"].lower():
#             return random.choice(item["answers"])
#     return None


def get_faq_response(user_text, threshold=0.5):
    """
    Returns a random answer from the most similar question if similarity > threshold.
    """
    user_vec = vectorizer.transform([user_text])
    sim = cosine_similarity(user_vec, question_vectors)
    idx = sim.argmax()
    
    if sim[0][idx] > threshold:
        # pick a random answer from faq_data[idx]['answers']
        return random.choice(faq_data[idx]['answers'])
    return None

# # Example loop with currency + conversational fallback
# conv = Conversation()


# print("Bot: Hello! Ask me anything.")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["exit", "quit"]:
#         break

#     # Check currency
#     currency_reply = handle_currency_conversion(user_input)
#     if currency_reply:
#         print("Bot:", currency_reply)
#         continue

#     # Check FAQ
#     faq_reply = get_faq_response(user_input)
#     if faq_reply:
#         print("Bot:", faq_reply)
#         continue

#     # Fallback to DialoGPT
#     conv.add_user_input(user_input)
#     output = chatbot(conv)
#     print("Bot:", output.generated_responses[-1])

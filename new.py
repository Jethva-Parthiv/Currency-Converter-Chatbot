from flask import Flask, render_template, request, jsonify
import re
import spacy
import random
from currencyconvert import get_conversion_rate
from transformers import pipeline
from generat_reply import get_faq_response
from word2number import w2n

# Load a lightweight conversational model for demo purposes
# (For production, use GPT-2 or better models and tune finetuning, as in web:2)
# chatbot_nlp = pipeline("conversational", model="microsoft/DialoGPT-small")

chatbot_nlp = pipeline("text-generation", model="microsoft/DialoGPT-small")
# ----------------------------
# Load NLP and mappings
# ----------------------------
nlp = spacy.load("en_core_web_sm")

SYMBOL_MAP = {
    "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"
}

CURRENCY_MAP = {
    "dollar": "USD", "dollars": "USD", "usd": "USD",
    "rupee": "INR", "rupees": "INR", "rs": "INR", "inr": "INR",
    "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "yen": "JPY", "jpy": "JPY",
    "pound": "GBP", "pounds": "GBP", "gbp": "GBP",
    "aud": "AUD", "cad": "CAD", "chf": "CHF", "cny": "CNY",
    "sgd": "SGD", "hkd": "HKD", "nzd": "NZD", "krw": "KRW",
    "rub": "RUB", "brl": "BRL", "zar": "ZAR", "mxn": "MXN",
    "aed": "AED", "sek": "SEK", "nok": "NOK", "dkk": "DKK",
    "ils": "ILS", "try": "TRY", "pln": "PLN", "thb": "THB",
    "php": "PHP", "idr": "IDR", "myr": "MYR", "vnd": "VND"
}
KNOWN_CURRENCY_CODES = set(v for v in CURRENCY_MAP.values())

# ----------------------------
# Entity extraction (reuse your function)
# ----------------------------

MULTIPLIERS = {
    "thousand": 1000,
    "million": 1000000,
    "billion": 1000000000,
    "lakh": 100000,       # Indian style
    "crore": 10000000
}

def _parse_text_number(text: str):
    """
    Parse number words or mixed text+digits like '10 thousand' or 'two million'.
    """
    text = text.lower().strip()

    # Case 1: Try direct word2number conversion
    try:
        return w2n.word_to_num(text)
    except:
        pass

    # Case 2: Check for "<number> <multiplier>" (both digits and words)
    m = re.match(r"(\d+(\.\d+)?|\w+)\s+(thousand|million|billion|lakh|crore)", text)
    if m:
        base, _, mult_word = m.groups()
        # Convert base part (could be digit or word)
        try:
            base_val = float(base) if re.match(r"\d+(\.\d+)?", base) else w2n.word_to_num(base)
            return int(base_val * MULTIPLIERS[mult_word])
        except:
            return None

    return None


def _parse_number(s: str):
    s = s.replace(",", "")
    try:
        return float(s)
    except:
        return None
    

def extract_entities_and_intent(text: str):
    orig = text
    text_low = text.lower()
    doc = nlp(text_low)
    amount = None
    source_currency = None
    target_currency = None

    # 1) Detect patterns with currency symbols: $20 or 20$
    m = re.search(r"(?P<sym>[$€£¥₹])\s*(?P<amt>\d[\d,]*\.?\d*)", orig)
    if m:
        amount = _parse_number(m.group("amt"))
        source_currency = SYMBOL_MAP.get(m.group("sym"))

    if amount is None:
        m = re.search(r"(?P<amt>\d[\d,]*\.?\d*)\s*(?P<sym>[$€£¥₹])", orig)
        if m:
            amount = _parse_number(m.group("amt"))
            source_currency = SYMBOL_MAP.get(m.group("sym"))

    # 2) Detect patterns like "20usd" or "20 usd"
    if amount is None:
        m = re.search(r"(?P<amt>\d[\d,]*\.?\d*)\s*(?P<code>[A-Za-z]{3})\b", orig)
        if m and m.group("code").upper() in KNOWN_CURRENCY_CODES:
            amount = _parse_number(m.group("amt"))
            source_currency = m.group("code").upper()

    # 3) Detect patterns like "usd 20"
    if amount is None:
        m = re.search(r"\b(?P<code>[A-Za-z]{3})\s*(?P<amt>\d[\d,]*\.?\d*)\b", orig)
        if m and m.group("code").upper() in KNOWN_CURRENCY_CODES:
            amount = _parse_number(m.group("amt"))
            source_currency = m.group("code").upper()

    # 4) Fallback to spaCy number detection
    if amount is None:
        for token in doc:  
            if token.like_num:
                amount = _parse_number(token.text)
                # look for "X <num>" or "<num> X"
                neigh = re.search(r"([A-Za-z₹$€£¥]{1,4})\s*" + re.escape(token.text), text_low)
                if neigh:
                    cand = neigh.group(1)
                    if cand in CURRENCY_MAP:
                        source_currency = CURRENCY_MAP[cand]
                    elif len(cand) == 3 and cand.upper() in KNOWN_CURRENCY_CODES:
                        source_currency = cand.upper()
                neigh2 = re.search(re.escape(token.text) + r"\s*([A-Za-z₹$€£¥]{1,4})", text_low)
                if neigh2 and source_currency is None:
                    cand = neigh2.group(1)
                    if cand in CURRENCY_MAP:
                        source_currency = CURRENCY_MAP[cand]
                    elif len(cand) == 3 and cand.upper() in KNOWN_CURRENCY_CODES:
                        source_currency = cand.upper()
                break

    # 4b) Detect written numbers (single / multiword / with multipliers)
    if amount is None:
        words = text_low.split()
        for i in range(len(words)):
            # Try up to 3-word spans: "twenty five", "10 thousand", etc.
            for j in range(i+1, min(len(words), i+3)+1):
                phrase = " ".join(words[i:j])
                cand = _parse_text_number(phrase)
                if cand is not None:
                    amount = cand
                    break
            if amount is not None:
                break

    # 5) Collect all currency mentions
    found = []
    for k, v in CURRENCY_MAP.items():
        if re.search(r"\b" + re.escape(k) + r"\b", text_low):
            if v not in found:
                found.append(v)

    for code in re.findall(r"\b([A-Za-z]{3})\b", orig):
        cu = code.upper()
        if cu in KNOWN_CURRENCY_CODES and cu not in found:
            found.append(cu)

    for sym, cu in SYMBOL_MAP.items():
        if sym in orig and cu not in found:
            found.append(cu)

    # 6) Explicit "X to Y" pattern (highest priority)
    m_pair = re.search(r"\b(?P<src>[A-Za-z₹$€£¥]{1,4})\s+to\s+(?P<dst>[A-Za-z₹$€£¥]{1,4})\b", text_low)
    if m_pair:
        cand_src, cand_dst = m_pair.group("src"), m_pair.group("dst")
        # resolve source
        if cand_src in SYMBOL_MAP: source_currency = SYMBOL_MAP[cand_src]
        elif cand_src in CURRENCY_MAP: source_currency = CURRENCY_MAP[cand_src]
        elif len(cand_src) == 3 and cand_src.upper() in KNOWN_CURRENCY_CODES: source_currency = cand_src.upper()
        # resolve target
        if cand_dst in SYMBOL_MAP: target_currency = SYMBOL_MAP[cand_dst]
        elif cand_dst in CURRENCY_MAP: target_currency = CURRENCY_MAP[cand_dst]
        elif len(cand_dst) == 3 and cand_dst.upper() in KNOWN_CURRENCY_CODES: target_currency = cand_dst.upper()

    # 7) Detect "in|to|of <currency>" (target only if not already set)
    if target_currency is None:
        m_to = re.search(r"\b(?:in|to|of)\s+(?P<cand>[A-Za-z₹$€£¥]{1,4})\b", text_low)
        if m_to:
            cand = m_to.group("cand")
            if cand in SYMBOL_MAP:
                target_currency = SYMBOL_MAP[cand]
            elif cand in CURRENCY_MAP:
                target_currency = CURRENCY_MAP[cand]
            elif len(cand) == 3 and cand.upper() in KNOWN_CURRENCY_CODES:
                target_currency = cand.upper()

    # 8) Fallback: use found list
    if source_currency is None and found:
        source_currency = found[0]

    if target_currency is None:
        if len(found) > 1:
            for c in found:
                if c != source_currency:
                    target_currency = c
                    break

    # 9) Final intent detection
    if amount is not None:
        intent = "currency_conversion"
    elif "rate" in text_low or "conversion" in text_low:
        intent = "exchange_rate"
    elif any(k in text_low for k in ("convert", "how many", "how much", " in ", " to ", " of ")):
        intent = "currency_conversion"
    else:
        intent = "unknown"

    return {
        "intent": intent,
        "entities": {
            "amount": amount,
            "source_currency": source_currency,
            "target_currency": target_currency
        }
    }


responses = [
    "{amount} {base} equals {converted:.2f} {target} today.",
    "You'll get about {converted:.2f} {target} for {amount} {base}.",
    "If you exchange {amount} {base}, you'll receive ~{converted:.2f} {target}.",
    "At today’s rate, {amount} {base} = {converted:.2f} {target}.",
    "Sure thing! {amount} {base} will get you about {converted:.2f} {target} right now.",
    "At the current rate, {amount} {base} comes to roughly {converted:.2f} {target}.",
    "Exchanging {amount} {base}? You’d get around {converted:.2f} {target} today.",
    "{amount} {base} equals about {converted:.2f} {target} at today’s rate.",
    "Right now, {amount} {base} is worth approximately {converted:.2f} {target}."
]

def format_response(amount, base, target, converted):
    template = random.choice(responses)
    return template.format(amount=amount, base=base.upper(),
                           target=target.upper(), converted=converted)


def same_currency_reply(currency):
    replies = [
        f"Converting {currency} to {currency}? Wow, mind blown 🤯!",
        f"Hmm... {currency} is already {currency}. Are we practicing math? 😄",
        f"Trying to turn {currency} into more {currency}? Magic! 🪄",
        f"{currency} → {currency}. That's a classic! 😂",
        f"Did you just try to duplicate {currency}? Nice try 😎",
        f"Oops! {currency} to {currency} is like copying homework 😜"
    ]
    return random.choice(replies)


# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")
    data = extract_entities_and_intent(user_msg)
    base = data['entities']['source_currency']
    target = data['entities']['target_currency']
    amount = data['entities']['amount']

    if base is None and target is None:
        # Use NLP model to generate conversational responses
        # nlp_response = chatbot_nlp(user_msg)
        # nlp_response = chatbot_nlp(user_msg, max_length=100, pad_token_id=50256)
        # reply = str(nlp_response[0]['generated_text'])
        
        reply = get_faq_response(user_msg)
        if reply == None : reply = "I can't understand!!"
        return jsonify({"reply": reply})

    if base is None :
        return jsonify({"reply": "I can't understand!!"})

    if target is None and base != 'inr':
        target = 'inr'
    
    if base == target :
        reply = same_currency_reply(target)
        return jsonify({"reply": reply})

    if amount is None:
        amount = 1

    try:
        rate = get_conversion_rate(base, target)
        conv = round(amount * rate, 2)
        reply = format_response(amount, base, target, conv)
    except:
        reply = "Sorry, I couldn't fetch the rate right now."
    print(reply)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)




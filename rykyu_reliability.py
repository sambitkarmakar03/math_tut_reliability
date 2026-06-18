"""
RYKYU Math Tutor — LLM Reliability Evaluation
==============================================
Run: python rykyu_reliability.py
Requires: pip install tensorflow tensorflow-metal sentence-transformers transformers torch requests wikipedia ddgs pandas
Make sure Ollama is running: ollama serve
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import requests
import wikipedia
import tensorflow as tf
from functools import lru_cache
from ddgs import DDGS
from transformers import pipeline
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "rykyu-math-tutor"
NUM_RUNS    = 3       # responses per prompt for self-consistency
TEMPERATURE = 0.8
NUM_PREDICT = 100

# ── GPU SETUP ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("RYKYU Math Tutor — LLM Reliability Analysis")
print("=" * 70)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"Metal GPU: ACTIVE ({gpus[0].name})")
else:
    print("GPU not detected — running on CPU")

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
print("\nLoading embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading NLI model (roberta-large-mnli)...")
nli_analyzer = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    framework="tf"
)
print("Models loaded.\n")

# ── DATASET — 120 MATH PROMPTS (40 per level) ─────────────────────────────────
dataset = [

    # ── LEVEL 1 : Addition & Subtraction ──────────────────────────────────────
    {"prompt": "Sam has 12 apples and gives away 5. How many does he have left?", "reference": "7"},
    {"prompt": "A shop has 34 items. 18 more arrive. How many items in total?", "reference": "52"},
    {"prompt": "There are 25 birds on a tree. 9 fly away. How many remain?", "reference": "16"},
    {"prompt": "Priya has 47 stickers. She gets 13 more. How many stickers does she have?", "reference": "60"},
    {"prompt": "A bag has 30 marbles. 11 are lost. How many marbles are left?", "reference": "19"},
    {"prompt": "There are 56 students in a hall. 24 leave. How many are still there?", "reference": "32"},
    {"prompt": "A farmer picks 68 oranges on Monday and 27 on Tuesday. How many in total?", "reference": "95"},
    {"prompt": "A library has 120 books. 45 are borrowed. How many remain?", "reference": "75"},
    {"prompt": "Tom has 15 red balls and 22 blue balls. How many balls does he have?", "reference": "37"},
    {"prompt": "There were 80 passengers on a bus. 33 got off. How many are left?", "reference": "47"},
    {"prompt": "A fruit shop sold 19 mangoes in the morning and 36 in the evening. How many mangoes were sold?", "reference": "55"},
    {"prompt": "Riya had 200 rupees. She spent 75. How much does she have left?", "reference": "125"},
    {"prompt": "A school planted 50 trees. 17 did not survive. How many trees survived?", "reference": "33"},
    {"prompt": "There are 14 boys and 19 girls in a class. How many students are there?", "reference": "33"},
    {"prompt": "A pond had 90 fish. 28 were caught. How many fish remain?", "reference": "62"},
    {"prompt": "John scored 45 in math and 38 in science. What is his total score?", "reference": "83"},
    {"prompt": "A jar has 100 candies. 43 are eaten. How many are left?", "reference": "57"},
    {"prompt": "There are 72 pages in a book. Maya read 29. How many pages are left?", "reference": "43"},
    {"prompt": "A store had 55 shirts. They sold 18. How many shirts remain?", "reference": "37"},
    {"prompt": "Arjun has 63 coins. He gives 27 to his friend. How many does he have left?", "reference": "36"},
    {"prompt": "A basket had 40 tomatoes. 15 were rotten. How many are good?", "reference": "25"},
    {"prompt": "A class collected 88 notebooks. They distributed 34. How many are left?", "reference": "54"},
    {"prompt": "There are 150 seats in a hall. 97 are occupied. How many are empty?", "reference": "53"},
    {"prompt": "Meena saved 320 rupees in January and 180 in February. How much did she save in total?", "reference": "500"},
    {"prompt": "A train had 240 passengers. At a station 75 got off and 40 got on. How many passengers are on the train now?", "reference": "205"},
    {"prompt": "A vendor has 95 bananas. He sells 48. How many are left?", "reference": "47"},
    {"prompt": "There are 37 red flowers and 28 yellow flowers in a garden. How many flowers are there?", "reference": "65"},
    {"prompt": "A box had 110 pens. 63 were given out. How many pens remain?", "reference": "47"},
    {"prompt": "Rita scored 76 in English and 54 in Hindi. What is her combined score?", "reference": "130"},
    {"prompt": "A swimming pool had 500 liters of water. 175 liters were drained. How many liters remain?", "reference": "325"},
    {"prompt": "A shelf has 85 books. 29 are removed. How many books are on the shelf?", "reference": "56"},
    {"prompt": "A worker earns 450 rupees a day. He spends 180. How much does he save?", "reference": "270"},
    {"prompt": "A zoo had 64 animals. 19 were transferred. How many animals remain?", "reference": "45"},
    {"prompt": "Raj has 23 toy cars and gets 17 more for his birthday. How many toy cars does he have?", "reference": "40"},
    {"prompt": "A canteen made 300 rotis. 134 were served. How many are left?", "reference": "166"},
    {"prompt": "There are 48 chairs in a room. 15 more are added. How many chairs are there now?", "reference": "63"},
    {"prompt": "A pond has 77 ducks. 32 fly away. How many ducks remain?", "reference": "45"},
    {"prompt": "Kavya has 500 rupees. She buys a book for 125 and a pen for 35. How much money is left?", "reference": "340"},
    {"prompt": "A shop received 200 packets of chips. They sold 148. How many packets are left?", "reference": "52"},
    {"prompt": "There are 91 players in a tournament. 46 are eliminated in round one. How many remain?", "reference": "45"},

    # ── LEVEL 2 : Multiplication ───────────────────────────────────────────────
    {"prompt": "A box has 6 rows of 8 chocolates. How many chocolates in total?", "reference": "48"},
    {"prompt": "Each student needs 4 pencils. There are 9 students. How many pencils are needed?", "reference": "36"},
    {"prompt": "A farmer plants 7 rows of 5 trees. How many trees did he plant?", "reference": "35"},
    {"prompt": "A packet contains 12 biscuits. How many biscuits are in 5 packets?", "reference": "60"},
    {"prompt": "There are 8 baskets each with 9 apples. How many apples in total?", "reference": "72"},
    {"prompt": "A school has 6 classrooms each with 30 students. How many students in total?", "reference": "180"},
    {"prompt": "A car travels 60 km in one hour. How far does it travel in 4 hours?", "reference": "240"},
    {"prompt": "A garden has 5 rows of 11 flowers. How many flowers are there?", "reference": "55"},
    {"prompt": "Each bag weighs 3 kg. How much do 14 bags weigh?", "reference": "42"},
    {"prompt": "A factory makes 25 toys per day. How many toys in 6 days?", "reference": "150"},
    {"prompt": "There are 7 days in a week. How many days are there in 8 weeks?", "reference": "56"},
    {"prompt": "A bookshelf has 9 shelves each holding 12 books. How many books in total?", "reference": "108"},
    {"prompt": "Each box holds 15 oranges. How many oranges are in 7 boxes?", "reference": "105"},
    {"prompt": "A builder lays 20 bricks per hour. How many bricks in 9 hours?", "reference": "180"},
    {"prompt": "A spider has 8 legs. How many legs do 6 spiders have?", "reference": "48"},
    {"prompt": "A class reads 3 chapters a day. How many chapters in 11 days?", "reference": "33"},
    {"prompt": "An egg tray holds 12 eggs. How many eggs in 8 trays?", "reference": "96"},
    {"prompt": "Each row in a cinema has 22 seats. How many seats in 10 rows?", "reference": "220"},
    {"prompt": "A bicycle has 2 wheels. How many wheels do 16 bicycles have?", "reference": "32"},
    {"prompt": "A shirt costs 250 rupees. What is the cost of 4 shirts?", "reference": "1000"},
    {"prompt": "A farmer harvests 45 kg of wheat per day. How much in 5 days?", "reference": "225"},
    {"prompt": "There are 24 hours in a day. How many hours in 7 days?", "reference": "168"},
    {"prompt": "A box of 6 pens costs 30 rupees. How much do 3 boxes cost?", "reference": "90"},
    {"prompt": "A school bus makes 4 trips a day. How many trips in 5 days?", "reference": "20"},
    {"prompt": "Each child gets 3 sweets. There are 18 children. How many sweets are needed?", "reference": "54"},
    {"prompt": "A train has 12 compartments each with 50 seats. How many seats in total?", "reference": "600"},
    {"prompt": "A field is 13 meters long and 6 meters wide. What is its area?", "reference": "78"},
    {"prompt": "A shop sells 35 bottles of water per hour. How many bottles in 8 hours?", "reference": "280"},
    {"prompt": "Each notebook has 80 pages. How many pages in 9 notebooks?", "reference": "720"},
    {"prompt": "A flower pot needs 2 liters of water daily. How much water for 15 pots?", "reference": "30"},
    {"prompt": "A bus has 11 windows on each side. How many windows does it have on both sides?", "reference": "22"},
    {"prompt": "A restaurant serves 40 meals per hour. How many meals in 6 hours?", "reference": "240"},
    {"prompt": "Each pack contains 16 crayons. How many crayons in 5 packs?", "reference": "80"},
    {"prompt": "A wall is built with 17 bricks per row. How many bricks in 9 rows?", "reference": "153"},
    {"prompt": "A monkey eats 4 bananas a day. How many bananas in 12 days?", "reference": "48"},
    {"prompt": "There are 60 minutes in an hour. How many minutes in 5 hours?", "reference": "300"},
    {"prompt": "A cricket team has 11 players. How many players in 6 teams?", "reference": "66"},
    {"prompt": "A bag of rice weighs 5 kg. How much do 13 bags weigh?", "reference": "65"},
    {"prompt": "Each student plants 3 saplings. There are 25 students. How many saplings are planted?", "reference": "75"},
    {"prompt": "A worker packs 18 boxes per hour. How many boxes in 7 hours?", "reference": "126"},

    # ── LEVEL 3 : Division ────────────────────────────────────────────────────
    {"prompt": "72 mangoes are shared equally among 9 children. How many does each get?", "reference": "8"},
    {"prompt": "A baker makes 56 cookies and packs them into bags of 7. How many bags?", "reference": "8"},
    {"prompt": "48 students are divided into groups of 6. How many groups are there?", "reference": "8"},
    {"prompt": "A farmer has 84 eggs and packs them in boxes of 12. How many boxes?", "reference": "7"},
    {"prompt": "120 books are placed equally on 10 shelves. How many books per shelf?", "reference": "12"},
    {"prompt": "A rope of 90 meters is cut into pieces of 9 meters. How many pieces?", "reference": "10"},
    {"prompt": "63 sweets are shared equally among 7 friends. How many sweets each?", "reference": "9"},
    {"prompt": "A teacher divides 96 pencils equally among 8 students. How many pencils per student?", "reference": "12"},
    {"prompt": "150 rupees are shared equally among 5 children. How much does each get?", "reference": "30"},
    {"prompt": "A factory produces 200 bottles in 4 hours. How many bottles per hour?", "reference": "50"},
    {"prompt": "A bag of 45 marbles is divided equally into 5 bags. How many marbles per bag?", "reference": "9"},
    {"prompt": "144 chairs are arranged in rows of 12. How many rows are there?", "reference": "12"},
    {"prompt": "A box of 60 chocolates is shared equally among 4 children. How many each?", "reference": "15"},
    {"prompt": "A tank holds 180 liters. It fills 6 equal buckets. How many liters per bucket?", "reference": "30"},
    {"prompt": "110 students are seated equally in 11 rows. How many students per row?", "reference": "10"},
    {"prompt": "A school buys 240 notebooks for 8 classrooms equally. How many per class?", "reference": "30"},
    {"prompt": "A driver covers 300 km in 5 hours. What is the speed per hour?", "reference": "60"},
    {"prompt": "There are 108 players to be divided into teams of 9. How many teams?", "reference": "12"},
    {"prompt": "A garden has 77 flowers arranged in 7 equal rows. How many flowers per row?", "reference": "11"},
    {"prompt": "A packet of 64 stickers is divided equally among 8 children. How many each?", "reference": "8"},
    {"prompt": "A shop earns 350 rupees in 7 days equally. How much per day?", "reference": "50"},
    {"prompt": "A class of 36 students is split into groups of 4. How many groups?", "reference": "9"},
    {"prompt": "A roll of cloth 96 meters long is cut into 8 equal pieces. How long is each?", "reference": "12"},
    {"prompt": "350 ml of juice is poured equally into 5 glasses. How much per glass?", "reference": "70"},
    {"prompt": "132 apples are packed into boxes of 11. How many boxes are needed?", "reference": "12"},
    {"prompt": "A worker earns 540 rupees in 6 days. How much per day?", "reference": "90"},
    {"prompt": "280 grams of rice is divided equally into 7 portions. How many grams each?", "reference": "40"},
    {"prompt": "A bus carries 45 passengers per trip. How many trips to carry 225 passengers?", "reference": "5"},
    {"prompt": "A field produces 480 kg of wheat. If packed in 8 kg bags, how many bags?", "reference": "60"},
    {"prompt": "A ribbon of 72 cm is cut into 6 equal parts. How long is each part?", "reference": "12"},
    {"prompt": "256 tiles are laid equally in 8 rows. How many tiles per row?", "reference": "32"},
    {"prompt": "A trader distributes 130 kg of sugar into bags of 5 kg. How many bags?", "reference": "26"},
    {"prompt": "90 flowers are arranged equally in 9 vases. How many flowers per vase?", "reference": "10"},
    {"prompt": "A zoo has 84 animals in 7 equal enclosures. How many animals per enclosure?", "reference": "12"},
    {"prompt": "A machine fills 360 bottles in 6 hours. How many bottles per hour?", "reference": "60"},
    {"prompt": "A box of 117 oranges is divided equally among 9 families. How many each?", "reference": "13"},
    {"prompt": "168 pages are divided equally into 8 chapters. How many pages per chapter?", "reference": "21"},
    {"prompt": "A vendor sells 210 kg of vegetables in 7 days equally. How many kg per day?", "reference": "30"},
    {"prompt": "A pipe fills a tank of 400 liters in 8 minutes. How many liters per minute?", "reference": "50"},
    {"prompt": "81 students are seated equally in 9 rows. How many students per row?", "reference": "9"},
]

# ── CORE FUNCTIONS ─────────────────────────────────────────────────────────────

@tf.function
def tf_cosine_similarity(a, b):
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    dot  = tf.reduce_sum(a * b)
    norm = tf.norm(a) * tf.norm(b)
    return tf.where(norm < 1e-8, tf.zeros_like(dot), dot / norm)


@lru_cache(maxsize=256)
def get_tf_embedding(text: str):
    vec = embedding_model.encode(text, convert_to_numpy=True)
    return tf.constant(vec, dtype=tf.float32)


@lru_cache(maxsize=128)
def get_automated_reference(prompt):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(prompt, max_results=1, timeout=5))
            if results:
                snippet = f"{results[0]['title']}: {results[0]['body']}"
                return snippet, True
    except Exception:
        pass
    try:
        import socket
        socket.setdefaulttimeout(5)
        text = wikipedia.summary(prompt, sentences=2)
        return text, True
    except Exception:
        pass
    return "No verifiable reference found.", False


def check_hallucination_penalty(response: str, reference: str) -> float:
    if not reference or "No verifiable reference" in reference:
        return 0.0
    try:
        premise    = reference[:400]
        hypothesis = response[:200]
        result = nli_analyzer(f"{premise} [SEP] {hypothesis}", truncation=True)[0]
        label  = result["label"].upper()
        score  = result["score"]
        if label == "CONTRADICTION":
            return score * 0.9
        elif label == "NEUTRAL":
            return score * 0.2
        else:
            return 0.0
    except Exception as e:
        print(f"  NLI check failed: {e}")
        return 0.1


def run_model(prompt, temperature=TEMPERATURE):
    start = time.time()
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": NUM_PREDICT,
            "temperature": temperature,
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            output = response.json().get("response", "")
        else:
            output = f"[Ollama error: status {response.status_code}]"
    except Exception as e:
        output = f"[Ollama error: {e}]"
    return output, (time.time() - start)


def semantic_similarity(model_output, reference):
    try:
        emb1 = get_tf_embedding(model_output)
        emb2 = get_tf_embedding(reference)
        return float(tf_cosine_similarity(emb1, emb2))
    except Exception as e:
        print(f"  Similarity failed: {e}")
        return 0.5


def evidence_score(response):
    try:
        query    = response.split(".")[0][:100]
        ref, ok  = get_automated_reference(query)
        if not ok or "No verifiable reference" in ref:
            return 0.5
        emb1 = get_tf_embedding(response)
        emb2 = get_tf_embedding(ref)
        return float(tf_cosine_similarity(emb1, emb2))
    except Exception as e:
        print(f"  Evidence failed: {e}")
        return 0.5


def self_consistency(outputs):
    if len(outputs) < 2:
        return 1.0
    try:
        embeddings = [get_tf_embedding(o) for o in outputs]
        sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sims.append(float(tf_cosine_similarity(embeddings[i], embeddings[j])))
        return np.mean(sims) if sims else 1.0
    except Exception as e:
        print(f"  Consistency failed: {e}")
        return 0.7


def evaluate_prompt(item, index, total):
    prompt    = item["prompt"]
    reference = item.get("reference", None)

    print(f"\n{'='*70}")
    print(f"[{index}/{total}] Prompt: {prompt}")
    print(f"{'='*70}\n")

    # Get reference
    if reference is not None:
        ref, ref_verified = reference, True
    else:
        ref, ref_verified = get_automated_reference(prompt)

    # Run model NUM_RUNS times
    outputs, latencies = [], []
    print("  Responses:")
    for i in range(NUM_RUNS):
        resp, lat = run_model(prompt)
        outputs.append(resp)
        latencies.append(lat)
        preview = resp[:100] + ("..." if len(resp) > 100 else "")
        print(f"    {i+1}: {preview}")

    main_output = outputs[0]

    sim   = semantic_similarity(main_output, ref)
    evid  = evidence_score(main_output)
    cons  = self_consistency(outputs)
    hpen  = check_hallucination_penalty(main_output, ref) if ref_verified else 0.0

    base  = 0.35 * sim + 0.35 * evid + 0.30 * cons
    final = max(0.0, min(1.0, base - 0.3 * hpen))

    print(f"\n  Sim={sim:.3f}, Evid={evid:.3f}, Cons={cons:.3f}, Hall={hpen:.3f} → Final={final:.3f}")
    print(f"\n  REFERENCE : {ref}")
    print(f"  MODEL     : {main_output[:200]}")

    # Determine level
    if index <= 40:
        level = "L1_Addition_Subtraction"
    elif index <= 80:
        level = "L2_Multiplication"
    else:
        level = "L3_Division"

    return {
        "Index":        index,
        "Level":        level,
        "Prompt":       prompt,
        "Reference":    ref,
        "Model_Response": main_output,
        "Similarity":   round(sim,  4),
        "Evidence":     round(evid, 4),
        "Consistency":  round(cons, 4),
        "H_Penalty":    round(hpen, 4),
        "Final_Score":  round(final, 4),
        "Avg_Latency":  round(np.mean(latencies), 2),
    }


# ── MAIN LOOP ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Quick connectivity check
    print("\nChecking Ollama connection...")
    try:
        test_resp, _ = run_model("What is 2 plus 2?", temperature=0.0)
        if "Ollama error" in test_resp:
            print(f"ERROR: {test_resp}")
            print("Make sure Ollama is running: ollama serve")
            exit(1)
        print(f"Connection OK. Test response: {test_resp.strip()}\n")
    except Exception as e:
        print(f"Cannot reach Ollama: {e}")
        exit(1)

    records = []
    total   = len(dataset)

    for i, item in enumerate(dataset, start=1):
        record = evaluate_prompt(item, i, total)
        records.append(record)

    # ── RESULTS ────────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)

    print("\n\n" + "=" * 70)
    print("FINAL RELIABILITY RESULTS — RYKYU Math Tutor")
    print("=" * 70)

    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 200)

    print("\n── Top 20 by Final Score ──")
    print(df[["Level", "Prompt", "Reference", "Similarity", "Consistency",
              "H_Penalty", "Final_Score"]]
          .sort_values("Final_Score", ascending=False)
          .head(20)
          .to_string(index=False))

    print("\n── Per-Level Summary ──")
    summary = df.groupby("Level")[["Similarity", "Evidence", "Consistency",
                                    "H_Penalty", "Final_Score", "Avg_Latency"]].mean().round(4)
    print(summary.to_string())

    print("\n── Overall Stats ──")
    print(f"  Total prompts evaluated : {len(df)}")
    print(f"  Mean Final Score        : {df['Final_Score'].mean():.4f}")
    print(f"  Std Final Score         : {df['Final_Score'].std():.4f}")
    print(f"  Mean Avg Latency (s)    : {df['Avg_Latency'].mean():.2f}")

    # ── SAVE CSV ───────────────────────────────────────────────────────────────
    out_path = "rykyu_reliability_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print("=" * 70)

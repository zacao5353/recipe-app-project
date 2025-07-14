from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import numpy as np
from PIL import Image # run_ml_model_on_imageでまだ使用
import io # run_ml_model_on_imageでまだ使用

# --- 英語から日本語への食材名マップ ---
ENG_TO_JPN_MAP = {
    "cabbage": "キャベツ",
    "carrot": "にんじん",
    "eggplant": "なす",
    "onion": "玉ねぎ",
    "potato": "じゃがいも",
    "tomato": "トマト",
    "asparagus": "アスパラガス",
    "enoki": "えのき",
    "okra": "オクラ",
    "pumpkin": "かぼちゃ",
    "cucumber": "きゅうり",
    "shiitake": "しいたけ",
    "egg": "卵",
    "shimeji": "しめじ",
    "daikon": "大根",
    "sake": "酒", # 鮭(salmon)と混同しないよう注意
    "saba": "鯖",
    "chicken": "鶏肉",
    "beef": "牛肉",
    "pork": "豚肉",
    "moyashi": "もやし",
    "tofu": "豆腐",
    "sausage": "ソーセージ",
    "spinach": "ほうれん草",
    "bacon": "ベーコン",
    # 以下はレシピ内で使用される一般的な調味料や材料で、認識対象外だが変換があると便利
    "soy_sauce": "醤油",
    "mirin": "みりん",
    "miso": "味噌",
    "ginger": "生姜",
    "garlic": "にんにく",
    "olive_oil": "オリーブオイル",
    "salt": "塩",
    "pepper": "こしょう",
    "dashi": "だし",
    "sugar": "砂糖",
    "chili_bean_paste": "豆板醤",
    "curry_powder": "カレー粉",
    "water": "水",
    "vinegar": "酢",
    "milk": "牛乳",
    "butter": "バター",
    "rice": "米",
}

# --- ヘルパー関数 ---
def clean_text(text):
    text = re.sub(r'[・、,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def has_all_main_ingredients(main_ingrs_str, recognized_ingrs_set):
    """主要材料が認識された食材のセットにすべて含まれているかチェックするヘルパー関数"""
    if pd.isna(main_ingrs_str) or main_ingrs_str.strip() == "":
        return True
    cleaned_main_ingrs = clean_text(main_ingrs_str).split()
    return all(ingr in recognized_ingrs_set for ingr in cleaned_main_ingrs)

def recommend_recipes_based_on_main_and_required(recognized_ingredients, recipe_df, top_n=5):
    """
    認識された食材と主要材料、必要材料の共通点を考慮して関連度の高いレシピを提案する関数
    """
    required_cols_for_logic = ['recipe_name', 'required_ingredients', 'main_ingredients']
    for col in required_cols_for_logic:
        if col not in recipe_df.columns:
            raise ValueError(f"recipe_dfに'{col}'カラムが見つかりません。")

    # recognized_ingredients はすでに日本語化されている想定
    recognized_ingredients_set = set(clean_text(" ".join(recognized_ingredients)).split())

    filtered_df = recipe_df[
        recipe_df['main_ingredients'].apply(
            lambda x: has_all_main_ingredients(x, recognized_ingredients_set)
        )
    ].copy()

    if filtered_df.empty:
        return pd.DataFrame(columns=list(recipe_df.columns) + ['similarity'])

    filtered_df['cleaned_required_ingredients'] = filtered_df['required_ingredients'].apply(clean_text)
    cleaned_recognized_ingredients_str = clean_text(" ".join(recognized_ingredients))

    all_ingredients_text_for_tfidf = filtered_df['cleaned_required_ingredients'].tolist() + [cleaned_recognized_ingredients_str]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_ingredients_text_for_tfidf)

    recipe_tfidf_matrix = tfidf_matrix[:-1]
    recognized_ingredients_tfidf = tfidf_matrix[-1]

    similarities = cosine_similarity(recognized_ingredients_tfidf, recipe_tfidf_matrix).flatten()

    filtered_df['similarity'] = similarities

    recommended_recipes_df = filtered_df.sort_values(by='similarity', ascending=False).head(top_n)

    return recommended_recipes_df[[col for col in recipe_df.columns if col != 'cleaned_required_ingredients'] + ['similarity']]


app = FastAPI()

# CORS設定: ウェブブラウザからのアクセスを許可するために重要
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500",
    "null",
    "https://recipe-frontend-axty.onrender.com" # あなたのウェブアプリの公開URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CSVファイルをロード
csv_file_path = os.path.join(os.path.dirname(__file__), '..', 'recipes.csv')
try:
    recipe_df_global = pd.read_csv(csv_file_path, encoding='utf-8-sig') # 'utf-8-sig' で読み込み
    print("CSVファイルを正常に読み込みました。")
except Exception as e:
    print(f"CSVファイル読み込みエラー: {e}")
    recipe_df_global = pd.DataFrame()

# --- 画像認識APIエンドポイント ---
# ここにあなたの機械学習モデルの推論ロジックを記述します。
# 実際にはモデルのロードと推論ロジックを記述してください。
def run_ml_model_on_image(image_bytes: bytes) -> list[str]:
    """
    ここにあなたの機械学習モデルをロードし、画像から食材を推論するロジックを記述します。
    モデルの出力は英語名で、この関数内で日本語に変換して返します。
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # 画像の前処理
        # processed_image = preprocess(image)

        # モデルの推論（例として仮の英語結果を返す）
        if image.width > 800 and image.height > 600:
            english_recognized_ingredients = ["chicken", "onion", "carrot"]
        elif image.width < 200 and image.height < 200:
            english_recognized_ingredients = ["egg", "tomato"]
        else:
            english_recognized_ingredients = ["potato", "cabbage"]

        # 英語名を日本語に変換
        japanese_recognized_ingredients = [ENG_TO_JPN_MAP.get(eng_name.lower(), eng_name) for eng_name in english_recognized_ingredients]
        return japanese_recognized_ingredients

    except Exception as e:
        print(f"画像認識モデルの実行中にエラーが発生しました: {e}")
        return []

@app.post("/recognize_image")
async def recognize_image_api(file: UploadFile = File(...)):
    """
    画像をアップロードし、食材を認識してリストで返すAPIエンドポイント
    （認識結果は日本語に変換されて返されます）
    """
    try:
        image_bytes = await file.read()
        recognized_ingredients_jpn = run_ml_model_on_image(image_bytes)

        return {"recognized_ingredients": recognized_ingredients_jpn}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"画像認識中にエラーが発生しました: {e}")

# --- レシピ提案APIエンドポイント ---
@app.post("/recommend_recipes")
async def recommend_recipes_api(ingredients: list[str]):
    """
    認識された食材リスト（日本語）を受け取り、レシピを提案するAPIエンドポイント
    """
    if recipe_df_global.empty:
        raise HTTPException(status_code=500, detail="Recipe data not loaded.")
    
    # recipes.csv の食材名も日本語化されているので、そのまま渡せる
    recommendations = recommend_recipes_based_on_main_and_required(ingredients, recipe_df_global.copy())
    
    # NaN値をJSON対応の値に変換
    for col in ['main_ingredients', 'required_ingredients', 'instructions', 'dietary_restrictions']:
        if col in recommendations.columns:
            recommendations[col] = recommendations[col].replace({np.nan: ''})
    
    if 'similarity' in recommendations.columns:
        recommendations['similarity'] = recommendations['similarity'].replace({np.nan: None})

    return recommendations.to_dict(orient="records")

@app.get("/")
async def read_root():
    return {"message": "レシピ提案APIが稼働しています。/recommend_recipes にPOSTリクエストを送ってください。"}
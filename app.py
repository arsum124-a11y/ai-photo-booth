import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io
import requests

# 設定網頁標題與圖示
st.set_page_config(page_title="AI 照相館", page_icon="📸")

st.title("📸 長者手機 AI 體驗：卡通化合成")
st.write("請使用手機鏡頭拍照，我們會自動為您去背、轉換成卡通風格，並與您選擇的背景合成！")

# ==========================================
# 步驟 1 & 2：讀取相機權限與拍照
# ==========================================
st.subheader("步驟 1：請拍攝您的照片")

# 這個指令會自動在手機/電腦上要求開啟相機權限
camera_photo = st.camera_input("點擊下方按鈕開啟相機拍照 ⬇️")

# 溫馨小設計：考量有些長者不習慣用網頁拍照，我們保留「從相簿上傳」作為備用方案
upload_photo = st.file_uploader("或者從手機相簿選擇照片：", type=["jpg", "jpeg", "png"])

# 決定使用哪一張照片 (優先使用相機拍的，如果沒有才用上傳的)
user_file = camera_photo if camera_photo else upload_photo

# ==========================================
# 步驟 3：用戶選擇預設的 4 個背景
# ==========================================
st.subheader("步驟 2：請選擇一個背景")

# 我為你準備了 4 個線上免費圖庫的高畫質風景照
bg_options = {
    "🌸 浪漫櫻花": "https://images.unsplash.com/photo-1522383225653-ed111181a951?w=800&q=80",
    "🌌 璀璨星空": "https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?w=800&q=80",
    "🌲 靜謐森林": "https://images.unsplash.com/photo-1448375240586-882707db888b?w=800&q=80",
    "🏖️ 陽光海灘": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800&q=80"
}

# 讓用戶用點選的方式選擇背景
bg_choice = st.radio("點選您喜歡的背景風格：", list(bg_options.keys()))

# ==========================================
# 步驟 4, 5, 6：AI 處理與合成
# ==========================================
# 只有當用戶已經拍照或上傳照片後，才顯示開始按鈕
if user_file is not None:
    if st.button("✨ 開始製作我的專屬圖片 ✨"):
        with st.spinner("AI 魔法正在施展中（去背與卡通化），大約需要 10~20 秒，請稍候..."):
            try:
                # 讀取用戶照片
                user_img = Image.open(user_file)

                # 從網路上偷偷下載用戶選擇的背景圖
                response = requests.get(bg_options[bg_choice])
                bg_img = Image.open(io.BytesIO(response.content)).convert("RGBA")

                # --- 步驟 4：去背 ---
                no_bg_img = remove(user_img)

                # --- 步驟 5：卡通化 ---
                img_array = np.array(no_bg_img)
                r, g, b, a = cv2.split(img_array)
                rgb_img = cv2.merge([r, g, b])
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                
                # 卡通濾鏡參數
                cartoon_bgr = cv2.stylization(bgr_img, sigma_s=150, sigma_r=0.25)
                cartoon_rgb = cv2.cvtColor(cartoon_bgr, cv2.COLOR_BGR2RGB)
                
                # 把透明通道加回去 (這樣去背的地方才會是透明的)
                cr, cg, cb = cv2.split(cartoon_rgb)
                final_rgba_array = cv2.merge([cr, cg, cb, a])
                cartoon_img = Image.fromarray(final_rgba_array, 'RGBA')

                # --- 步驟 6：將去背卡通化後的人像，貼在背景中 ---
                target_height = int(bg_img.height * 0.8) # 人像佔背景高度的 80%
                ratio = target_height / cartoon_img.height
                target_width = int(cartoon_img.width * ratio)
                
                fg_resized = cartoon_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # 計算貼上的位置 (置中靠下)
                paste_x = (bg_img.width - target_width) // 2
                paste_y = bg_img.height - target_height
                
                bg_img.paste(fg_resized, (paste_x, paste_y), fg_resized)
                final_result = bg_img.convert("RGB")

                # --- 顯示最終結果 ---
                st.success("🎉 完成啦！請往下看您的成品：")
                st.image(final_result, caption="您的專屬卡通合成照", use_column_width=True)

                # 提供按鈕讓長者下載圖片到手機裡
                buf = io.BytesIO()
                final_result.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="📥 點擊這裡下載圖片到手機", 
                    data=byte_im, 
                    file_name="my_cartoon_photo.jpg", 
                    mime="image/jpeg"
                )

            except Exception as e:
                st.error(f"抱歉，處理過程中發生錯誤：{e}")

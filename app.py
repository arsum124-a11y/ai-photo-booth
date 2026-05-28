import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io

# 網頁標題
st.title("🎨 長者手機 AI 體驗：卡通化照片合成")
st.write("請上傳您的照片，我們會幫您去背、卡通化，並合成到背景上！")

# 步驟 1：上傳背景圖
st.subheader("1. 請先上傳一張背景圖")
bg_file = st.file_uploader("選擇背景圖片...", type=["jpg", "jpeg", "png"], key="bg")

# 步驟 2：上傳人像圖
st.subheader("2. 請上傳您的照片 (包含人像)")
user_file = st.file_uploader("選擇您拍攝的照片...", type=["jpg", "jpeg", "png"], key="user")

# 當兩張圖片都上傳後，顯示一個按鈕開始處理
if bg_file and user_file:
    if st.button("✨ 開始施展魔法 ✨"):
        with st.spinner("AI 正在努力去背與繪圖中，請稍候約十秒鐘..."):
            try:
                # 讀取圖片
                bg_img = Image.open(bg_file).convert("RGBA")
                user_img = Image.open(user_file)

                # --- 1. 去背 ---
                no_bg_img = remove(user_img)

                # --- 2. 卡通化 ---
                img_array = np.array(no_bg_img)
                r, g, b, a = cv2.split(img_array)
                rgb_img = cv2.merge([r, g, b])
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                
                # 卡通濾鏡
                cartoon_bgr = cv2.stylization(bgr_img, sigma_s=150, sigma_r=0.25)
                cartoon_rgb = cv2.cvtColor(cartoon_bgr, cv2.COLOR_BGR2RGB)
                
                # 把透明通道加回去
                cr, cg, cb = cv2.split(cartoon_rgb)
                final_rgba_array = cv2.merge([cr, cg, cb, a])
                cartoon_img = Image.fromarray(final_rgba_array, 'RGBA')

                # --- 3. 合成 ---
                target_height = int(bg_img.height * 0.8)
                ratio = target_height / cartoon_img.height
                target_width = int(cartoon_img.width * ratio)
                
                fg_resized = cartoon_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                paste_x = (bg_img.width - target_width) // 2
                paste_y = bg_img.height - target_height
                
                bg_img.paste(fg_resized, (paste_x, paste_y), fg_resized)
                final_result = bg_img.convert("RGB")

                # --- 4. 在網頁上顯示結果 ---
                st.success("完成啦！看看結果吧！")
                st.image(final_result, caption="您的專屬合成照", use_column_width=True)

                # 提供下載按鈕
                buf = io.BytesIO()
                final_result.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="📥 點我下載圖片",
                    data=byte_im,
                    file_name="my_cartoon_photo.jpg",
                    mime="image/jpeg"
                )
            except Exception as e:
                st.error(f"發生錯誤：{e}")

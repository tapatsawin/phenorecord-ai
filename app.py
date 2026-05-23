import os

import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor

# 👑 ตั้งค่าหน้าเว็บ Dashboard ของเรา
st.set_page_config(page_title="PhenoRecord AI", layout="wide")
st.title("🧬 PhenoRecord AI - NCDs Precision Dashboard")
st.write("ระบบบรูณาการข้อมูลพันธุกรรมและเวชระเบียนเพื่อคัดกรองโรคเบาหวานรายบุคคล")

# ---------------------------------------------------------
# Step 1: โหลดข้อมูลจากไฟล์จริงของงานแข่ง
# ---------------------------------------------------------
# 💡 วันแข่งจริง: ให้เปลี่ยนชื่อไฟล์ในวงเล็บเป็นชื่อไฟล์ที่งานแข่งแจกมาได้เลย
# ---------------------------------------------------------
# Step 1: โหลดข้อมูลจากไฟล์จริงของงานแข่ง
# ---------------------------------------------------------
@st.cache_data 
def load_competition_data():
    try:
        df_dna = pd.read_csv('data/dna_data.csv') 
        df_emr = pd.read_csv('data/emr_data.csv')
        df_merged = pd.merge(df_dna, df_emr, on='Patient_ID')
        return df_merged
    except FileNotFoundError:
        import numpy as np
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            'Patient_ID': [f"P{str(i).zfill(3)}" for i in range(1, n + 1)],
            'Patient_Name': [f"คนไข้ รายที่ {i}" for i in range(1, n + 1)],
            'SNP_rs123': np.random.choice(['AA', 'AG', 'GG'], n), # 🟢 เพิ่มยีนส์ที่หายไป
            'SNP_rs456': np.random.choice(['CC', 'CT', 'TT'], n),
            'SNP_rs789': np.random.choice(['GG', 'GA', 'AA'], n), # 🟢 เพิ่มยีนส์ที่หายไป
            'Age': np.random.randint(30, 80, n),
            'BMI': np.random.uniform(18.5, 35.0, n).round(1),
            'Blood_Sugar': np.random.randint(80, 200, n),
            'Diabetes_Risk': np.random.binomial(1, 0.4, n)
        })

df_all = load_competition_data()

import os
import glob
from autogluon.tabular import TabularPredictor

# ---------------------------------------------------------
# Step 2: โหลดโมเดล AI พร้อมระบบตรวจจับข้อผิดพลาด (Debug Mode)
# ---------------------------------------------------------
@st.cache_resource
def load_ai_predictor():
    try:
        # 1. ตรวจสอบว่าปัจจุบัน Streamlit ทำงานอยู่ที่โฟลเดอร์ไหน
        current_dir = os.getcwd()
        st.info(f"📁 ตอนนี้โปรแกรมกำลังทำงานอยู่ที่โฟลเดอร์: `{current_dir}`")
        
        # 2. ลองค้นหาโฟลเดอร์ใน AutogluonModels
        list_of_dirs = glob.glob('AutogluonModels/ag-*')
        st.write("🔍 โฟลเดอร์ที่ระบบค้นพบตอนนี้คือ:", list_of_dirs)
        
        if not list_of_dirs:
            st.error("❌ ระบบไม่เจอโฟลเดอร์ที่ขึ้นต้นด้วย 'ag-' ในโฟลเดอร์ AutogluonModels เลยครับ")
            st.write("💡 วิธีแก้: ให้ลองเปิด Terminal แล้วรัน `py -3.11 work.py` อีกครั้ง เพื่อให้มันสร้างโมเดลใหม่ในโฟลเดอร์ปัจจุบันก่อนครับ")
            return None
            
        # 3. เลือกตัวล่าสุด
        latest_model_path = max(list_of_dirs, key=os.path.getmtime)
        st.success(f"🤖 กำลังโหลดโมเดลจาก: `{latest_model_path}`")
        
        return TabularPredictor.load(latest_model_path)
    except Exception as e:
        st.error(f"💥 เกิดข้อผิดพลาดขณะโหลดโมเดล: {e}")
        return None

predictor = load_ai_predictor()
# ---------------------------------------------------------
# 🔍 Step 3: ส่วนหน้าต่างการค้นหาผู้ป่วย (Search Interface)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔍 ค้นหาข้อมูลผู้ป่วย")

# สร้างกล่องพิมพ์ข้อความ หรือเมนูให้เลือกสำหรับเสิร์ชรหัสคนไข้
search_id = st.text_input("พิมพ์รหัสผู้ป่วย (เช่น P001, P002) แล้วกด Enter:", "").strip()

if search_id:
    # 🔍 ใช้ Pandas เสิร์ชกรองข้อมูลเฉพาะแถวที่มี Patient_ID ตรงกับที่พิมพ์ค้นหา
    patient_row = df_all[df_all['Patient_ID'] == search_id]
    
    if not patient_row.empty:
        # ดึงข้อมูลออกมาเป็นตัวแปรเดียวเพื่อให้เรียกใช้ง่ายๆ
        p_data = patient_row.iloc[0]
        
        st.success(f"พบข้อมูลผู้ป่วยรหัส: {search_id}")
        
        # จัดเลย์เอาต์แสดงประวัติคนไข้เป็นคอลัมน์สวยๆ
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("ชื่อผู้ป่วย", p_data.get('Patient_Name', 'ไม่ระบุชื่อ'))
        with col2: st.metric("รหัสยีน (SNP_rs456)", p_data['SNP_rs456'])
        with col3: st.metric("ค่าน้ำตาลในเลือด", f"{p_data['Blood_Sugar']} mg/dL")
        with col4: st.metric("ดัชนีมวลกาย (BMI)", p_data['BMI'])
        
        # 🤖 ส่งข้อมูลเฉพาะของคนไข้คนนี้ให้ AI ประมวลผลทำนายความเสี่ยง
        if predictor is not None:
            # เตรียมข้อมูลฟีเจอร์สำหรับส่งให้ AI (ตัด ID และคอลัมน์ที่ไม่ใช้ออก)
            input_df = patient_row.drop(columns=['Patient_ID', 'Diabetes_Risk'], errors='ignore')
            if 'Patient_Name' in input_df.columns:
                input_df = input_df.drop(columns=['Patient_Name'])
                
            # สั่งให้ AI คำนวณเปอร์เซ็นต์ความเสี่ยง
            prob_risk = predictor.predict_proba(input_df).iloc[0][1] * 100
            
            st.markdown("### 🤖 ผลการประเมินจากระบบ AI")
            if prob_risk > 50:
                st.error(f"⚠️ ความเสี่ยงต่อโรคเบาหวานสูงมาก: {prob_risk:.2f}%")
            else:
                st.success(f"🟢 ความเสี่ยงต่อโรคเบาหวานต่ำ: {prob_risk:.2f}%")
                
            # 📋 ดึง Logic สร้างแผน Action Plan เชิงป้องกันเฉพาะคนนั้นขึ้นมาโชว์
            st.markdown("### 📋 คู่มือคำแนะนำการรักษาเฉพาะบุคคล (Personalized Action Plan)")
            
            if prob_risk > 50:
                if p_data['SNP_rs456'] == 'TT':
                    st.warning("ท่านมีความเสี่ยงแฝงสูงจากรหัสพันธุกรรมภาวะดื้ออินซูลิน (TT)")
                if p_data['Blood_Sugar'] >= 126:
                    st.info(f"- 🔴 ค่าน้ำตาลในปัจจุบัน ({p_data['Blood_Sugar']} mg/dL) เกินเกณฑ์มาตรฐาน! แนะนำพบแพทย์ด่วนเพื่อปรับพฤติกรรมโภชนาการ ป้องกันสภาวะแทรกซ้อนสู่โรคไต")
                if p_data['BMI'] > 25.0:
                    st.info(f"- 🏃 ดัชนีมวลกายเกินเกณฑ์ (BMI: {p_data['BMI']}) แนะนำควบคุมอาหารกลุ่มแป้งและออกกำลังกายเพื่อกดระดับการแสดงออกของยีนเสี่ยง")
            else:
                if p_data['SNP_rs456'] == 'TT':
                    st.info("✨ ร่างกายมียีนเสี่ยงเบาหวาน (TT) แต่เนื่องจากพฤติกรรมปัจจุบัน (EMR) คุมค่าน้ำตาลและน้ำหนักได้ดีเยี่ยม ยีนจึงไม่แสดงออก ขอให้รักษามาตรฐานนี้ไว้ครับ")
                else:
                    st.info("✨ ร่างกายสมบูรณ์ดี พันธุกรรมปกติและพฤติกรรมอยู่ในเกณฑ์ปลอดภัย")
        else:
            st.warning("⚠️ ยังไม่ได้โหลดโมเดล AI กรุณาตรวจสอบเส้นทางโฟลเดอร์มเดลในโค้ด")
            
    else:
        st.error(f"❌ ไม่พบข้อมูลผู้ป่วยรหัส '{search_id}' ในฐานข้อมูล กรุณาลองรหัสอื่น (ลองพิมพ์ P001 - P050)")
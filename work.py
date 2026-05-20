import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor

# ---------------------------------------------------------
# Step 1: จำลองข้อมูล DNA และ EMR
# ---------------------------------------------------------
print("📊 1. กำลังโหลดและจำลองข้อมูล...")
np.random.seed(42)
n_samples = 200

df_dna = pd.DataFrame({
    'Patient_ID': [f"P{str(i).zfill(3)}" for i in range(1, n_samples + 1)],
    'SNP_rs123': np.random.choice(['AA', 'AG', 'GG'], n_samples),
    'SNP_rs456': np.random.choice(['CC', 'CT', 'TT'], n_samples, p=[0.5, 0.3, 0.2]),
    'SNP_rs789': np.random.choice(['GG', 'GA', 'AA'], n_samples)
})

df_emr = pd.DataFrame({
    'Patient_ID': [f"P{str(i).zfill(3)}" for i in range(1, n_samples + 1)],
    'Age': np.random.randint(20, 80, n_samples),
    'BMI': np.random.uniform(18.5, 35.0, n_samples).round(1),
    'Blood_Sugar': np.random.randint(80, 200, n_samples)
})

# ---------------------------------------------------------
# Step 2: ผสานข้อมูล (Merge)
# ---------------------------------------------------------
df_merged = pd.merge(df_dna, df_emr, on='Patient_ID')

risk_score = np.where(df_merged['SNP_rs456'] == 'TT', 0.4, 0)
risk_score += np.where(df_merged['Blood_Sugar'] > 125, 0.4, 0)
risk_score += np.where(df_merged['BMI'] > 25.0, 0.2, 0)
df_merged['Disease_Risk'] = np.random.binomial(1, np.clip(risk_score, 0, 1))

df_clean = df_merged.drop(columns=['Patient_ID'])
train_data, test_data = train_test_split(df_clean, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# Step 3: เทรนโมเดล AI (AutoGluon)
# ---------------------------------------------------------
print("🚀 2. เริ่มเทรนโมเดล AI...")
predictor = TabularPredictor(label='Disease_Risk', eval_metric='accuracy', verbosity=0).fit(
    train_data, presets='medium_quality', time_limit=30
)

# ---------------------------------------------------------
# 🌟 ไฮไลท์ใหม่: ฟังก์ชันระบบแนะนำเชิงป้องกัน (Action Plan Logic)
# ---------------------------------------------------------
def generate_action_plan(row, risk_prob):
    advice = []
    
    # กำหนดเกณฑ์สุขภาพ
    is_genetic_risk = (row['SNP_rs456'] == 'TT')
    is_high_sugar = (row['Blood_Sugar'] > 125)
    is_high_bmi = (row['BMI'] > 25.0)

    # กรณี AI ประเมินว่า "เสี่ยงโรค" (ความน่าจะเป็น > 50%)
    if risk_prob > 0.5:
        if is_genetic_risk:
            advice.append("⚠️ ตรวจพบยีนเสี่ยง (TT): ท่านมีความเสี่ยงทางพันธุกรรมเป็นทุนเดิม")
            if is_high_sugar or is_high_bmi:
                advice.append("💡 คำแนะนำ (Action Plan): แม้จะมียีนเสี่ยง แต่คุณสามารถลดความเสี่ยงโดยรวมได้โดยการปรับพฤติกรรม ดังนี้:")
                if is_high_sugar:
                    advice.append(f"   - ลดน้ำตาลในเลือดด่วน! (ปัจจุบัน {row['Blood_Sugar']} mg/dL → เป้าหมาย < 125 mg/dL)")
                if is_high_bmi:
                    advice.append(f"   - ลดน้ำหนัก (ปัจจุบัน BMI {row['BMI']} → เป้าหมาย < 25.0)")
            else:
                advice.append("💡 คำแนะนำ (Action Plan): ปัจจุบันสุขภาพ EMR ดีมาก ขอให้รักษาระดับน้ำตาลให้อยู่ในเกณฑ์นี้ต่อไปเพื่อกดทับการแสดงออกของยีน")
        
        else: # ไม่มียีนเสี่ยง แต่ป่วยเพราะพฤติกรรม
            advice.append("⚠️ พันธุกรรมปกติ แต่มีความเสี่ยงสูงจากพฤติกรรม (ข้อมูล EMR)!")
            advice.append("💡 คำแนะนำ (Action Plan):")
            if is_high_sugar:
                advice.append(f"   - ควรคุมอาหารและลดค่าน้ำตาล (ปัจจุบัน {row['Blood_Sugar']} mg/dL → เป้าหมาย < 125 mg/dL)")
            if is_high_bmi:
                advice.append(f"   - ควรออกกำลังกายลด BMI (ปัจจุบัน {row['BMI']} → เป้าหมาย < 25.0)")
                
    # กรณี AI ประเมินว่า "ไม่เสี่ยง"
    else: 
        if is_genetic_risk:
             advice.append("🟢 ตรวจพบยีนเสี่ยง (TT) แต่ปัจจุบันพฤติกรรมดูแลตัวเองดีเยี่ยม (น้ำตาลและ BMI ปกติ) ความเสี่ยงรวมจึงต่ำมาก ขอให้รักษามาตรฐานนี้ไว้!")
        else:
             advice.append("🟢 สุขภาพโดยรวมและพันธุกรรมอยู่ในเกณฑ์ดีมาก ไม่มีข้อควรระวังพิเศษ")

    return "\n".join(advice)

# ---------------------------------------------------------
# Step 4: ทดสอบ Test Cases + สร้าง Action Plan
# ---------------------------------------------------------
print("\n🧪 3. ทดสอบการออก Action Plan เฉพาะบุคคล:")
test_cases = pd.DataFrame({
    'Patient': ['นาย A', 'นาย B', 'นาย C'],
    'SNP_rs123': ['AA', 'GG', 'AG'],
    'SNP_rs456': ['TT', 'CC', 'TT'], # A กับ C มียีนเสี่ยง
    'SNP_rs789': ['GA', 'GG', 'AA'],
    'Age': [55, 30, 45],
    'BMI': [28.5, 21.0, 22.0],        # A อ้วน
    'Blood_Sugar': [145, 95, 98]      # A น้ำตาลสูง
})

# ให้ AI ทายผล
features_only = test_cases.drop(columns=['Patient'])
predictions = predictor.predict(features_only)
probabilities = predictor.predict_proba(features_only)

# นำผลลัพธ์มาเข้าฟังก์ชัน Action Plan
for i in range(len(test_cases)):
    patient_name = test_cases['Patient'].iloc[i]
    prob_risk = probabilities.iloc[i][1]
    pred_label = "🔴 เสี่ยงเป็นโรค" if predictions.iloc[i] == 1 else "🟢 ไม่เสี่ยง"
    
    print("-" * 60)
    print(f"👤 ผู้ป่วย: {patient_name} | 🤖 ประเมิน: {pred_label} (โอกาส {prob_risk*100:.1f}%)")
    print(f"📋 ข้อมูล: ยีน {test_cases['SNP_rs456'].iloc[i]} | BMI {test_cases['BMI'].iloc[i]} | น้ำตาล {test_cases['Blood_Sugar'].iloc[i]}")
    
    # เรียกใช้ระบบแนะนำ
    action_plan = generate_action_plan(test_cases.iloc[i], prob_risk)
    print(f"\n{action_plan}")

print("-" * 60)

# ---------------------------------------------------------
# Step 5: สร้างกราฟเปรียบเทียบประสิทธิภาพของ AI แต่ละโมเดล (Model Comparison)
# ---------------------------------------------------------
print("\n📊 5. กำลังสร้างกราฟเปรียบเทียบประสิทธิภาพ AI ทุกโมเดล...")

# 🟢 แก้ไขบรรทัดนี้: ลบ .drop() ออก โยน test_data เข้าไปได้เลย
leaderboard = predictor.leaderboard(test_data, silent=True)

# แปลงคะแนนให้อยู่ในรูปเปอร์เซ็นต์ (0-100%) เพื่อให้กราฟดูง่าย
leaderboard['Accuracy (%)'] = leaderboard['score_test'] * 100

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# วาดกราฟแท่งแนวนอน (เรียงลำดับความเก่งจากบนลงล่างอัตโนมัติ)
ax = sns.barplot(
    x='Accuracy (%)', 
    y='model', 
    data=leaderboard, 
    palette='Blues_r', 
    hue='model',
    legend=False
)

# เพิ่มตัวเลข % แปะไว้ที่ปลายแท่งกราฟแต่ละแท่ง
for p in ax.patches:
    width = p.get_width()
    plt.text(width + 1.5, p.get_y() + p.get_height()/2. + 0.1, 
             f'{width:.1f}%', 
             ha="left", va="center", fontweight='bold', color='#333333')

# ตกแต่งรายละเอียดให้พร้อมแปะลงสไลด์
plt.title('🤖 AI Algorithms Performance Comparison', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Accuracy Score (%) on Test Data', fontsize=12, fontweight='bold')
plt.ylabel('Machine Learning Models', fontsize=12, fontweight='bold')
plt.xlim(0, 115) 

sns.despine(left=True, bottom=True)
plt.tight_layout()

# บันทึกรูปภาพ
output_leaderboard = 'model_comparison.png'
plt.savefig(output_leaderboard, dpi=300)
print(f"✅ บันทึกกราฟเปรียบเทียบโมเดลเรียบร้อยแล้วที่ไฟล์: '{output_leaderboard}'")

plt.show()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') # ปิดแจ้งเตือนจุกจิก

# ---------------------------------------------------------
# Step 1: จำลองข้อมูล DNA และ EMR (โฟกัส NCDs - เบาหวานชนิดที่ 2)
# ---------------------------------------------------------
print("📊 1. กำลังโหลดข้อมูล DNA และ EMR (NCDs Focus)...")
np.random.seed(42)
n_samples = 200

# 🧬 ข้อมูลพันธุกรรม (สมมติ SNP_rs456 คือยีนที่เสี่ยงต่อภาวะดื้ออินซูลิน)
df_dna = pd.DataFrame({
    'Patient_ID': [f"P{str(i).zfill(3)}" for i in range(1, n_samples + 1)],
    'SNP_rs123': np.random.choice(['AA', 'AG', 'GG'], n_samples),
    'SNP_rs456': np.random.choice(['CC', 'CT', 'TT'], n_samples, p=[0.5, 0.3, 0.2]), # TT = เสี่ยงเบาหวาน
    'SNP_rs789': np.random.choice(['GG', 'GA', 'AA'], n_samples)
})

# 🏥 ข้อมูลเวชระเบียน (ปัจจัยพฤติกรรมที่กระตุ้นเบาหวาน)
df_emr = pd.DataFrame({
    'Patient_ID': [f"P{str(i).zfill(3)}" for i in range(1, n_samples + 1)],
    'Age': np.random.randint(30, 80, n_samples), # เจาะกลุ่มวัยทำงานถึงผู้สูงอายุ
    'BMI': np.random.uniform(18.5, 35.0, n_samples).round(1),
    'Blood_Sugar': np.random.randint(80, 200, n_samples) # ค่าน้ำตาล (FBS)
})

# ---------------------------------------------------------
# Step 2: ผสานข้อมูล (Merge) และกำหนดตรรกะโรคเบาหวาน
# ---------------------------------------------------------
df_merged = pd.merge(df_dna, df_emr, on='Patient_ID')

# Logic เบาหวาน: ยีนเสี่ยง + น้ำตาลสูง(>126 คือเกณฑ์เบาหวาน) + อ้วน(BMI>25) + อายุมาก
risk_score = np.where(df_merged['SNP_rs456'] == 'TT', 0.3, 0)
risk_score += np.where(df_merged['Blood_Sugar'] >= 126, 0.4, 0)
risk_score += np.where(df_merged['BMI'] > 25.0, 0.2, 0)
risk_score += np.where(df_merged['Age'] > 50, 0.1, 0)

# สร้างคอลัมน์เป้าหมาย 'Diabetes_Risk'
df_merged['Diabetes_Risk'] = np.random.binomial(1, np.clip(risk_score, 0, 1))

# จัดการ Privacy
df_clean = df_merged.drop(columns=['Patient_ID'])
train_data, test_data = train_test_split(df_clean, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# Step 3: เทรนโมเดล AI (AutoGluon)
# ---------------------------------------------------------
print("🚀 2. เริ่มเทรนโมเดลประเมินความเสี่ยง NCDs...")
predictor = TabularPredictor(label='Diabetes_Risk', eval_metric='accuracy', verbosity=0).fit(
    train_data, presets='medium_quality', time_limit=30
)

# ---------------------------------------------------------
# Step 4: ฟังก์ชันระบบแนะนำเชิงป้องกัน (Action Plan NCDs)
# ---------------------------------------------------------
def generate_ncd_action_plan(row, risk_prob):
    advice = []
    
    is_genetic_risk = (row['SNP_rs456'] == 'TT')
    is_high_sugar = (row['Blood_Sugar'] >= 126)
    is_high_bmi = (row['BMI'] > 25.0)

    if risk_prob > 0.5:
        advice.append("🚨 [แจ้งเตือนระดับสีแดง] มีความเสี่ยงโรคเบาหวาน (Type 2 Diabetes) สูง!")
        if is_genetic_risk:
            advice.append("⚠️ ตรวจพบยีนภาวะดื้ออินซูลิน (TT): ท่านมีความเสี่ยงทางพันธุกรรมเป็นทุนเดิม")
            
        advice.append("💡 [Action Plan ป้องกันก่อนฟอกไต]:")
        if is_high_sugar:
            advice.append(f"   - ค่าน้ำตาลของคุณ ({row['Blood_Sugar']} mg/dL) ทะลุเกณฑ์อันตราย ควรพบแพทย์เพื่อประเมินและงดของหวานเด็ดขาด")
        if is_high_bmi:
            advice.append(f"   - BMI ที่สูง ({row['BMI']}) เร่งให้ยีนเบาหวานทำงาน ควรลดน้ำหนักให้ต่ำกว่าเกณฑ์ 25.0")
        advice.append("   * การปรับพฤติกรรมทันที จะช่วยชะลอความเสื่อมของตับอ่อนและไตได้ในระยะยาว")
                
    else: 
        advice.append("🟢 [สถานะปลอดภัย] ความเสี่ยงโรคเบาหวานต่ำ")
        if is_genetic_risk:
             advice.append("✨ ข่าวดี: แม้คุณจะมียีนเสี่ยง (TT) แต่การที่คุณคุมน้ำตาลและน้ำหนัก (EMR) ได้ดีเยี่ยม ทำให้ยีนไม่แสดงออก ขอให้รักษาพฤติกรรมนี้ไว้ครับ!")
        else:
             advice.append("✨ พันธุกรรมและสุขภาพปัจจุบันอยู่ในเกณฑ์ยอดเยี่ยม")

    return "\n".join(advice)

# ---------------------------------------------------------
# Step 5: ทดสอบ Test Cases แบบเจาะจง NCDs
# ---------------------------------------------------------
print("\n🧪 3. ทดสอบระบบกับคนไข้กลุ่มเสี่ยงเบาหวาน (Preventive Care):")
test_cases = pd.DataFrame({
    'Patient': ['คุณลุง A (เสี่ยง)', 'คุณพี่ B (ปกติ)', 'คุณป้า C (คุมได้)'],
    'SNP_rs123': ['AA', 'GG', 'AG'],
    'SNP_rs456': ['TT', 'CC', 'TT'], # A, C มียีนเบาหวาน
    'SNP_rs789': ['GA', 'GG', 'AA'],
    'Age': [65, 35, 58],              # เข้าสู่สังคมผู้สูงอายุ
    'BMI': [28.5, 21.0, 23.0],        
    'Blood_Sugar': [150, 95, 105]     # A น้ำตาลสูงเกินเกณฑ์ 126
})

features_only = test_cases.drop(columns=['Patient'])
predictions = predictor.predict(features_only)
probabilities = predictor.predict_proba(features_only)

for i in range(len(test_cases)):
    patient_name = test_cases['Patient'].iloc[i]
    prob_risk = probabilities.iloc[i][1]
    
    print("-" * 70)
    print(f"👤 ผู้ป่วย: {patient_name} | อายุ: {test_cases['Age'].iloc[i]} ปี | โอกาสเกิดโรคเบาหวาน: {prob_risk*100:.1f}%")
    print(f"📋 EMR: น้ำตาล {test_cases['Blood_Sugar'].iloc[i]} mg/dL | BMI {test_cases['BMI'].iloc[i]}")
    
    action_plan = generate_ncd_action_plan(test_cases.iloc[i], prob_risk)
    print(f"\n{action_plan}")
print("-" * 70)

# ---------------------------------------------------------
# Step 6: สร้างกราฟ (อัปเดตชื่อให้เข้ากับธีม NCDs)
# ---------------------------------------------------------
print("\n📊 4. กำลังสร้างกราฟเปรียบเทียบโมเดล AI สำหรับคัดกรองเบาหวาน...")
leaderboard = predictor.leaderboard(test_data, silent=True)
leaderboard['Accuracy (%)'] = leaderboard['score_test'] * 100

plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

ax = sns.barplot(x='Accuracy (%)', y='model', data=leaderboard, palette='Blues_r', hue='model', legend=False)

for p in ax.patches:
    width = p.get_width()
    plt.text(width + 1.5, p.get_y() + p.get_height()/2. + 0.1, 
             f'{width:.1f}%', ha="left", va="center", fontweight='bold', color='#333333')

# เปลี่ยนชื่อกราฟให้เจาะจงกลุ่ม NCDs
plt.title('🤖 NCDs (Diabetes) AI Models Performance Comparison', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Accuracy Score (%) in Predicting Type 2 Diabetes', fontsize=12, fontweight='bold')
plt.ylabel('Machine Learning Models', fontsize=12, fontweight='bold')
plt.xlim(0, 115) 

sns.despine(left=True, bottom=True)
plt.tight_layout()

output_leaderboard = 'ncd_model_comparison.png'
plt.savefig(output_leaderboard, dpi=300)
print(f"✅ บันทึกกราฟเรียบร้อยที่ไฟล์: '{output_leaderboard}'")
plt.show()
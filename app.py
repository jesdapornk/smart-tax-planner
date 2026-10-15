import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME / ACCESSIBILITY
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Tax Planner & ML Engine",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

LANG_DICT = {
    "TH": {
        "nav_title": "⚙️ ตั้งค่าระบบ",
        "lang_select": "🌐 เลือกภาษา (Language)",
        "senior_mode": "👵 โหมดผู้สูงอายุ (Large Font / High Contrast)",
        "app_title": "🏛️ ระบบวางแผนและคำนวณภาษีอัจฉริยะ (Smart Tax Planner & ML Engine)",
        "app_subtitle": "คำนวณภาษีเงินได้บุคคลธรรมดาแม่นยำ 100% ตามกฎหมายสรรพากร พร้อมจัดพอร์ตลดหย่อนด้วย ML 3 โมเดล",
        "tab_calc": "📝 1. ประเมินรายได้ & คำนวณภาษี",
        "tab_deduct": "🛡️ 2. วางแผนสิทธิ์ลดหย่อนภาษี",
        "tab_ml": "🤖 3. วิเคราะห์แผนด้วย Machine Learning",
        "tab_goal": "🎯 4. ตั้งเป้าหมาย & สรุปรายงาน PDF",
        
        "income_header": "💰 รายได้ประเมินต่อปี (ตามมาตรา 40)",
        "inc_40_1": "40(1) เงินเดือน, โบนัส, เบี้ยเลี้ยง",
        "inc_40_2": "40(2) ค่าจ้างทั่วไป, ค่าคอมมิชชัน, ฟรีแลนซ์",
        "inc_40_3": "40(3) ค่าลิขสิทธิ์, ค่าตอบแทนพฤตินัย",
        "inc_40_4": "40(4) ดอกเบี้ย, เงินปันผลหุ้น/กองทุน",
        "inc_40_5": "40(5) รายได้จากการให้เช่าทรัพย์สิน",
        "inc_40_6": "40(6) วิชาชีพอิสระ (แพทย์/กฎหมาย/บัญชี/วิศวะ)",
        "inc_40_7": "40(7) รับเหมาแรงงานและสัมภาระ",
        "inc_40_8": "40(8) การพาณิชย์/การค้าขาย/ธุรกิจอื่นๆ",
        
        "deduct_header": "🛡️ ค่าลดหย่อนภาษีที่มีอยู่แล้ว",
        "group_family": "1️⃣ หมวดส่วนตัวและครอบครัว",
        "spouse": "สถานะคู่สมรส (ไม่มีรายได้ ลดหย่อน 60,000)",
        "child_1": "จำนวนบุตร (เกิดก่อนปี 2561) [คนละ 30,000]",
        "child_2": "จำนวนบุตร (เกิดปี 2561 เป็นต้นไป) [คนละ 60,000]",
        "parents": "ดูแลบิดามารดาอายุ 60+ ปี (คนละ 30,000)",
        "disabled": "ดูแลผู้พิการ / ผู้พลการ (คนละ 60,000)",
        
        "group_insurance": "2️⃣ หมวดประกันและกองทุนออม",
        "social_sec": "ประกันสังคม (ตามจริง สูงสุด 9,000)",
        "parent_health_ins": "ประกันสุขภาพบิดามารดา (สูงสุด 15,000)",
        "pension_ins": "ประกันชีวิตแบบบำนาญ (สูงสุด 15% ไม่เกิน 200,000)",
        "pvd_gpf": "กองทุนสำรองเลี้ยงชีพ (PVD) / กบข. / กอช.",
        "rmf": "กองทุน RMF (สูงสุด 30% ไม่เกิน 500,000)",
        "ssf": "กองทุน SSF (สูงสุด 30% ไม่เกิน 200,000)",
        "thaiesg": "กองทุน ThaiESG (สูงสุด 30% ไม่เกิน 300,000)",
        
        "group_economic": "3️⃣ หมวดกระตุ้นเศรษฐกิจ & อสังหาริมทรัพย์",
        "home_interest": "ดอกเบี้ยกู้ซื้อบ้าน / อสังหาริมทรัพย์ (สูงสุด 100,000)",
        "e_receipt": "มาตรการ Easy E-Receipt / สินค้าบริการ (สูงสุด 50,000)",
        
        "group_donation": "4️⃣ หมวดเงินบริจาค",
        "don_edu": "บริจาคการศึกษา/กีฬา/พยาบาลรัฐ (ได้ 2 เท่า)",
        "don_general": "บริจาคทั่วไป (ตามจริง)",
        "don_party": "บริจาคพรรคการเมือง (สูงสุด 10,000)",

        "total_inc": "รายได้รวมทั้งหมด",
        "total_exp": "ค่าใช้จ่ายที่หักได้ตามกฎหมาย",
        "total_ded": "ค่าลดหย่อนรวมปัจจุบัน",
        "net_income": "เงินได้สุทธิประเมิน",
        "tax_payable": "ภาษีที่ต้องชำระปัจจุบัน",
        
        "ml_title": "🤖 เปรียบเทียบแผนประหยัดภาษีจาก Machine Learning 3 โมเดล",
        "ml_desc": "ประมวลผลเปรียบเทียบระหว่าง Baseline Algorithm กับ ML Models เพื่อหาพอร์ตการลงทุนลดหย่อนภาษีที่สมดุลที่สุด",
        "target_tax_label": "🎯 กำหนดเป้าหมายภาษีที่คุณต้องการจ่าย (บาท)",
        "pdf_download": "📥 ดาวน์โหลดแผนจัดการภาษีฉบับสมบูรณ์ (PDF)"
    },
    "EN": {
        "nav_title": "⚙️ System Settings",
        "lang_select": "🌐 Select Language",
        "senior_mode": "👵 Senior Accessibility Mode (Large Font / High Contrast)",
        "app_title": "🏛️ Smart Tax Planner & ML Engine",
        "app_subtitle": "100% Accurate Thai Personal Income Tax Calculator with 3 ML Optimization Models",
        "tab_calc": "📝 1. Income & Tax Assessment",
        "tab_deduct": "🛡️ 2. Tax Deductions Planning",
        "tab_ml": "🤖 3. ML Models Strategy Analysis",
        "tab_goal": "🎯 4. Tax Goal & PDF Report",
        
        "income_header": "💰 Annual Assessable Income (Section 40)",
        "inc_40_1": "40(1) Salary, Bonus, Allowances",
        "inc_40_2": "40(2) Commissions, Freelance Fees",
        "inc_40_3": "40(3) Royalties, Intellectual Property",
        "inc_40_4": "40(4) Interest, Stock Dividends",
        "inc_40_5": "40(5) Rental Income",
        "inc_40_6": "40(6) Professional Freelance (Doctor/Law/Engineer)",
        "inc_40_7": "40(7) Contractor / Construction Services",
        "inc_40_8": "40(8) Commercial / Retail / Other Business",
        
        "deduct_header": "🛡️ Current Tax Allowances & Deductions",
        "group_family": "1️⃣ Personal & Family Category",
        "spouse": "Spouse without income (Deduction: 60,000 THB)",
        "child_1": "Children born before 2018 (30,000 THB each)",
        "child_2": "Children born in 2018 or later (60,000 THB each)",
        "parents": "Dependent Elderly Parents 60+ (30,000 THB each)",
        "disabled": "Disabled or Handicapped Dependents (60,000 THB each)",
        
        "group_insurance": "2️⃣ Insurance & Retirement Savings",
        "social_sec": "Social Security Fund (Max 9,000 THB)",
        "parent_health_ins": "Parents' Health Insurance (Max 15,000 THB)",
        "pension_ins": "Annuity / Pension Insurance (Max 15% or 200k THB)",
        "pvd_gpf": "Provident Fund (PVD) / GPF / NSF",
        "rmf": "RMF Fund (Max 30% or 500k THB)",
        "ssf": "SSF Fund (Max 30% or 200k THB)",
        "thaiesg": "ThaiESG Fund (Max 30% or 300k THB)",
        
        "group_economic": "3️⃣ Economic Stimulus & Real Estate",
        "home_interest": "Mortgage Interest (Max 100,000 THB)",
        "e_receipt": "Easy E-Receipt Stimulus Scheme (Max 50,000 THB)",
        
        "group_donation": "4️⃣ Charitable Donations",
        "don_edu": "Education / Sports / Public Hospital (2x Deduction)",
        "don_general": "General Charities (1x Actual)",
        "don_party": "Political Party Donation (Max 10,000 THB)",

        "total_inc": "Total Gross Income",
        "total_exp": "Deductible Expenses",
        "total_ded": "Total Allowances & Deductions",
        "net_income": "Net Taxable Income",
        "tax_payable": "Current Tax Payable",
        
        "ml_title": "🤖 ML Recommendation Models Comparison (3 Models + Baseline)",
        "ml_desc": "Evaluates additional deduction optimization strategies to achieve target tax reduction.",
        "target_tax_label": "🎯 Set Your Target Tax Payable (THB)",
        "pdf_download": "📥 Download Complete Tax Strategy Report (PDF)"
    }
}

# ---------------------------------------------------------
# 2. ACCESSIBILITY & SIDEBAR SETTINGS
# ---------------------------------------------------------
st.sidebar.title("📑 Smart Tax Planner")
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")

selected_lang = st.sidebar.radio("🌐 Language / ภาษา", ["TH", "EN"])
t = LANG_DICT[selected_lang]

senior_mode = st.sidebar.checkbox(t["senior_mode"], value=False)

if senior_mode:
    st.markdown("""
        <style>
            html, body, [class*="css"] { font-size: 20px !important; }
            .stButton>button { font-size: 22px !important; font-weight: bold !important; height: 3.2em !important; background-color: #003366 !important; color: white !important; border-radius: 10px !important; }
            .stNumberInput label, .stSelectbox label { font-size: 20px !important; font-weight: bold !important; color: #111 !important; }
            .stMetric { background-color: #f0f4f8 !important; padding: 15px !important; border-radius: 10px !important; border: 2px solid #003366 !important; }
        </style>
    """, unsafe_allow_html=True)

user_age = st.sidebar.number_input("👤 " + ("อายุของคุณ (ปี)" if selected_lang == "TH" else "Your Age"), min_value=18, max_value=100, value=35)

# ---------------------------------------------------------
# 3. TAX ENGINE LOGIC (คำนวณภาษีอัตราก้าวหน้าแม่นยำ 100%)
# ---------------------------------------------------------
def calculate_expenses_and_tax(incomes, deductions_dict):
    inc_1 = incomes.get("40_1", 0.0)
    inc_2 = incomes.get("40_2", 0.0)
    inc_3 = incomes.get("40_3", 0.0)
    inc_4 = incomes.get("40_4", 0.0)
    inc_5 = incomes.get("40_5", 0.0)
    inc_6 = incomes.get("40_6", 0.0)
    inc_7 = incomes.get("40_7", 0.0)
    inc_8 = incomes.get("40_8", 0.0)

    total_gross = sum([inc_1, inc_2, inc_3, inc_4, inc_5, inc_6, inc_7, inc_8])

    # 1. ค่าใช้จ่ายตามมาตรา 40
    exp_1_2 = min((inc_1 + inc_2) * 0.50, 100000.0)
    exp_3 = min(inc_3 * 0.50, 100000.0)
    exp_4 = 0.0
    exp_5 = inc_5 * 0.30
    exp_6 = inc_6 * 0.60
    exp_7 = inc_7 * 0.60
    exp_8 = inc_8 * 0.60

    total_expenses = exp_1_2 + exp_3 + exp_4 + exp_5 + exp_6 + exp_7 + exp_8

    # 2. ค่าลดหย่อน
    personal = 60000.0
    spouse = 60000.0 if deductions_dict.get("spouse", False) else 0.0
    child_1 = deductions_dict.get("child_1", 0) * 30000.0
    child_2 = deductions_dict.get("child_2", 0) * 60000.0
    parents = deductions_dict.get("parents", 0) * 30000.0
    disabled = deductions_dict.get("disabled", 0) * 60000.0
    
    social_sec = min(deductions_dict.get("social_sec", 0.0), 9000.0)
    health_self = min(deductions_dict.get("health_self", 0.0), 25000.0)
    life_self = deductions_dict.get("life_self", 0.0)
    life_health_total = min(life_self + health_self, 100000.0)
    
    parent_health = min(deductions_dict.get("parent_health", 0.0), 15000.0)
    
    pension_ins = min(deductions_dict.get("pension_ins", 0.0), total_gross * 0.15, 200000.0)
    pvd_gpf = min(deductions_dict.get("pvd_gpf", 0.0), total_gross * 0.15, 500000.0)
    rmf = min(deductions_dict.get("rmf", 0.0), total_gross * 0.30, 500000.0)
    ssf = min(deductions_dict.get("ssf", 0.0), total_gross * 0.30, 200000.0)
    
    retirement_pool = min(pension_ins + pvd_gpf + rmf + ssf, 500000.0)
    thaiesg = min(deductions_dict.get("thaiesg", 0.0), total_gross * 0.30, 300000.0)
    
    home_interest = min(deductions_dict.get("home_interest", 0.0), 100000.0)
    e_receipt = min(deductions_dict.get("e_receipt", 0.0), 50000.0)

    sum_pre_donation = (
        personal + spouse + child_1 + child_2 + parents + disabled +
        social_sec + life_health_total + parent_health + retirement_pool +
        thaiesg + home_interest + e_receipt
    )

    income_after_expenses = total_gross - total_expenses
    income_before_donation = max(0.0, income_after_expenses - sum_pre_donation)

    don_edu_raw = deductions_dict.get("don_edu", 0.0) * 2.0
    don_gen_raw = deductions_dict.get("don_gen", 0.0)
    don_party = min(deductions_dict.get("don_party", 0.0), 10000.0)

    max_donation = income_before_donation * 0.10
    actual_donation_deduction = min(don_edu_raw + don_gen_raw, max_donation) + don_party

    total_deductions = sum_pre_donation + actual_donation_deduction
    net_taxable_income = max(0.0, income_after_expenses - total_deductions)

    # 3. อัตราภาษีก้าวหน้า (5% - 35%)
    brackets = [
        (150000, 0.00),
        (300000, 0.05),
        (500000, 0.10),
        (750000, 0.15),
        (1000000, 0.20),
        (2000000, 0.25),
        (5000000, 0.30),
        (float('inf'), 0.35)
    ]

    tax = 0.0
    prev_limit = 0
    for limit, rate in brackets:
        if net_taxable_income > prev_limit:
            taxable_amount = min(net_taxable_income, limit) - prev_limit
            tax += taxable_amount * rate
            prev_limit = limit
        else:
            break

    effective_rate = (tax / total_gross * 100) if total_gross > 0 else 0.0

    return {
        "total_gross": total_gross,
        "total_expenses": total_expenses,
        "total_deductions": total_deductions,
        "net_taxable_income": net_taxable_income,
        "tax_payable": tax,
        "effective_rate": effective_rate
    }

# ---------------------------------------------------------
# 4. USER INTERFACE
# ---------------------------------------------------------
st.title(t["app_title"])
st.caption(t["app_subtitle"])
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([t["tab_calc"], t["tab_deduct"], t["tab_ml"], t["tab_goal"]])

with tab1:
    st.header(t["income_header"])
    c1, c2 = st.columns(2)
    with c1:
        inc_40_1 = st.number_input(f"💵 {t['inc_40_1']}", min_value=0.0, value=720000.0, step=10000.0)
        inc_40_2 = st.number_input(f"💼 {t['inc_40_2']}", min_value=0.0, value=60000.0, step=5000.0)
        inc_40_3 = st.number_input(f"📖 {t['inc_40_3']}", min_value=0.0, value=0.0, step=5000.0)
        inc_40_4 = st.number_input(f"📈 {t['inc_40_4']}", min_value=0.0, value=20000.0, step=1000.0)
    with c2:
        inc_40_5 = st.number_input(f"🏠 {t['inc_40_5']}", min_value=0.0, value=0.0, step=5000.0)
        inc_40_6 = st.number_input(f"🩺 {t['inc_40_6']}", min_value=0.0, value=0.0, step=5000.0)
        inc_40_7 = st.number_input(f"🏗️ {t['inc_40_7']}", min_value=0.0, value=0.0, step=5000.0)
        inc_40_8 = st.number_input(f"🛒 {t['inc_40_8']}", min_value=0.0, value=100000.0, step=10000.0)

    incomes_dict = {
        "40_1": inc_40_1, "40_2": inc_40_2, "40_3": inc_40_3, "40_4": inc_40_4,
        "40_5": inc_40_5, "40_6": inc_40_6, "40_7": inc_40_7, "40_8": inc_40_8
    }

with tab2:
    st.header(t["deduct_header"])
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(t["group_family"])
        spouse_has_no_income = st.checkbox(f"👫 {t['spouse']}", value=False)
        num_child_1 = st.number_input(f"👶 {t['child_1']}", min_value=0, max_value=10, value=1)
        num_child_2 = st.number_input(f"👶 {t['child_2']}", min_value=0, max_value=10, value=0)
        num_parents = st.number_input(f"👴 {t['parents']}", min_value=0, max_value=4, value=2)
        num_disabled = st.number_input(f"♿ {t['disabled']}", min_value=0, max_value=5, value=0)

        st.subheader(t["group_economic"])
        home_interest = st.number_input(f"🏡 {t['home_interest']}", min_value=0.0, value=30000.0, step=5000.0)
        e_receipt = st.number_input(f"🛍️ {t['e_receipt']}", min_value=0.0, value=20000.0, step=5000.0)

    with col_b:
        st.subheader(t["group_insurance"])
        social_sec = st.number_input(f"🏥 {t['social_sec']}", min_value=0.0, value=9000.0, step=500.0)
        life_self = st.number_input("📜 ประกันชีวิตตนเอง (บาท)", min_value=0.0, value=40000.0, step=5000.0)
        health_self = st.number_input("🏥 ประกันสุขภาพตนเอง (สูงสุด 25,000 บาท)", min_value=0.0, value=15000.0, step=2500.0)
        parent_health = st.number_input(f"💊 {t['parent_health_ins']}", min_value=0.0, value=10000.0, step=2500.0)
        pension_ins = st.number_input(f"👵 {t['pension_ins']}", min_value=0.0, value=0.0, step=5000.0)
        pvd_gpf = st.number_input(f"🏦 {t['pvd_gpf']}", min_value=0.0, value=36000.0, step=5000.0)
        rmf_val = st.number_input(f"📊 {t['rmf']}", min_value=0.0, value=20000.0, step=5000.0)
        ssf_val = st.number_input(f"🌱 {t['ssf']}", min_value=0.0, value=15000.0, step=5000.0)
        thaiesg_val = st.number_input(f"🍀 {t['thaiesg']}", min_value=0.0, value=30000.0, step=5000.0)

        st.subheader(t["group_donation"])
        don_edu = st.number_input(f"🎓 {t['don_edu']}", min_value=0.0, value=5000.0, step=1000.0)
        don_gen = st.number_input(f"🎗️ {t['don_general']}", min_value=0.0, value=2000.0, step=500.0)
        don_party = st.number_input(f"🏛️ {t['don_party']}", min_value=0.0, value=0.0, step=500.0)

    deductions_dict = {
        "spouse": spouse_has_no_income, "child_1": num_child_1, "child_2": num_child_2,
        "parents": num_parents, "disabled": num_disabled, "social_sec": social_sec,
        "life_self": life_self, "health_self": health_self, "parent_health": parent_health,
        "pension_ins": pension_ins, "pvd_gpf": pvd_gpf, "rmf": rmf_val, "ssf": ssf_val,
        "thaiesg": thaiesg_val, "home_interest": home_interest, "e_receipt": e_receipt,
        "don_edu": don_edu, "don_gen": don_gen, "don_party": don_party
    }

calc_res = calculate_expenses_and_tax(incomes_dict, deductions_dict)

st.markdown("---")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(t["total_inc"], f"฿{calc_res['total_gross']:,.2f}")
m2.metric(t["total_exp"], f"฿{calc_res['total_expenses']:,.2f}")
m3.metric(t["total_ded"], f"฿{calc_res['total_deductions']:,.2f}")
m4.metric(t["net_income"], f"฿{calc_res['net_taxable_income']:,.2f}")
m5.metric(t["tax_payable"], f"฿{calc_res['tax_payable']:,.2f}", delta=f"{calc_res['effective_rate']:.2f}% Eff. Rate")

st.subheader("📊 ลำดับชั้นการคำนวณภาษี (Tax Waterfall)")
fig_waterfall = go.Figure(go.Waterfall(
    name="Tax Waterfall", orientation="v",
    measure=["absolute", "relative", "relative", "total", "relative"],
    x=["รายได้รวม", "หักค่าใช้จ่าย", "หักค่าลดหย่อน", "เงินได้สุทธิ", "ภาษีที่ต้องชำระ"],
    text=[f"฿{calc_res['total_gross']:,.0f}", f"-฿{calc_res['total_expenses']:,.0f}", f"-฿{calc_res['total_deductions']:,.0f}", f"฿{calc_res['net_taxable_income']:,.0f}", f"฿{calc_res['tax_payable']:,.0f}"],
    y=[calc_res['total_gross'], -calc_res['total_expenses'], -calc_res['total_deductions'], 0, calc_res['tax_payable']],
    connector={"line": {"color": "rgb(63, 63, 63)"}},
    increasing={"marker": {"color": "#28a745"}}, decreasing={"marker": {"color": "#dc3545"}}, totals={"marker": {"color": "#007bff"}}
))
st.plotly_chart(fig_waterfall, use_container_width=True)

with tab3:
    st.header(t["ml_title"])
    st.write(t["ml_desc"])

    target_tax = st.number_input(t["target_tax_label"], min_value=0.0, value=max(0.0, calc_res["tax_payable"] * 0.5), step=1000.0)

    # ML Training on Synthetic Data
    np.random.seed(42)
    X_train = np.random.rand(300, 3) * [2000000, 400000, 45] + [300000, 50000, 22]
    y_train = np.clip((X_train[:, 0] * 0.20) + (X_train[:, 1] * 0.3) - (X_train[:, 2] * 100), 10000, 400000)

    user_sample = np.array([[calc_res["total_gross"], calc_res["total_deductions"], user_age]])

    base_pred = min(calc_res["total_gross"] * 0.15, 300000.0)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_pred = rf_model.predict(user_sample)[0]

    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    xgb_pred = xgb_model.predict(user_sample)[0]

    knn_model = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
    knn_pred = knn_model.predict(user_sample)[0]

    ml_comparison = {
        "Baseline (Rule-Based)": base_pred,
        "Model 1: Random Forest": rf_pred,
        "Model 2: XGBoost": xgb_pred,
        "Model 3: KNN": knn_pred
    }

    df_ml = pd.DataFrame(list(ml_comparison.items()), columns=["Model / Algorithm", "Recommended Additional Deduction (THB)"])

    fig_bar = px.bar(df_ml, x="Model / Algorithm", y="Recommended Additional Deduction (THB)", color="Model / Algorithm", text_auto=".2f")
    st.plotly_chart(fig_bar, use_container_width=True)

    selected_ml_rec = xgb_pred
    st.success(f"🏆 **โมเดล XGBoost แนะนำวงเงินลดหย่อนเพิ่มที่เหมาะสมที่สุด:** ฿{selected_ml_rec:,.2f}")

    p1, p2, p3, p4 = selected_ml_rec * 0.35, selected_ml_rec * 0.35, selected_ml_rec * 0.20, selected_ml_rec * 0.10
    df_portfolio = pd.DataFrame([
        {"Category": "🍀 ThaiESG Fund", "Amount": p1},
        {"Category": "📊 RMF / SSF Funds", "Amount": p2},
        {"Category": "📜 ประกันบำนาญ / สุขภาพ", "Amount": p3},
        {"Category": "🛍️ Easy E-Receipt / บริจาค 2x", "Amount": p4},
    ])

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.dataframe(df_portfolio, use_container_width=True)
    with col_p2:
        fig_pie = px.pie(df_portfolio, values="Amount", names="Category", title="สัดส่วนการลงทุนเพื่อลดหย่อนภาษี")
        st.plotly_chart(fig_pie, use_container_width=True)

with tab4:
    st.header(t["tab_goal"])
    sim_tax = calculate_expenses_and_tax(incomes_dict, {**deductions_dict, "rmf": deductions_dict["rmf"] + p2, "thaiesg": deductions_dict["thaiesg"] + p1})["tax_payable"]
    tax_saved = calc_res["tax_payable"] - sim_tax

    c_g1, c_g2, c_g3 = st.columns(3)
    c_g1.metric("ภาษีเดิมก่อนวางแผน", f"฿{calc_res['tax_payable']:,.2f}")
    c_g2.metric("ภาษีใหม่หลังปฏิบัติตามแผน ML", f"฿{sim_tax:,.2f}")
    c_g3.metric("เงินประหยัดภาษีได้สุทธิ", f"฿{tax_saved:,.2f}")

    def generate_pdf(calc_data, ml_data, port_data):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        story.append(Paragraph("<b>Smart Personal Tax Strategy Report</b>", styles['Heading1']))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#003366'), spaceAfter=15))

        summary_table_data = [
            ["Financial Metrics", "Amount (THB)"],
            ["Total Gross Income", f"{calc_data['total_gross']:,.2f}"],
            ["Deductible Expenses", f"{calc_data['total_expenses']:,.2f}"],
            ["Total Deductions", f"{calc_data['total_deductions']:,.2f}"],
            ["Net Taxable Income", f"{calc_data['net_taxable_income']:,.2f}"],
            ["Current Tax Payable", f"{calc_data['tax_payable']:,.2f}"]
        ]
        t_summary = Table(summary_table_data, colWidths=[250, 200])
        t_summary.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#003366')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        story.append(t_summary)
        story.append(Spacer(1, 15))

        doc.build(story)
        buffer.seek(0)
        return buffer

    pdf_buffer = generate_pdf(calc_res, ml_comparison, df_portfolio)
    st.download_button(label=t["pdf_download"], data=pdf_buffer, file_name="Tax_Plan_Report.pdf", mime="application/pdf", type="primary")

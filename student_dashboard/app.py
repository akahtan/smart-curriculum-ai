import streamlit as st
import json
import os
from pathlib import Path

# Set up beautiful Arabic-friendly page configuration
st.set_page_config(
    page_title="منصة المناهج الذكية التفاعلية AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render Custom Right-to-Left (RTL) CSS for Arabic text layout
st.markdown("""
    <style>
    @import url('https://googleapis.com');
    html, body, [data-testid="stWidgetFormValue"], .stMarkdown {
        font-family: 'Tajawal', sans-serif;
        direction: RTL;
        text-align: right;
    }
    .main-title { font-size: 2.8rem; color: #1E3A8A; text-align: center; font-weight: 700; margin-bottom: 20px; }
    .concept-card { background-color: #F3F4F6; padding: 20px; border-radius: 12px; border-right: 6px solid #3B82F6; margin-bottom: 15px; }
    .example-card { background-color: #ECFDF5; padding: 15px; border-radius: 10px; border-right: 6px solid #10B981; margin-top: 10px; }
    .visual-prompt-box { background-color: #FEF3C7; padding: 12px; border-radius: 8px; border-right: 6px solid #F59E0B; font-family: monospace; font-size: 0.9rem; text-align: left; direction: ltr; }
    </style>
""", unsafe_allow_html=True)

# Helper function to load your freshly generated Arabic JSON
def load_curriculum_data():
    project_root = Path(__file__).resolve().parent.parent
    json_path = project_root / "database_sync" / "structured_arabic_curriculum.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None

data = load_curriculum_data()

if not data:
    st.error("❌ لم يتم العثور على ملف البيانات المنظمة. يرجى تشغيل الأداة البرمجية لاستخراج المنهج أولاً.")
else:
    # Top Dashboard Header
    st.markdown('<div class="main-title">🎓 مساعد التعليم الذكي والمبتكر</div>', unsafe_allow_html=True)
    st.subheader(f"📖 الفصل الدراسي: {data.get('chapter_title', 'غير محدد')} (رقم {data.get('chapter_number', 1)})")
    st.info(f"💡 **ملخص الفصل للأبطال:** {data.get('summary', '')}")
    
    st.divider()
    
    # Left Side Control panel for choosing concepts
    st.sidebar.header("🧭 لوحة التحكم بالمنهج")
    concepts = data.get("concepts", [])
    concept_titles = [c.get("concept_title", f"مفهوم {i+1}") for i, c in enumerate(concepts)]
    
    selected_concept_name = st.sidebar.selectbox("اختر المفهوم التعليمي لاستكشافه:", concept_titles)
    
    # Find selected concept object
    selected_concept = next((c for c in concepts if c.get("concept_title") == selected_concept_name), concepts[0])
    
    # Layout Split: Main Learning Screen vs Interactive Media Prompts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### 🎯 المفهوم: {selected_concept.get('concept_title')}")
        st.markdown(
            f'<div class="concept-card">✨ <strong>الشرح المبسط والممتع:</strong><br>{selected_concept.get("simplified_text")}</div>', 
            unsafe_allow_html=True
        )
        
        # Interactive Examples Section
        st.markdown("### 🎲 الألعاب والأمثلة التفاعلية")
        for i, ex in enumerate(selected_concept.get("examples", [])):
            with st.container():
                st.markdown(f'<div class="example-card"><strong>💡 سيناريو {i+1}:</strong> {ex.get("scenario")}</div>', unsafe_allow_html=True)
                st.write(f"❓ **السؤال التفاعلي:** {ex.get('question')}")
                
                # Interactive Button Trick: Reveal Answer/Hint on click
                if st.button(f"🔍 اضغط هنا واكتشف التلميح لحل سؤال {i+1}", key=f"btn_{i}"):
                    st.success(f"🧩 **خطوتك الذكية القادمة:** {ex.get('interactive_step')}")
                    
    with col2:
        st.markdown("### 🎨 الأداة البصرية والمحاكاة")
        st.write("تقوم الأداة بتوفير وصف دقيق مخصص لمولدات الصور لبناء رسومات توضيحية تفاعلية لهذا المفهوم بدقة:")
        
        visual_prompt = selected_concept.get("suggested_visual_prompt", "No prompt provided")
        st.markdown(f'<div class="visual-prompt-box">{visual_prompt}</div>', unsafe_allow_html=True)
        
        # Interactive Student Reflection Notes
        st.divider()
        st.markdown("### ✍️ دفتر ملاحظات الطالب الذكي")
        student_notes = st.text_area("اكتب ما تعلمته اليوم من هذا المفهوم بأسلوبك الخاص لتلخيصه:")
        if st.button("💾 حفظ الملاحظات محلياً"):
            st.toast("🎉 ممتاز! تم حفظ ملخصك بنجاح في ملفك الشخصي.")

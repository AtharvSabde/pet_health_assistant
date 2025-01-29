import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os
import groq
import asyncio
from typing import Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import base64
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Initialize Groq client with API key from environment variable
groq_client = groq.AsyncGroq(
    api_key=os.environ.get("GROQ_API_KEY")
)

async def generate_recommendation(prompt: str) -> Optional[str]:
    """Generate recommendations using Groq API"""
    try:
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are a veterinary expert specializing in precise, evidence-based pet care recommendations. 
                    Your responses must be:
                    1. Highly specific with exact measurements, durations, and frequencies
                    2. Tailored to the exact breed, age, and health conditions
                    3. Based on current veterinary research
                    4. Formatted in clear bullet points
                    5. Free of generic advice
                    
                    Never provide general statements - each point must include specific metrics, measurements, or actionable steps."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.5,  # Reduced for more consistent outputs
            max_tokens=500,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating recommendation: {str(e)}")
        return None

async def generate_diet_recommendation(pet_info: dict) -> str:
    """Generate diet recommendations based on pet information"""
    prompt = f"""
    Generate highly specific diet recommendations for:
    Species: {pet_info['type']}
    Breed: {pet_info['breed']}
    Age: {pet_info['age']} years
    Weight: {pet_info['weight']} kg
    Health Conditions: {pet_info['health_conditions'] or 'None'}
    Allergies: {pet_info['allergies'] or 'None'}
    
    Provide exact measurements and specific products where applicable. Format your response precisely as follows:

    • Daily Caloric Requirements:
      - Exact calories: [number] kcal/day
      - Divided into [number] meals
      - Calorie distribution: [% per meal]
    
    • Macronutrient Breakdown:
      - Protein: [exact %]
      - Fats: [exact %]
      - Carbohydrates: [exact %]
    
    • Recommended Diet:
      - Commercial Foods:
        ∘ [specific brand and product name]
        ∘ [exact portion size in grams]
      - Fresh Foods:
        ∘ [specific ingredient]
        ∘ [exact portion in grams]
    
    • Feeding Schedule:
      - Morning [exact time]: [exact amount in grams]
      - Evening [exact time]: [exact amount in grams]
    
    • Required Supplements:
      - [specific supplement name]
      - [exact dosage]
      - [frequency]
    
    • Foods to Strictly Avoid:
      - [specific food]
      - [reason for avoiding]
    
    • Special Considerations:
      - [specific consideration based on breed/health]
      - [actionable recommendation]
    """
    return await generate_recommendation(prompt)

async def generate_care_recommendation(pet_info: dict) -> str:
    """Generate care recommendations based on pet information"""
    prompt = f"""
    Generate highly specific care recommendations for:
    Species: {pet_info['type']}
    Breed: {pet_info['breed']}
    Age: {pet_info['age']} years
    Weight: {pet_info['weight']} kg
    Health Conditions: {pet_info['health_conditions'] or 'None'}
    
    Provide exact durations, frequencies, and specific products where applicable. Format your response precisely as follows:

    • Exercise Requirements:
      - Daily Duration: [exact minutes]
      - Activity Breakdown:
        ∘ [specific exercise type]: [exact minutes]
        ∘ [intensity level]: [specific indicators]
      - Rest Periods: [exact duration]
    
    • Grooming Protocol:
      - Brushing:
        ∘ [specific brush type]
        ∘ [exact frequency]
        ∘ [technique description]
      - Bathing:
        ∘ [specific shampoo type]
        ∘ [exact frequency]
        ∘ [water temperature]
      - Nail Care:
        ∘ [specific tool]
        ∘ [exact frequency]
    
    • Health Monitoring:
      - Vital Signs:
        ∘ Normal Temperature Range: [exact range]
        ∘ Normal Heart Rate: [exact range]
        ∘ Normal Respiratory Rate: [exact range]
      - Regular Checks:
        ∘ [specific check]
        ∘ [exact frequency]
        ∘ [warning signs]
    
    • Behavioral Monitoring:
      - Key Indicators:
        ∘ [specific behavior]
        ∘ [normal frequency/duration]
        ∘ [warning signs]
    
    • Preventive Care Schedule:
      - Vaccinations:
        ∘ [specific vaccine]
        ∘ [exact timing]
      - Parasite Prevention:
        ∘ [specific product]
        ∘ [exact dosage and frequency]
    
    • Environment Requirements:
      - Temperature: [exact range]
      - Exercise Area: [specific dimensions]
      - Rest Area: [specific requirements]
    """
    return await generate_recommendation(prompt)

async def generate_emergency_guide(pet_info: dict) -> str:
    """Generate emergency care guidelines"""
    prompt = f"""
    Provide emergency care guidelines for a {pet_info['type']}, {pet_info['breed']}.
    Include common emergency situations and immediate actions to take before reaching vet.
    
    Format as bullet points:
    • Signs of Emergency:
      - [sign 1]
    • Immediate Actions:
      - [action 1]
    • When to Contact Vet:
      - [situation 1]
    """
    return await generate_recommendation(prompt)

async def generate_training_tips(pet_info: dict) -> str:
    """Generate pet training recommendations"""
    prompt = f"""
    Provide training tips for a {pet_info['type']}, {pet_info['breed']}, {pet_info['age']} years old.
    Focus on essential commands and behavior training.
    
    Format as bullet points:
    • Basic Commands:
      - [command]: [how to train]
    • Behavior Training:
      - [behavior]: [training method]
    • Common Mistakes:
      - [mistake to avoid]
    """
    return await generate_recommendation(prompt)

async def generate_seasonal_care(pet_info: dict) -> str:
    """Generate seasonal care recommendations"""
    current_month = datetime.now().strftime("%B")
    prompt = f"""
    Provide seasonal care tips for {current_month} for a {pet_info['type']}, {pet_info['breed']}.
    Include specific seasonal challenges and preparations.
    
    Format as bullet points:
    • Seasonal Risks:
      - [risk 1]
    • Preventive Measures:
      - [measure 1]
    • Essential Items:
      - [item 1]
    """
    return await generate_recommendation(prompt)

def create_pdf(pet_info: dict, recommendations: dict) -> BytesIO:
    """Create a PDF report from the recommendations"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph(f"Pet Care Report for {pet_info['name']}", title_style))
    
    # Pet Information
    story.append(Paragraph("Pet Information", styles['Heading2']))
    pet_info_text = f"""
    Type: {pet_info['type']}
    Breed: {pet_info['breed']}
    Age: {pet_info['age']} years
    Weight: {pet_info['weight']} kg
    Health Conditions: {pet_info['health_conditions'] or 'None'}
    Allergies: {pet_info['allergies'] or 'None'}
    """
    story.append(Paragraph(pet_info_text, styles['Normal']))
    story.append(Spacer(1, 12))

    # Add each recommendation section
    for title, content in recommendations.items():
        story.append(Paragraph(title, styles['Heading2']))
        story.append(Paragraph(content, styles['Normal']))
        story.append(Spacer(1, 12))

    # Add timestamp and disclaimer
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Report Generated: {timestamp}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    disclaimer = """
    DISCLAIMER: This report was generated using artificial intelligence. While the recommendations are based on veterinary knowledge, they should not replace professional veterinary advice. Always consult with a qualified veterinarian for your pet's specific needs.
    """
    story.append(Paragraph(disclaimer, styles['Italic']))

    doc.build(story)
    buffer.seek(0)
    return buffer

def get_download_link(buffer):
    """Generate a download link for the PDF"""
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="pet_care_report.pdf">Download PDF Report</a>'

async def analyze_previous_report(current_pet_info: dict, previous_pet_info: dict) -> str:
    """Generate analysis of changes between current and previous report"""
    prompt = f"""
    Compare the following pet information and provide specific recommendations based on changes:
    
    Previous Report ({previous_pet_info['timestamp']}):
    Weight: {previous_pet_info['weight']} kg
    Health: {previous_pet_info['health_conditions'] or 'None'}
    
    Current Information:
    Weight: {current_pet_info['weight']} kg
    Health: {current_pet_info['health_conditions'] or 'None'}
    
    Format your response as bullet points highlighting:
    • Notable Changes
    • Recommendations Based on Changes
    • Areas to Monitor
    """
    return await generate_recommendation(prompt)

def save_pet_data(pet_data: dict):
    """Save pet data to a JSON file"""
    filename = "pet_data.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = []
    
    pet_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data.append(pet_data)
    
    with open(filename, 'w') as f:
        json.dump(data, f)

def main():
    st.title("🐾 Advanced Pet Care Assistant")
    
    # File uploader for previous report
    uploaded_file = st.file_uploader("Upload Previous Report (JSON)", type=['json'])
    previous_pet_info = None
    if uploaded_file:
        previous_pet_info = json.load(uploaded_file)
        st.success("Previous report loaded successfully!")
    
    # Pet Information Input
    with st.sidebar:
        st.header("Pet Information")
        pet_name = st.text_input("Pet's Name")
        pet_type = st.selectbox("Pet Type", ["Dog", "Cat"])
        breed = st.text_input("Breed")
        age = st.number_input("Age (years)", min_value=0.0, max_value=30.0, value=1.0)
        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=100.0, value=5.0)
        health_conditions = st.text_area("Health Conditions (if any)")
        
        st.header("Food Preferences & Allergies")
        favorite_foods = st.text_area("Favorite Foods")
        allergies = st.text_area("Known Allergies")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Care Guide", "Emergency Care", "Training", "Seasonal Care", "Health Records", "Previous Report Analysis"
    ])
    
    pet_info = {
        "name": pet_name,
        "type": pet_type,
        "breed": breed,
        "age": age,
        "weight": weight,
        "health_conditions": health_conditions,
        "favorite_foods": favorite_foods,
        "allergies": allergies,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with tab1:
        if st.button("Generate Care Recommendations", key="care_button"):
            with st.spinner("Generating comprehensive care guide..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                diet_rec = loop.run_until_complete(generate_diet_recommendation(pet_info))
                care_rec = loop.run_until_complete(generate_care_recommendation(pet_info))
                loop.close()
                
                if diet_rec and care_rec:
                    recommendations = {
                        "Diet Recommendations": diet_rec,
                        "Care Recommendations": care_rec
                    }
                    
                    st.header("🍽️ Diet Recommendations")
                    st.markdown(diet_rec)
                    
                    st.header("💝 Care Recommendations")
                    st.markdown(care_rec)
                    
                    # Generate PDF
                    pdf_buffer = create_pdf(pet_info, recommendations)
                    st.markdown(get_download_link(pdf_buffer), unsafe_allow_html=True)
                    
                    save_pet_data(pet_info)

    with tab2:
        if st.button("Get Emergency Guide", key="emergency_button"):
            with st.spinner("Generating emergency care guide..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                emergency_rec = loop.run_until_complete(generate_emergency_guide(pet_info))
                loop.close()
                
                if emergency_rec:
                    st.header("🚨 Emergency Care Guide")
                    st.markdown(emergency_rec)
                    st.markdown("---")
                    st.markdown("### 📞 Emergency Contacts")
                    st.markdown("• Clinic Hours: 24/7")
                    st.markdown("• Emergency Number: [Your Clinic Phone]")
                    st.markdown("• After Hours Care: Available")

    with tab3:
        if st.button("Get Training Tips", key="training_button"):
            with st.spinner("Generating training recommendations..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                training_rec = loop.run_until_complete(generate_training_tips(pet_info))
                loop.close()
                
                if training_rec:
                    st.header("🎓 Training Guide")
                    st.markdown(training_rec)

    with tab4:
        if st.button("Get Seasonal Care Tips", key="seasonal_button"):
            with st.spinner("Generating seasonal care guide..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                seasonal_rec = loop.run_until_complete(generate_seasonal_care(pet_info))
                loop.close()
                
                if seasonal_rec:
                    st.header("🌤️ Seasonal Care Guide")
                    st.markdown(seasonal_rec)

    with tab5:
        st.header("📋 Health Records")
        if st.button("Save Health Record"):
            save_pet_data(pet_info)
            st.success("Health record saved successfully!")
        
        if os.path.exists("pet_data.json"):
            with open("pet_data.json", 'r') as f:
                records = json.load(f)
            if records:
                st.dataframe(pd.DataFrame(records))

    with tab6:
        if previous_pet_info and st.button("Analyze Changes", key="analysis_button"):
            with st.spinner("Analyzing changes from previous report..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                analysis = loop.run_until_complete(analyze_previous_report(pet_info, previous_pet_info))
                loop.close()
                
                if analysis:
                    st.header("📊 Changes Analysis")
                    st.markdown(analysis)

if __name__ == "__main__":
    main()
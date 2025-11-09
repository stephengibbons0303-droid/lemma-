import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime
import requests

def call_openai_api(prompt, api_key, max_tokens=4000):
    """Call OpenAI GPT-4o API for lemmatization."""
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3
            },
            timeout=60  # Add timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            error_data = response.json()
            st.error(f"API Error {response.status_code}: {error_data.get('error', {}).get('message', 'Unknown error')}")
            st.error(f"Full error response: {error_data}")
            return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def generate_lemmatization_prompt(words_list):
    """Create a prompt for GPT-4o to lemmatize words and generate inflectional variations."""
    
    words_str = "\n".join([f"{i+1}. {word}" for i, word in enumerate(words_list)])
    
    prompt = f"""Analyze these vocabulary words/phrases and provide their ROOT FORM and ALL MORPHOLOGICAL VARIATIONS.

VOCABULARY LIST:
{words_str}

REMEMBER:
- For "watch out for" → variations are "watching out for", "watched out for", "watches out for" (NOT "observe", "look out for")
- For "analyze" → variations are "analyzes", "analyzing", "analyzed", "analysing" (NOT "examine", "study")
- These variations help recognize when students use different grammatical forms of the SAME word

OUTPUT FORMAT:
Return ONLY a valid JSON array with no additional text, markdown, or formatting:

[
  {{
    "original": "exact word/phrase from the list",
    "lemma": "root/dictionary form",
    "variations": ["all", "morphological", "variations", "including", "lemma"]
  }}
]

CRITICAL: Response must be ONLY valid JSON. No explanations, no markdown, no code blocks."""

    return prompt

def generate_derivation_prompt(words_list):
    """Create a prompt for GPT-4o to generate derivational variations."""
    
    words_str = "\n".join([f"{i+1}. {word}" for i, word in enumerate(words_list)])
    
    prompt = f"""Analyze these vocabulary words/phrases and provide their ROOT FORM and ALL DERIVATIONAL VARIATIONS.

VOCABULARY LIST:
{words_str}

REMEMBER:
- Derivational variations are words from the SAME ROOT but in a DIFFERENT WORD CLASS.
- Example 1: For "analyze" (verb) -> "analysis" (noun), "analytic" (adjective), "analyst" (noun).
- Example 2: For "beautiful" (adjective) -> "beauty" (noun), "beautify" (verb), "beautifully" (adverb).
- Example 3: For "demonstrate" (verb) -> "demonstration" (noun), "demonstrative" (adjective).
- These variations help recognize when students use a different, but related, word from the same root.

STRICT RULES:
✅ DO: Provide derivational variations (e.g., noun -> verb, verb -> adjective).
❌ DON'T: Provide morphological/inflectional variations (e.g., "analyze" -> "analyzing", "analyzed"). Agent 1 handles this.
❌ DON'T: Provide synonyms (e.g., "analyze" -> "examine").
❌ DON'T: Break up multi-word expressions. (Note: Most MWEs like "watch out for" won't have derivations, which is fine. Just return the lemma.)

OUTPUT FORMAT:
Return ONLY a valid JSON array with no additional text, markdown, or formatting:

[
  {{
    "original": "exact word/phrase from the list",
    "lemma": "root/dictionary form",
    "variations": ["all", "derivational", "variations", "including", "lemma"]
  }}
]

CRITICAL: Response must be ONLY valid JSON. No explanations, no markdown, no code blocks."""

    return prompt

def parse_gpt_response(response_text):
    """Parse GPT's JSON response."""
    try:
        # Remove any markdown code blocks if present
        cleaned = response_text.strip()
        if "```" in cleaned:
            # Extract content between code blocks
            start = cleaned.find("[")
            end = cleaned.rfind("]") + 1
            if start != -1 and end != 0:
                cleaned = cleaned[start:end]
        
        # Parse JSON
        data = json.loads(cleaned)
        return data
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {str(e)}")
        with st.expander("View raw response for debugging"):
            st.code(response_text)
        return None

def process_vocabulary_batch(words_list, api_key, prompt_generator_func, batch_size=50):
    """Process vocabulary in batches using the specified prompt generator."""
    results = []
    total_batches = (len(words_list) + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(words_list), batch_size):
        batch = words_list[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        status_text.text(f"Processing batch {batch_num}/{total_batches} ({len(batch)} words)...")
        
        # Generate prompt using the provided function
        prompt = prompt_generator_func(batch)
        
        # Debug: Show first batch prompt
        if batch_num == 1:
            with st.expander("🔍 Debug: View first batch prompt"):
                st.code(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        
        response = call_openai_api(prompt, api_key, max_tokens=4000)
        
        if response:
            # Debug: Show first batch response
            if batch_num == 1:
                with st.expander("🔍 Debug: View first batch response"):
                    st.code(response[:500] + "..." if len(response) > 500 else response)
            
            parsed_data = parse_gpt_response(response)
            if parsed_data:
                results.extend(parsed_data)
                st.success(f"✅ Batch {batch_num} processed: {len(parsed_data)} items")
            else:
                st.warning(f"⚠️ Batch {batch_num} failed to parse. Skipping...")
        else:
            st.warning(f"⚠️ Batch {batch_num} failed. No response from API.")
        
        # Update progress
        progress = min((i + batch_size) / len(words_list), 1.0)
        progress_bar.progress(progress)
    
    progress_bar.empty()
    status_text.empty()
    
    st.info(f"Total results collected: {len(results)} items")
    
    return results

def create_results_dataframe(results):
    """Convert results to a pandas DataFrame."""
    df_data = []
    for item in results:
        variations_list = item.get("variations", [])
        df_data.append({
            "Original": item.get("original", ""),
            "Lemma (Root)": item.get("lemma", ""),
            "Variations": ", ".join(variations_list),
            "Number of Variations": len(variations_list)
        })
    return pd.DataFrame(df_data)

# Streamlit App
def main():
    st.set_page_config(page_title="Vocabulary Lemmatization Tool", layout="wide", page_icon="📚")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 0;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #666;
            margin-top: 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">📚 Vocabulary Lemmatization Tool</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by GPT-4o for Educational Assessment</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Instructions in expandable section
    with st.expander("ℹ️ About This Tool", expanded=False):
        st.markdown("""
        ### Purpose
        This tool is designed for **educational assessment systems** that need to recognize when students use different 
        forms of vocabulary words without penalizing them.
        
        ### What It Does
        - ✅ **Identifies root forms (lemmas)** of words and phrases
        - ✅ **Generates morphological variations (inflections)** (e.g., "analyze" → "analyzing", "analyzed", "analyzes")
        - ✅ **Generates derivational variations** (e.g., "analyze" → "analysis", "analytic", "analyst")
        - ✅ **Handles multi-word expressions** (e.g., "watch out for" → "watching out for", "watched out for")
        - ✅ **Processes up to 50 words per batch** for efficiency
        
        ### What It Does NOT Do
        - ❌ **No synonyms** - Won't suggest "observe" for "watch" (students must use actual vocabulary words)
        - ❌ **No antonyms** - Won't suggest opposite words
        - ❌ **No semantic relations** - Only morphological variations of the same root
        
        ### Example
        **Vocabulary word:** "analyze"  
        **Inflections:** "analyzing", "analyzed", "analyzes"  
        **Derivations:** "analysis", "analytic", "analyst", "analytical"  
        **Combined:** All of the above
        """)
    
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key. Get one at https://platform.openai.com/api-keys"
    )
    
    if not api_key:
        st.sidebar.warning("⚠️ Please enter your OpenAI API key to use this tool")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📋 Instructions
    1. Enter your OpenAI API key above
    2. Upload a CSV file with vocabulary words
    3. Select the column containing vocabulary
    4. Choose generation mode (inflections/derivations/both)
    5. Click "Generate Variations"
    6. Download the results
    
    ### 📊 Processing
    - **Batch size:** 50 words per batch
    - **Model:** GPT-4o
    - **Context:** Educational assessment
    """)
    
    # File upload
    st.subheader("1️⃣ Upload Your Vocabulary List")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with at least one column containing vocabulary words"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
            
            with st.expander("📄 Preview Original Data", expanded=True):
                st.dataframe(df.head(20), use_container_width=True)
            
            st.markdown("---")
            
            # Column selection
            st.subheader("2️⃣ Select Vocabulary Column")
            columns = df.columns.tolist()
            selected_column = st.selectbox(
                "Which column contains the vocabulary words?",
                options=columns,
                help="Select the column that contains the words/phrases you want to lemmatize"
            )
            
            # Get words list
            words_list = df[selected_column].dropna().astype(str).str.strip().tolist()
            words_list = [w for w in words_list if w and w.lower() != 'nan']  # Remove empty and NaN strings
            
            if len(words_list) == 0:
                st.error("❌ No valid words found in the selected column!")
            else:
                st.info(f"📊 Found **{len(words_list)} vocabulary items** to process")
                
                # Show first few words
                with st.expander("👀 Preview Vocabulary Items"):
                    preview_count = min(10, len(words_list))
                    for i, word in enumerate(words_list[:preview_count], 1):
                        st.text(f"{i}. {word}")
                    if len(words_list) > preview_count:
                        st.text(f"... and {len(words_list) - preview_count} more")
                
                st.markdown("---")
                
                # Generation mode selection
                st.subheader("3️⃣ Select Generation Mode & Run")
                
                mode = st.radio(
                    "Select the variations you want to generate:",
                    (
                        "Inflections only (e.g., analyze → analyzing, analyzed)",
                        "Derivations only (e.g., analyze → analysis, analytic)",
                        "Inflections AND Derivations (Combined)"
                    ),
                    index=0,
                    key="generation_mode",
                    help="Choose which type of variations to generate"
                )
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    process_button = st.button(
                        "🚀 Generate Variations",
                        type="primary",
                        disabled=not api_key,
                        use_container_width=True
                    )
                
                with col2:
                    estimated_batches = (len(words_list) + 49) // 50
                    if mode == "Inflections AND Derivations (Combined)":
                        estimated_batches *= 2  # Double for combined mode
                    st.metric("Batches", estimated_batches)
                
                with col3:
                    estimated_time = estimated_batches * 5
                    st.metric("Est. Time", f"~{estimated_time}s")
                
                if not api_key:
                    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue")
                
                if process_button and api_key:
                    st.markdown("---")
                    
                    final_results = []
                    
                    if mode == "Inflections only (e.g., analyze → analyzing, analyzed)":
                        st.subheader("🔄 Processing Inflections...")
                        with st.spinner("Calling GPT-4o for inflections..."):
                            final_results = process_vocabulary_batch(
                                words_list, api_key, generate_lemmatization_prompt
                            )
                    
                    elif mode == "Derivations only (e.g., analyze → analysis, analytic)":
                        st.subheader("🔄 Processing Derivations...")
                        with st.spinner("Calling GPT-4o for derivations..."):
                            final_results = process_vocabulary_batch(
                                words_list, api_key, generate_derivation_prompt
                            )
                    
                    elif mode == "Inflections AND Derivations (Combined)":
                        # 1. Get Inflections
                        st.subheader("🔄 Processing Inflections (Step 1 of 2)...")
                        with st.spinner("Calling GPT-4o for inflections..."):
                            inflection_results = process_vocabulary_batch(
                                words_list, api_key, generate_lemmatization_prompt
                            )
                        
                        # 2. Get Derivations
                        st.subheader("🔄 Processing Derivations (Step 2 of 2)...")
                        with st.spinner("Calling GPT-4o for derivations..."):
                            derivation_results = process_vocabulary_batch(
                                words_list, api_key, generate_derivation_prompt
                            )
                        
                        # 3. Merge the results
                        st.subheader("🔀 Merging results...")
                        
                        # Use dictionaries for easy lookup
                        inflection_map = {item['original']: item for item in inflection_results}
                        derivation_map = {item['original']: item for item in derivation_results}
                        
                        all_originals = set(inflection_map.keys()) | set(derivation_map.keys())
                        
                        for word in all_originals:
                            inf_item = inflection_map.get(word)
                            der_item = derivation_map.get(word)
                            
                            # Combine variations using sets to remove duplicates
                            inf_vars = set(inf_item['variations']) if inf_item else set()
                            der_vars = set(der_item['variations']) if der_item else set()
                            
                            combined_vars = sorted(list(inf_vars | der_vars))
                            
                            # Use the lemma from the inflection agent as the "primary" lemma
                            lemma = inf_item['lemma'] if inf_item else (der_item['lemma'] if der_item else word)
                            
                            final_results.append({
                                "original": word,
                                "lemma": lemma,
                                "variations": combined_vars
                            })
                    
                    # Display results
                    if final_results and len(final_results) > 0:
                        st.success(f"✅ Successfully processed {len(final_results)} vocabulary items!")
                        
                        # Create DataFrame
                        results_df = create_results_dataframe(final_results)
                        
                        st.markdown("---")
                        
                        # Display results
                        st.subheader("📊 Results")
                        
                        # Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Words/Phrases", len(results_df))
                        with col2:
                            avg_variations = results_df["Number of Variations"].mean()
                            st.metric("Avg. Variations", f"{avg_variations:.1f}")
                        with col3:
                            total_variations = results_df["Number of Variations"].sum()
                            st.metric("Total Variations", total_variations)
                        with col4:
                            max_variations = results_df["Number of Variations"].max()
                            st.metric("Max Variations", max_variations)
                        
                        # Results table
                        st.dataframe(
                            results_df,
                            use_container_width=True,
                            height=400
                        )
                        
                        # Download button
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        mode_suffix = "inflections" if "Inflections only" in mode else "derivations" if "Derivations only" in mode else "combined"
                        
                        st.download_button(
                            label="⬇️ Download Results as CSV",
                            data=csv_data,
                            file_name=f"vocabulary_{mode_suffix}_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Detailed examples
                        with st.expander("🔍 View Detailed Examples"):
                            st.markdown("### First 5 Results:")
                            for idx, row in results_df.head(5).iterrows():
                                st.markdown(f"""
                                **{idx + 1}. {row['Original']}**
                                - **Lemma (Root):** {row['Lemma (Root)']}
                                - **Variations:** {row['Variations']}
                                - **Count:** {row['Number of Variations']} variations
                                """)
                                st.divider()
                    else:
                        st.error("❌ Failed to process vocabulary. Please check your API key and try again.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted.")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Show example
        st.markdown("---")
        st.subheader("💡 Example Usage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📥 Sample Input CSV:**")
            st.code("""Word
analyze
watch out for
running
beautiful
children
take into account
demonstrate""")
        
        with col2:
            st.markdown("**📤 Expected Output:**")
            st.markdown("""
            **Inflections:**
            - **analyze** → analyzing, analyzed, analyzes, analysing
            
            **Derivations:**
            - **analyze** → analysis, analytic, analyst, analytical
            
            **Combined:**
            - All of the above together
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Vocabulary Lemmatization Tool</strong> | Powered by GPT-4o</p>
        <p>Designed for educational assessment and vocabulary evaluation</p>
        <p style='font-size: 0.9rem;'>This tool helps recognize morphological variations without rewarding synonyms</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

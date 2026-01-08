import streamlit as st
import os
import pandas as pd
import json
import re
from datetime import datetime
from collections import Counter

# =============================================================================
# [설정] 경로 및 파일 지정
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__name__))
DATA_ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "Data_Analysis")
LLM_JUDGEMENT_DIR = os.path.join(PROJECT_ROOT, "LLM_Judgement")

ANNOTATION_FILE = os.path.join(DATA_ANALYSIS_DIR, "qualitative_analysis.jsonl")
CLAIM_FILE = os.path.join(DATA_ANALYSIS_DIR, "claims.jsonl")

IMAGE_ROOT = os.path.join(PROJECT_ROOT, "Charts")

JUDGMENT_FILES = [
    os.path.join(LLM_JUDGEMENT_DIR, "claude_responses_judged.jsonl"),
    os.path.join(LLM_JUDGEMENT_DIR, "openai_responses_judged.jsonl"),
    os.path.join(LLM_JUDGEMENT_DIR, "gemini_responses_judged.jsonl"),
]

st.set_page_config(layout="wide", page_title="HCI Research Integrated Tool")

# =============================================================================
# 1. 헬퍼 함수
# =============================================================================
def get_normalized_model_name(json_model_info):
    info = str(json_model_info).lower()
    if "claude" in info: return "claude-sonnet4.5"
    elif "gemini" in info: return "gemini-3"
    elif "gpt" in info or "openai" in info: return "gpt-5.2"
    return info

def generate_unique_id(row):
    csv = str(row.get('input_csv_file', '')).strip()
    model = str(row.get('model_info', '')).strip()
    test = str(row.get('test_type', '')).strip()
    return f"{csv}|{model}|{test}"

# =============================================================================
# 2. Claim 저장 함수 (다중 선택 지원)
# =============================================================================
def save_claim(unique_id, correct_lib, correct_types, original_row):
    """
    correct_types: list of strings (예: ['Bar chart', 'Line chart'])
    """
    record = {
        "unique_id": unique_id,
        "timestamp": datetime.now().isoformat(),
        "corrected_library": correct_lib,
        "corrected_chart_type": correct_types, # 리스트 형태로 저장
        "original_library": original_row.get('judge_library'),
        "original_chart_type": original_row.get('judge_chart_type'),
        "meta_info": {
            "csv": original_row.get('input_csv_file'),
            "model": original_row.get('model_info'),
            "test_type": original_row.get('test_type')
        }
    }
    try:
        with open(CLAIM_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        st.error(f"저장 실패: {e}")
        return False

# =============================================================================
# 3. 데이터 로드 (Claim 반영)
# =============================================================================
@st.cache_data(ttl=0)
def load_all_judgments(file_list):
    # 1. Claim 로드
    claims_map = {}
    if os.path.exists(CLAIM_FILE):
        with open(CLAIM_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        c = json.loads(line)
                        claims_map[c['unique_id']] = {
                            'lib': c['corrected_library'],
                            'type': c['corrected_chart_type'] # 리스트일 수도, 문자열일 수도 있음
                        }
                    except: continue

    # 2. Judgment 로드
    all_rows = []
    for filepath in file_list:
        if not os.path.exists(filepath): continue
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    row = json.loads(line)
                    if 'judgment' in row and isinstance(row['judgment'], dict):
                        for k, v in row['judgment'].items():
                            row[f"judge_{k}"] = v
                    
                    if 'test_type' in row:
                        row['test_type'] = str(row['test_type']).replace('1_default_gemini', '1_default').strip()

                    uid = generate_unique_id(row)
                    row['unique_id'] = uid
                    
                    if uid in claims_map:
                        row['judge_library'] = claims_map[uid]['lib']
                        # 차트 타입 덮어쓰기 (리스트가 들어갈 수 있음)
                        row['judge_chart_type'] = claims_map[uid]['type']
                        row['is_corrected'] = True
                    else:
                        row['is_corrected'] = False

                    all_rows.append(row)
                except: continue

    if not all_rows: return pd.DataFrame()
    return pd.DataFrame(all_rows)

@st.cache_data(ttl=600)
def load_image_data(folder_path):
    all_data = []
    print(os.path.exists(folder_path))
    if not os.path.exists(folder_path): return pd.DataFrame()

    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                name_no_ext = os.path.splitext(f)[0]
                blocks = re.findall(r"\[(.*?)\]", name_no_ext)
                if len(blocks) >= 3:
                    dataset, model, rq = blocks[0], blocks[1], blocks[2]
                    detail_parts = blocks[3:] if len(blocks) > 3 else ["default"]
                    reconstructed_test_type = f"{rq}_{'_'.join(detail_parts)}"
                    
                    all_data.append({
                        "dataset": dataset,
                        "model": model,
                        "rq": rq,
                        "reconstructed_test_type": reconstructed_test_type,
                        "full_detail": "_".join(detail_parts),
                        "category": detail_parts[0] if len(detail_parts) > 1 else "General",
                        "value": "_".join(detail_parts[1:]) if len(detail_parts) > 1 else detail_parts[0],
                        "filename": f,
                        "path": os.path.join(dirpath, f)
                    })
    return pd.DataFrame(all_data)

df_images = load_image_data(IMAGE_ROOT)
df_judgments = load_all_judgments(JUDGMENT_FILES)

if df_images.empty: st.error("이미지를 찾을 수 없습니다."); st.stop()

# =============================================================================
# 4. 매칭 및 통계 함수
# =============================================================================
def find_matching_image(judge_row, image_df):
    if image_df.empty: return None
    target_csv = str(judge_row.get('input_csv_file', '')).replace('.csv', '').strip()
    target_test_type = str(judge_row.get('test_type', '')).strip()
    target_model_filename = get_normalized_model_name(judge_row.get('model_info', ''))

    mask = (
        (image_df['dataset'] == target_csv) & 
        (image_df['reconstructed_test_type'] == target_test_type) &
        (image_df['model'] == target_model_filename)
    )
    matched = image_df[mask]
    return matched.iloc[0] if not matched.empty else None

def find_matching_judgment(image_row, judge_df):
    if judge_df.empty: return None
    mask = (
        (judge_df['input_csv_file'].str.replace('.csv','').str.strip() == image_row['dataset']) &
        (judge_df['test_type'].str.strip() == image_row['reconstructed_test_type']) &
        (judge_df['model_info'].apply(get_normalized_model_name) == image_row['model'])
    )
    matched = judge_df[mask]
    return matched.iloc[0] if not matched.empty else None

def calculate_chart_counts(df):
    """
    데이터프레임의 judge_chart_type 컬럼을 카운트합니다.
    값이 리스트(Multi-chart)인 경우 개별 요소를 분리하여 카운트합니다.
    """
    all_types = []
    for val in df['judge_chart_type'].dropna():
        if isinstance(val, list):
            all_types.extend(val)
        else:
            all_types.append(val)
    
    counts = Counter(all_types)
    count_df = pd.DataFrame.from_dict(counts, orient='index', columns=['Count']).reset_index()
    count_df.columns = ['Chart Type', 'Count']
    return count_df.sort_values(by='Count', ascending=False)

def get_unique_chart_types(df):
    """필터용 고유 차트 타입 목록 추출 (리스트 내부까지 탐색)"""
    all_types = set()
    for val in df['judge_chart_type'].dropna():
        if isinstance(val, list):
            all_types.update(val)
        else:
            all_types.add(val)
    return sorted(list(all_types))

# =============================================================================
# 5. UI 모드 설정
# =============================================================================
st.sidebar.title("🛠️ Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Select Mode:",
    [
        "📊 Library & Chart Gallery (Stats & Audit)",
        "Cross-Model (Compare Models)", 
        "Within-Model (Compare Conditions)"
    ]
)
st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# MODE 1: Library & Chart Gallery (통계/검수)
# -----------------------------------------------------------------------------
if analysis_mode == "📊 Library & Chart Gallery (Stats & Audit)":
    st.title("📊 Judgment Statistics & Audit")
    
    if df_judgments.empty: st.error("No judgment data found."); st.stop()

    # --- [1] 필터 설정 ---
    st.markdown("### 1️⃣ Filter Settings")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        test_types = ["All"] + sorted([str(x) for x in df_judgments['test_type'].unique()])
        sel_test = st.selectbox("Filter by Test Type", test_types)
    with c2:
        models = ["All"] + sorted([str(x) for x in df_judgments['model_info'].unique()])
        sel_model = st.selectbox("Filter by Model", models)
    with c3:
        # 리스트 내부까지 뒤져서 고유 타입 추출
        unique_types = get_unique_chart_types(df_judgments)
        chart_types = ["All"] + unique_types
        sel_chart = st.selectbox("Filter by Chart Type", chart_types)

    # 필터 적용
    filtered_df = df_judgments.copy()
    if sel_test != "All":
        filtered_df = filtered_df[filtered_df['test_type'] == sel_test]
    if sel_model != "All":
        filtered_df = filtered_df[filtered_df['model_info'] == sel_model]
    if sel_chart != "All":
        # 리스트인 경우 포함 여부 확인, 문자열인 경우 일치 여부 확인
        filtered_df = filtered_df[filtered_df['judge_chart_type'].apply(
            lambda x: sel_chart in x if isinstance(x, list) else x == sel_chart
        )]

    st.divider()

    # --- [2] 통계 ---
    st.markdown(f"### 2️⃣ Analysis Results (Images: {len(filtered_df)})")
    
    corrected_count = filtered_df['is_corrected'].sum() if 'is_corrected' in filtered_df.columns else 0
    if corrected_count > 0:
        st.info(f"💡 Statistics currently reflect **{corrected_count}** manual corrections.")
    if not filtered_df.empty:
        sc1, sc2 = st.columns(2)
        with sc1:
            st.caption("**Library Count**")
            lib_counts = filtered_df['judge_library'].value_counts().reset_index()
            lib_counts.columns = ['Library', 'Count']
            st.dataframe(lib_counts, use_container_width=True, hide_index=True)
        with sc2:
            st.caption("**Chart Type Count (Multi-counted)**")
            # [수정됨] 차트 타입 카운팅 함수 사용 (중복 카운트 지원)
            chart_counts = calculate_chart_counts(filtered_df)
            st.dataframe(chart_counts, use_container_width=True, hide_index=True)
            
        st.write("")
        st.markdown("#### 📚 Chart Types by Library")
        active_libs = sorted(filtered_df['judge_library'].dropna().unique())
        if active_libs:
            cols = st.columns(3)
            for idx, lib in enumerate(active_libs):
                with cols[idx % 3]:
                    st.markdown(f"**🔹 {lib}**")
                    lib_df = filtered_df[filtered_df['judge_library'] == lib]
                    # 라이브러리별로도 중복 카운트 적용
                    type_counts = calculate_chart_counts(lib_df)
                    st.dataframe(type_counts, use_container_width=True, hide_index=True)
    else:
        st.warning("조건에 맞는 데이터가 없습니다.")

    st.divider()

    # --- [3] 갤러리 및 수정 ---
    st.markdown("### 3️⃣ Audit Gallery")
    
    # 수정용 옵션 리스트 (전체 고유값 기준)
    ALL_LIBS = sorted(df_judgments['judge_library'].dropna().unique().tolist())
    ALL_TYPES = get_unique_chart_types(df_judgments)
    
    page_size = 10
    if len(filtered_df) > 0:
        pages = (len(filtered_df) // page_size) + 1
        page = st.number_input("Page", 1, pages, 1)
        view_df = filtered_df.iloc[(page-1)*page_size : page*page_size]

        for idx, row in view_df.iterrows():
            img_info = find_matching_image(row, df_images)
            
            with st.container():
                col_img, col_info = st.columns([1.2, 2])
                
                with col_img:
                    if img_info is not None:
                        st.image(img_info['path'], caption=img_info['filename'], use_container_width=True)
                    else:
                        st.error("🖼️ Image Matching Failed")
                        st.caption(f"CSV: {row.get('input_csv_file')}")

                with col_info:
                    status_badge = "✅ Corrected" if row.get('is_corrected') else "🤖 LLM Original"
                    
                    st.caption(f"{status_badge} | {row.get('model_info')} | {row.get('test_type')}")
                    
                    # 차트 타입 표시 (리스트면 예쁘게 조인)
                    display_type = row.get('judge_chart_type')
                    if isinstance(display_type, list):
                        display_type_str = ", ".join(display_type)
                    else:
                        display_type_str = str(display_type)
                        
                    st.subheader(f"{display_type_str}")
                    st.caption(f"Library: {row.get('judge_library')}")
                    
                    st.info(f"**Reasoning:** {row.get('judge_reasoning')}")
                    
                    with st.expander("Show Code"):
                        st.code(row.get('response', ''), language='python')
                    
                    # [수정됨] Multi-select 지원 UI
                    with st.expander(f"🛠️ Fix / Claim (ID: ...{str(row['unique_id'])[-6:]})"):
                        with st.form(key=f"claim_form_{row['unique_id']}"):
                            st.write("If the LLM judgment is incorrect, select the correct values below.")
                            
                            curr_lib = row.get('judge_library')
                            curr_val_raw = row.get('judge_chart_type')
                            
                            # 멀티셀렉트의 초기값 설정 (리스트화)
                            if isinstance(curr_val_raw, list):
                                default_types = [t for t in curr_val_raw if t in ALL_TYPES]
                            else:
                                default_types = [curr_val_raw] if curr_val_raw in ALL_TYPES else []

                            # UI
                            new_lib = st.selectbox("Correct Library", ALL_LIBS, index=ALL_LIBS.index(curr_lib) if curr_lib in ALL_LIBS else 0)
                            
                            # Multiselect 사용
                            new_types = st.multiselect("Correct Chart Types", ALL_TYPES, default=default_types)
                            
                            if st.form_submit_button("💾 Save Correction"):
                                
                                # 값 유효성 체크
                                if not new_types:
                                    st.error("적어도 하나의 차트 타입을 선택해야 합니다.")
                                else:
                                    success = save_claim(row['unique_id'], new_lib, new_types, row)
                                    if success:
                                        st.success("수정 완료! 새로고침 중...")
                                        st.cache_data.clear()
                                        st.rerun()
                st.divider()
    else:
        st.info("No data available.")

# -----------------------------------------------------------------------------
# MODE 2 & 3: Qualitative Analysis (기존 유지)
# -----------------------------------------------------------------------------
else:
    datasets = sorted(df_images['dataset'].unique())
    selected_dataset = st.sidebar.selectbox("1. Dataset", datasets)
    subset_ds = df_images[df_images['dataset'] == selected_dataset]
    rqs = sorted(subset_ds['rq'].unique())
    selected_rq = st.sidebar.selectbox("2. RQ (Test Type)", rqs)
    subset_rq = subset_ds[subset_ds['rq'] == selected_rq]
    
    target_df = pd.DataFrame()

    if analysis_mode.startswith("Cross"):
        details = sorted(subset_rq['full_detail'].unique())
        selected_detail = st.sidebar.selectbox("3. Detail Condition", details)
        target_df = subset_rq[subset_rq['full_detail'] == selected_detail]
        auto_context_tag = f"RQ{selected_rq}-Model-{selected_detail}"
    else:
        models = sorted(subset_rq['model'].unique())
        selected_model = st.sidebar.selectbox("3. Model", models)
        subset_model = subset_rq[subset_rq['model'] == selected_model]
        categories = sorted(subset_model['category'].unique())
        if "General" in categories and len(categories)>1: categories.remove("General")
        selected_category = st.sidebar.selectbox("4. Condition Category", categories)
        target_df = subset_model[subset_model['category'] == selected_category]
        def custom_sort_key(val):
            # 원하는 순서를 딕셔너리로 정의 (숫자가 작을수록 먼저 나옴)
            order_map = {
                "L": 0, "Low": 0, "beginner": 0,
                "M": 1, "Mid": 1, "intermediate": 1,
                "H": 2, "High": 2, "expert": 2,
                "eng": 0, "kor": 1, "chi": 2, # 언어 순서 예시
                "default": 0
            }
            # 정의되지 않은 값은 99를 반환하여 맨 뒤로 보냄
            return order_map.get(str(val), 99)

        # 'value' 컬럼 기준으로 정렬 적용
        if 'value' in target_df.columns:
            target_df['sort_temp'] = target_df['value'].apply(custom_sort_key)
            target_df = target_df.sort_values('sort_temp').drop(columns=['sort_temp'])
        # =========================================================
        
        auto_context_tag = f"RQ{selected_rq}-{selected_model}-{selected_category}"

    st.title(f"📊 {selected_dataset}")
    st.caption(f"Context: {auto_context_tag}")
    
    if not target_df.empty:
        cols = st.columns(len(target_df))
        for idx, (_, row) in enumerate(target_df.iterrows()):
            with cols[idx % len(cols)]:
                label = row['model'] if analysis_mode.startswith("Cross") else row['value']
                st.image(row['path'], caption=label)
                
                matched_judge = find_matching_judgment(row, df_judgments)
                if matched_judge is not None:
                    is_fixed = matched_judge.get('is_corrected', False)
                    badge_color = "green" if is_fixed else "gray"
                    badge_text = "✅ Fixed" if is_fixed else "🤖 Auto"
                    
                    # 리스트 타입 처리
                    ctype = matched_judge.get('judge_chart_type', 'N/A')
                    if isinstance(ctype, list):
                        ctype_str = ", ".join(ctype)
                    else:
                        ctype_str = str(ctype)

                    st.markdown(f"""
                    <div style="font-size:0.8em; color:{badge_color}; border:1px solid {badge_color}; padding:2px; border-radius:4px; display:inline-block;">
                        {badge_text}
                    </div>
                    <div style="font-size:0.8em; color:gray; margin-top:4px;">
                        <b>Type:</b> {ctype_str}<br>
                        <b>Lib:</b> {matched_judge.get('judge_library', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
                    
    # Note 작성 폼 (기존 유지)
    st.markdown("---")
    st.header(f"📝 Analysis Note")
    with st.form("analysis_form"):
        c1, c2 = st.columns([1, 2])
        with c1:
            observation_tags = st.multiselect("Select Patterns:", ["Significant Difference", "No Significant Difference", "Inconsistent Behavior", "Format Compliance"])
        with c2:
            note = st.text_area("Insight:", height=100)
        if st.form_submit_button("💾 Save Insight"):
            record = {"timestamp": datetime.now().isoformat(), "context": auto_context_tag, "tags": observation_tags, "note": note}
            with open(ANNOTATION_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            st.success("✅ Saved!")